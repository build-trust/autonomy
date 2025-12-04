import asyncio
import psycopg
from typing import Optional, Dict, List, Tuple, Set, TYPE_CHECKING
from collections import defaultdict
from os import environ
from urllib.parse import quote
from psycopg.rows import dict_row
from ..logs import get_logger

if TYPE_CHECKING:
  from ..models.model import Model

logger = get_logger("memory")

MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT = 100
MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT = 8000


# =============================================================================
# Tool Pair Detection Utilities
# =============================================================================
# These functions help preserve the pairing between tool_use (assistant messages
# with tool_calls) and tool_result (messages with role: "tool") required by
# Claude's API. Breaking this pairing causes "Expected toolResult blocks" errors.


def get_tool_use_ids(message: dict) -> List[str]:
  """
  Extract tool_call IDs from an assistant message with tool_calls.

  Args:
    message: A message dict that may contain tool_calls

  Returns:
    List of tool_call IDs, or empty list if none found
  """
  if message.get("role") == "assistant" and "tool_calls" in message:
    return [tc.get("id") for tc in message.get("tool_calls", []) if tc.get("id")]
  return []


def get_tool_result_id(message: dict) -> Optional[str]:
  """
  Extract tool_call_id from a tool result message.

  Args:
    message: A message dict that may be a tool result

  Returns:
    The tool_call_id if this is a tool result message, None otherwise
  """
  if message.get("role") == "tool":
    return message.get("tool_call_id")
  return None


def find_tool_pair_boundary(messages: List[dict], start_index: int) -> int:
  """
  Find a safe boundary that doesn't split tool pairs.

  Given a starting index, move backwards if needed to include any tool_use
  messages whose results appear at or after start_index. This prevents
  orphaned tool results.

  Args:
    messages: List of conversation messages
    start_index: The proposed starting index for the verbatim window

  Returns:
    Adjusted index that won't leave orphaned tool results (may be <= start_index)
  """
  if start_index <= 0 or start_index >= len(messages):
    return start_index

  # Build a map of tool_call_id -> index of the assistant message containing it
  tool_use_index: Dict[str, int] = {}
  for i, msg in enumerate(messages):
    for tool_id in get_tool_use_ids(msg):
      tool_use_index[tool_id] = i

  # Find the minimum index we need to include to avoid orphaned tool results
  min_required_index = start_index

  for i in range(start_index, len(messages)):
    tool_result_id = get_tool_result_id(messages[i])
    if tool_result_id and tool_result_id in tool_use_index:
      tool_use_idx = tool_use_index[tool_result_id]
      if tool_use_idx < min_required_index:
        min_required_index = tool_use_idx
        logger.debug(f"Tool pair boundary: tool result at index {i} requires tool_use at index {tool_use_idx}")

  if min_required_index < start_index:
    logger.debug(f"Adjusted boundary from {start_index} to {min_required_index} to preserve tool pairs")

  return min_required_index


def find_safe_trim_count(messages: List[dict], requested_trim_count: int) -> int:
  """
  Find the maximum number of messages that can be safely trimmed from the front.

  This ensures we don't trim a tool_use message while leaving its tool_result,
  or vice versa.

  Args:
    messages: List of conversation messages
    requested_trim_count: How many messages we want to trim

  Returns:
    The actual number of messages that can be safely trimmed (may be less than requested)
  """
  if requested_trim_count <= 0 or not messages:
    return 0

  # Build maps of tool pairs
  # tool_call_id -> index of assistant message with tool_calls
  tool_use_index: Dict[str, int] = {}
  # tool_call_id -> index of tool result message
  tool_result_index: Dict[str, int] = {}

  for i, msg in enumerate(messages):
    for tool_id in get_tool_use_ids(msg):
      tool_use_index[tool_id] = i
    tool_result_id = get_tool_result_id(msg)
    if tool_result_id:
      tool_result_index[tool_result_id] = i

  # Find all tool_call_ids that have both a use and a result
  paired_tool_ids = set(tool_use_index.keys()) & set(tool_result_index.keys())

  # Find the maximum safe trim count
  safe_trim_count = requested_trim_count

  for tool_id in paired_tool_ids:
    use_idx = tool_use_index[tool_id]
    result_idx = tool_result_index[tool_id]

    # If trimming would remove the tool_use but keep the result, we can't trim that far
    if use_idx < safe_trim_count <= result_idx:
      # We can only trim up to (but not including) the tool_use
      safe_trim_count = min(safe_trim_count, use_idx)
      logger.debug(f"Safe trim reduced to {safe_trim_count}: tool_use at {use_idx}, result at {result_idx}")

    # If trimming would remove the result but keep the tool_use, we can't trim that far
    # (This shouldn't normally happen as results come after uses, but handle it anyway)
    if result_idx < safe_trim_count <= use_idx:
      safe_trim_count = min(safe_trim_count, result_idx)
      logger.debug(f"Safe trim reduced to {safe_trim_count}: result at {result_idx}, tool_use at {use_idx}")

  return safe_trim_count


def validate_tool_pairing(messages: List[dict]) -> Tuple[bool, Optional[str]]:
  """
  Validate that all tool_use blocks have corresponding tool_result blocks.

  Args:
    messages: List of conversation messages to validate

  Returns:
    Tuple of (is_valid, error_message). error_message is None if valid.
  """
  tool_use_ids: Set[str] = set()
  tool_result_ids: Set[str] = set()

  for msg in messages:
    for tool_id in get_tool_use_ids(msg):
      tool_use_ids.add(tool_id)
    tool_result_id = get_tool_result_id(msg)
    if tool_result_id:
      tool_result_ids.add(tool_result_id)

  # Check for orphaned tool results (results without corresponding uses)
  orphaned_results = tool_result_ids - tool_use_ids
  if orphaned_results:
    return False, f"Orphaned tool results (no matching tool_use): {orphaned_results}"

  # Check for orphaned tool uses (uses without corresponding results)
  # Note: This is less critical as the model can still process, but log it
  orphaned_uses = tool_use_ids - tool_result_ids
  if orphaned_uses:
    # This might be okay if tool calls are still pending, so just debug log
    logger.debug(f"Tool calls without results (may be pending): {orphaned_uses}")

  return True, None


class Memory:
  """
  Manages conversation messages with two-tier storage:
  - Short-term memory: Recent messages in memory (for fast context retrieval)
  - Long-term memory: All messages persisted in PostgreSQL (optional)
  """

  def __init__(
    self,
    max_messages_in_short_term_memory: int = MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT,
    max_tokens_in_short_term_memory: Optional[int] = MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT,
    model: Optional["Model"] = None,  # For token counting
    enable_long_term_memory: bool = False,
  ):
    # Validate configuration
    if max_messages_in_short_term_memory < 1:
      raise ValueError("max_messages_in_short_term_memory must be at least 1")
    if max_tokens_in_short_term_memory is not None and max_tokens_in_short_term_memory < 100:
      raise ValueError("max_tokens_in_short_term_memory must be at least 100")

    self.tenant_id = None
    self.db_url = None
    self.connection = None
    self.lock = asyncio.Lock()

    # Configuration
    self.max_messages_in_short_term_memory = max_messages_in_short_term_memory
    self.max_tokens_in_short_term_memory = max_tokens_in_short_term_memory
    self.model = model
    self.enable_long_term_memory = enable_long_term_memory

    # In-memory storage
    # Key: (scope, conversation), Value: list of message dicts
    self.short_term_memory: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    # System instructions (prepended to all message lists)
    self.instructions = []

    # Metrics tracking
    self.metrics = {
      "messages_added": 0,
      "messages_trimmed": 0,
      "db_writes": 0,
      "db_reads": 0,
      "active_conversations": 0,
      "batch_operations": 0,
    }

  async def initialize_database(self, max_retries: int = 3):
    """Initialize database connection with retry logic."""
    if not self.enable_long_term_memory:
      logger.info("Long-term memory disabled, running in short-term memory-only mode")
      return

    if "OCKAM_DATABASE_INSTANCE" not in environ:
      logger.info("No database configured, running in short-term memory-only mode")
      return

    db_instance = environ.get("OCKAM_DATABASE_INSTANCE")
    self.tenant_id = environ.get("OCKAM_DATABASE_USER")
    password = environ.get("OCKAM_DATABASE_PASSWORD")

    self.db_url = f"postgresql://{self.tenant_id}:{quote(password)}@{db_instance}"
    db_url_anonymized = f"postgresql://*****:*****@{db_instance}"
    logger.info("Connecting to database at %s", db_url_anonymized)

    # Connection with retry logic and keepalive settings
    for attempt in range(max_retries):
      try:
        self.connection = await psycopg.AsyncConnection.connect(
          self.db_url, row_factory=dict_row, keepalives=1, keepalives_idle=30, keepalives_interval=5, keepalives_count=5
        )
        break
      except Exception as e:
        if attempt == max_retries - 1:
          logger.error(f"Failed to connect after {max_retries} attempts: {e}")
          raise
        logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
        await asyncio.sleep(2**attempt)  # Exponential backoff

    # Create index if it doesn't exist
    async with self.connection.cursor() as cur:
      await cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversation_lookup
        ON conversation (tenant_id, scope, conversation)
      """)
      await self.connection.commit()

    # Load existing messages into short-term memory
    await self._load_short_term_memory()

  async def _load_short_term_memory(self):
    """Load recent messages from database into short-term memory (optimized)."""
    if not self.connection:
      return

    async with self.connection.cursor() as cur:
      # Only load the most recent N messages per conversation
      # Using window function for efficient loading
      await cur.execute(
        """
        WITH ranked_messages AS (
          SELECT scope, conversation, message,
                ROW_NUMBER() OVER (
                  PARTITION BY scope, conversation
                  ORDER BY id DESC
                ) as rn
          FROM conversation
          WHERE tenant_id = %s
        )
        SELECT scope, conversation, message
        FROM ranked_messages
        WHERE rn <= %s
        ORDER BY scope, conversation, rn DESC
      """,
        (self.tenant_id, self.max_messages_in_short_term_memory),
      )

      rows = await cur.fetchall()
      self.metrics["db_reads"] += len(rows)

      # Group messages by conversation and reverse order
      import json

      conversations = defaultdict(list)
      for row in rows:
        message = json.loads(row["message"])
        scope = row["scope"]
        conversation = row["conversation"]
        conversations[(scope, conversation)].append(message)

      # Reverse each conversation's messages (oldest first)
      for key, messages in conversations.items():
        self.short_term_memory[key] = list(reversed(messages))

      logger.info(f"Loaded {len(rows)} messages from {len(conversations)} conversations")

  def set_instructions(self, instructions: dict):
    """Set system instructions (prepended to all conversations)."""
    self.instructions = [instructions]

  async def add_message(self, scope: str, conversation: str, message: dict):
    """
    Add a message to short-term memory and optionally to long-term memory (database).
    Automatically trims short-term memory if needed.
    Adds timestamp if not present.
    """
    import time

    # Add timestamp if not present
    if "timestamp" not in message:
      message["timestamp"] = time.time()

    async with self.lock:
      # Add to short-term memory
      self.short_term_memory[(scope, conversation)].append(message)
      self.metrics["messages_added"] += 1

      # Persist to long-term memory (database) if enabled
      if self.enable_long_term_memory and self.connection:
        import json

        try:
          async with self.connection.cursor() as cur:
            await cur.execute(
              "INSERT INTO conversation (tenant_id, scope, conversation, message) VALUES (%s, %s, %s, %s)",
              (self.tenant_id, scope, conversation, json.dumps(message)),
            )
            await self.connection.commit()
            self.metrics["db_writes"] += 1
        except Exception as e:
          logger.error(f"Failed to persist message to database: {e}")
          # Continue even if DB write fails (memory still has it)

      # Trim if necessary
      await self._trim_short_term_memory(scope, conversation)

  async def add_messages(self, scope: str, conversation: str, messages: List[dict]):
    """
    Add multiple messages efficiently in a batch.
    Automatically trims short-term memory if needed.
    """
    import time

    # Add timestamps if not present
    for message in messages:
      if "timestamp" not in message:
        message["timestamp"] = time.time()

    async with self.lock:
      # Add to short-term memory
      self.short_term_memory[(scope, conversation)].extend(messages)
      self.metrics["messages_added"] += len(messages)
      self.metrics["batch_operations"] += 1

      # Persist to long-term memory (database) in batch if enabled
      if self.enable_long_term_memory and self.connection:
        import json

        try:
          async with self.connection.cursor() as cur:
            # Use executemany for batch insert
            await cur.executemany(
              "INSERT INTO conversation (tenant_id, scope, conversation, message) VALUES (%s, %s, %s, %s)",
              [(self.tenant_id, scope, conversation, json.dumps(msg)) for msg in messages],
            )
            await self.connection.commit()
            self.metrics["db_writes"] += len(messages)
        except Exception as e:
          logger.error(f"Failed to persist batch messages to database: {e}")
          # Continue even if DB write fails

      # Trim if necessary
      await self._trim_short_term_memory(scope, conversation)

  async def get_messages(self, scope: str, conversation: str) -> List[dict]:
    """
    Get messages for a conversation (instructions + short-term memory).
    This is the primary method used by agents.
    """
    async with self.lock:
      return self.instructions + self.short_term_memory[(scope, conversation)]

  async def get_messages_only(self, scope: str, conversation: str) -> List[dict]:
    """Get only conversation messages (no instructions) from short-term memory."""
    async with self.lock:
      return self.short_term_memory[(scope, conversation)].copy()

  async def get_all_messages(self, scope: str, conversation: str, limit: Optional[int] = None) -> List[dict]:
    """
    Retrieve all messages for a conversation from long-term memory (database).
    Use this for analysis, export, or when you need all messages.
    Falls back to short-term memory if long-term memory is not enabled.
    """
    if not self.enable_long_term_memory or not self.connection:
      # Fallback to short-term memory if no long-term memory
      return await self.get_messages_only(scope, conversation)

    import json

    async with self.connection.cursor() as cur:
      if limit:
        await cur.execute(
          "SELECT message FROM conversation WHERE tenant_id = %s AND scope = %s AND conversation = %s ORDER BY id LIMIT %s",
          (self.tenant_id, scope, conversation, limit),
        )
      else:
        await cur.execute(
          "SELECT message FROM conversation WHERE tenant_id = %s AND scope = %s AND conversation = %s ORDER BY id",
          (self.tenant_id, scope, conversation),
        )
      rows = await cur.fetchall()
      self.metrics["db_reads"] += len(rows)
      return [json.loads(row["message"]) for row in rows]

  async def get_message_count(self, scope: str, conversation: str) -> int:
    """Get total number of messages in a conversation (from long-term memory if enabled)."""
    if not self.enable_long_term_memory or not self.connection:
      async with self.lock:
        return len(self.short_term_memory[(scope, conversation)])

    async with self.connection.cursor() as cur:
      await cur.execute(
        "SELECT COUNT(*) as count FROM conversation WHERE tenant_id = %s AND scope = %s AND conversation = %s",
        (self.tenant_id, scope, conversation),
      )
      row = await cur.fetchone()
      return row["count"] if row else 0

  async def _estimate_tokens(self, messages: List[dict]) -> int:
    """Estimate token count for a list of messages."""
    if not self.model or not messages:
      # Rough estimate: 4 characters per token
      total_chars = sum(len(str(msg)) for msg in messages)
      return total_chars // 4

    # Check if any message has structured content (dict)
    # If so, skip model token counting to avoid warnings from litellm
    has_structured_content = any(isinstance(msg.get("content"), dict) for msg in messages)

    if has_structured_content:
      # Use character estimate for structured content
      total_chars = 0
      for msg in messages:
        if isinstance(msg.get("content"), dict):
          # Extract text from structured content like {'text': '...', 'type': 'text'}
          if "text" in msg["content"]:
            total_chars += len(msg["content"]["text"])
          else:
            total_chars += len(str(msg["content"]))
        else:
          total_chars += len(str(msg.get("content", "")))
      return total_chars // 4

    try:
      # Use model's token counter for plain string content
      return self.model.count_tokens(messages)
    except Exception as e:
      logger.debug(f"Token counting failed: {e}, using character estimate")
      total_chars = sum(len(str(msg)) for msg in messages)
      return total_chars // 4

  async def _should_trim_short_term_memory(self, scope: str, conversation: str) -> bool:
    """Check if short-term memory needs trimming."""
    messages = self.short_term_memory[(scope, conversation)]

    # Check message count limit
    if len(messages) > self.max_messages_in_short_term_memory:
      return True

    # Check token limit (if configured and model available)
    if self.max_tokens_in_short_term_memory:
      token_count = await self._estimate_tokens(messages)
      if token_count > self.max_tokens_in_short_term_memory:
        return True

    return False

  async def _trim_short_term_memory(self, scope: str, conversation: str):
    """
    Trim short-term memory to stay within configured limits.
    Removes oldest messages first (FIFO), but preserves tool_use/tool_result pairs.

    This method ensures that when trimming messages, we don't break the pairing
    between tool_use (assistant messages with tool_calls) and tool_result
    (messages with role: "tool"), which would cause Claude API errors.
    """
    messages = self.short_term_memory[(scope, conversation)]
    initial_count = len(messages)

    if self.max_tokens_in_short_term_memory and len(messages) > 1:
      # Token-aware trimming with tool pair preservation
      while await self._estimate_tokens(messages) > self.max_tokens_in_short_term_memory and len(messages) > 1:
        # Calculate safe trim count (at least 1 if possible)
        safe_trim = find_safe_trim_count(messages, 1)
        if safe_trim > 0:
          messages.pop(0)
          self.metrics["messages_trimmed"] += 1
          logger.debug(f"Trimmed message from short-term memory (token limit) for {scope}/{conversation}")
        else:
          # Can't trim without breaking tool pairs, stop trimming
          logger.debug(f"Cannot trim further without breaking tool pairs (token limit) for {scope}/{conversation}")
          break

    # Count-based trimming with tool pair preservation
    if len(messages) > self.max_messages_in_short_term_memory:
      # Calculate how many messages we need to trim
      requested_trim = len(messages) - self.max_messages_in_short_term_memory

      # Find safe trim count that preserves tool pairs
      safe_trim = find_safe_trim_count(messages, requested_trim)

      if safe_trim < requested_trim:
        logger.debug(
          f"Reduced trim from {requested_trim} to {safe_trim} to preserve tool pairs for {scope}/{conversation}"
        )

      # Trim the safe number of messages
      for _ in range(safe_trim):
        messages.pop(0)
        self.metrics["messages_trimmed"] += 1

      if safe_trim > 0:
        logger.debug(f"Trimmed {safe_trim} messages from short-term memory (count limit) for {scope}/{conversation}")

    if len(messages) < initial_count:
      logger.debug(f"Trimmed {initial_count - len(messages)} messages total from {scope}/{conversation}")

      # Validate tool pairing after trimming (debug safety check)
      is_valid, error = validate_tool_pairing(messages)
      if not is_valid:
        logger.error(f"Tool pairing validation failed after trimming for {scope}/{conversation}: {error}")

  async def clear_short_term_memory(self, scope: str, conversation: str):
    """Clear short-term memory for a conversation (messages remain in long-term memory if enabled)."""
    async with self.lock:
      self.short_term_memory[(scope, conversation)].clear()
      logger.info(f"Cleared short-term memory for {scope}/{conversation}")

  async def delete_conversation(self, scope: str, conversation: str):
    """Delete all messages for a conversation from both short-term and long-term memory."""
    async with self.lock:
      # Clear from short-term memory
      if (scope, conversation) in self.short_term_memory:
        del self.short_term_memory[(scope, conversation)]

      # Delete from long-term memory (database) if enabled
      if self.enable_long_term_memory and self.connection:
        async with self.connection.cursor() as cur:
          await cur.execute(
            "DELETE FROM conversation WHERE tenant_id = %s AND scope = %s AND conversation = %s",
            (self.tenant_id, scope, conversation),
          )
          await self.connection.commit()
        logger.info(f"Deleted conversation {scope}/{conversation}")

  async def get_metrics(self) -> dict:
    """Return memory system metrics for monitoring."""
    async with self.lock:
      total_messages = sum(len(msgs) for msgs in self.short_term_memory.values())
      return {
        **self.metrics,
        "active_conversations": len(self.short_term_memory),
        "total_active_messages": total_messages,
        "avg_messages_per_conversation": total_messages / len(self.short_term_memory) if self.short_term_memory else 0,
      }

  async def health_check(self) -> bool:
    """Check if database connection is healthy."""
    if not self.enable_long_term_memory or not self.connection:
      return True  # Short-term memory-only mode is always "healthy"

    try:
      async with self.connection.cursor() as cur:
        await cur.execute("SELECT 1")
        await cur.fetchone()
      return True
    except Exception as e:
      logger.error(f"Database health check failed: {e}")
      return False

import asyncio
import psycopg
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
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
    has_structured_content = any(isinstance(msg.get('content'), dict) for msg in messages)

    if has_structured_content:
      # Use character estimate for structured content
      total_chars = 0
      for msg in messages:
        if isinstance(msg.get('content'), dict):
          # Extract text from structured content like {'text': '...', 'type': 'text'}
          if 'text' in msg['content']:
            total_chars += len(msg['content']['text'])
          else:
            total_chars += len(str(msg['content']))
        else:
          total_chars += len(str(msg.get('content', '')))
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
    Removes oldest messages first (FIFO).
    """
    messages = self.short_term_memory[(scope, conversation)]
    initial_count = len(messages)

    if self.max_tokens_in_short_term_memory and len(messages) > 1:
      # Token-aware trimming
      while await self._estimate_tokens(messages) > self.max_tokens_in_short_term_memory and len(messages) > 1:
        messages.pop(0)  # Remove oldest
        self.metrics["messages_trimmed"] += 1
        logger.debug(f"Trimmed message from short-term memory (token limit) for {scope}/{conversation}")

    # Count-based trimming
    while len(messages) > self.max_messages_in_short_term_memory and len(messages) > 1:
      messages.pop(0)  # Remove oldest
      self.metrics["messages_trimmed"] += 1
      logger.debug(f"Trimmed message from short-term memory (count limit) for {scope}/{conversation}")

    if len(messages) < initial_count:
      logger.debug(f"Trimmed {initial_count - len(messages)} messages from {scope}/{conversation}")

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

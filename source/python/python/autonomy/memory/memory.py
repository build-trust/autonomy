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

DEFAULT_MAX_ACTIVE_MESSAGES = 100
DEFAULT_MAX_ACTIVE_TOKENS = 8000


class Memory:
  """
  Manages conversation messages with two-tier storage:
  - Active memory: Recent messages in memory (for fast context retrieval)
  - Database: All messages persisted in PostgreSQL
  """

  def __init__(
    self,
    max_active_messages: int = DEFAULT_MAX_ACTIVE_MESSAGES,
    max_active_tokens: Optional[int] = DEFAULT_MAX_ACTIVE_TOKENS,
    model: Optional["Model"] = None,  # For token counting
  ):
    # Validate configuration
    if max_active_messages < 1:
      raise ValueError("max_active_messages must be at least 1")
    if max_active_tokens is not None and max_active_tokens < 100:
      raise ValueError("max_active_tokens must be at least 100")

    self.tenant_id = None
    self.db_url = None
    self.connection = None
    self.lock = asyncio.Lock()

    # Configuration
    self.max_active_messages = max_active_messages
    self.max_active_tokens = max_active_tokens
    self.model = model

    # In-memory storage
    # Key: (scope, conversation), Value: list of message dicts
    self.active_messages: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

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
    if "OCKAM_DATABASE_INSTANCE" not in environ:
      logger.info("No database configured, running in memory-only mode")
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

    # Load existing messages into active memory
    await self._load_active_memory()

  async def _load_active_memory(self):
    """Load recent messages from database into active memory (optimized)."""
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
        (self.tenant_id, self.max_active_messages),
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
        self.active_messages[key] = list(reversed(messages))

      logger.info(f"Loaded {len(rows)} messages from {len(conversations)} conversations")

  def set_instructions(self, instructions: dict):
    """Set system instructions (prepended to all conversations)."""
    self.instructions = [instructions]

  async def add_message(self, scope: str, conversation: str, message: dict):
    """
    Add a message to both active memory and database.
    Automatically trims active memory if needed.
    Adds timestamp if not present.
    """
    import time

    # Add timestamp if not present
    if "timestamp" not in message:
      message["timestamp"] = time.time()

    async with self.lock:
      # Add to active memory
      self.active_messages[(scope, conversation)].append(message)
      self.metrics["messages_added"] += 1

      # Persist to database
      if self.connection:
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
      await self._trim_active_memory(scope, conversation)

  async def add_messages(self, scope: str, conversation: str, messages: List[dict]):
    """
    Add multiple messages efficiently in a batch.
    Automatically trims active memory if needed.
    """
    import time

    # Add timestamps if not present
    for message in messages:
      if "timestamp" not in message:
        message["timestamp"] = time.time()

    async with self.lock:
      # Add to active memory
      self.active_messages[(scope, conversation)].extend(messages)
      self.metrics["messages_added"] += len(messages)
      self.metrics["batch_operations"] += 1

      # Persist to database in batch
      if self.connection:
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
      await self._trim_active_memory(scope, conversation)

  async def get_messages(self, scope: str, conversation: str) -> List[dict]:
    """
    Get messages for a conversation (instructions + active memory).
    This is the primary method used by agents.
    """
    async with self.lock:
      return self.instructions + self.active_messages[(scope, conversation)]

  async def get_messages_only(self, scope: str, conversation: str) -> List[dict]:
    """Get only conversation messages (no instructions) from active memory."""
    async with self.lock:
      return self.active_messages[(scope, conversation)].copy()

  async def get_all_messages(self, scope: str, conversation: str, limit: Optional[int] = None) -> List[dict]:
    """
    Retrieve all messages for a conversation from the database.
    Use this for analysis, export, or when you need all messages.
    """
    if not self.connection:
      # Fallback to active memory if no database
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
    """Get total number of messages in a conversation (from database)."""
    if not self.connection:
      async with self.lock:
        return len(self.active_messages[(scope, conversation)])

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

    try:
      # Use model's token counter if available
      return self.model.count_tokens(messages)
    except Exception as e:
      logger.warning(f"Token counting failed: {e}, using character estimate")
      total_chars = sum(len(str(msg)) for msg in messages)
      return total_chars // 4

  async def _should_trim_active_memory(self, scope: str, conversation: str) -> bool:
    """Check if active memory needs trimming."""
    messages = self.active_messages[(scope, conversation)]

    # Check message count limit
    if len(messages) > self.max_active_messages:
      return True

    # Check token limit (if configured and model available)
    if self.max_active_tokens:
      token_count = await self._estimate_tokens(messages)
      if token_count > self.max_active_tokens:
        return True

    return False

  async def _trim_active_memory(self, scope: str, conversation: str):
    """
    Trim active memory to stay within configured limits.
    Removes oldest messages first (FIFO).
    """
    messages = self.active_messages[(scope, conversation)]
    initial_count = len(messages)

    if self.max_active_tokens and len(messages) > 1:
      # Token-aware trimming
      while await self._estimate_tokens(messages) > self.max_active_tokens and len(messages) > 1:
        messages.pop(0)  # Remove oldest
        self.metrics["messages_trimmed"] += 1
        logger.debug(f"Trimmed message from active memory (token limit) for {scope}/{conversation}")

    # Count-based trimming
    while len(messages) > self.max_active_messages and len(messages) > 1:
      messages.pop(0)  # Remove oldest
      self.metrics["messages_trimmed"] += 1
      logger.debug(f"Trimmed message from active memory (count limit) for {scope}/{conversation}")

    if len(messages) < initial_count:
      logger.debug(f"Trimmed {initial_count - len(messages)} messages from {scope}/{conversation}")

  async def clear_active_memory(self, scope: str, conversation: str):
    """Clear active memory for a conversation (messages remain in database)."""
    async with self.lock:
      self.active_messages[(scope, conversation)].clear()
      logger.info(f"Cleared active memory for {scope}/{conversation}")

  async def delete_conversation(self, scope: str, conversation: str):
    """Delete all messages for a conversation from both memory and database."""
    async with self.lock:
      # Clear from active memory
      if (scope, conversation) in self.active_messages:
        del self.active_messages[(scope, conversation)]

      # Delete from database
      if self.connection:
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
      total_messages = sum(len(msgs) for msgs in self.active_messages.values())
      return {
        **self.metrics,
        "active_conversations": len(self.active_messages),
        "total_active_messages": total_messages,
        "avg_messages_per_conversation": total_messages / len(self.active_messages) if self.active_messages else 0,
      }

  async def health_check(self) -> bool:
    """Check if database connection is healthy."""
    if not self.connection:
      return True  # Memory-only mode is always "healthy"

    try:
      async with self.connection.cursor() as cur:
        await cur.execute("SELECT 1")
        await cur.fetchone()
      return True
    except Exception as e:
      logger.error(f"Database health check failed: {e}")
      return False

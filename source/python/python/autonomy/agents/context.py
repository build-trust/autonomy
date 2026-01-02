"""
Context template system for organizing model input context into logical sections.

This module provides a flexible way to structure the context that is sent to
language models. Instead of sending a flat list of messages, context can be
organized into logical sections (system instructions, conversation history,
additional context, etc.) that can be individually controlled and customized.

## Overview

Context templates solve the problem of managing complex model inputs by:
- Breaking context into logical, reusable sections
- Enabling dynamic context based on conversation state
- Supporting filtering and transformation of messages
- Providing clean separation of concerns
- Making it easy to add features like RAG, summarization, etc.

## Key Components

1. **ContextSection**: Base class for sections (optional - uses duck typing)
  - Defines the interface: `get_messages(memory, scope, conversation, context)`
  - Memory is passed as first argument for accessing conversation history
  - Can be enabled/disabled independently
  - Supports shared context dict for inter-section communication
  - Any object with `get_messages()` can be a section

2. **SystemInstructionsSection**: Agent's system prompt and instructions
  - Always appears first (typically)
  - Sets behavioral parameters for the model
  - Static content, rarely changes during conversation

3. **ConversationHistorySection**: Message history from memory
  - Core conversational context
  - Supports filtering (e.g., exclude internal messages)
  - Supports transformation (e.g., redact sensitive info)
  - Can limit message count for long conversations

4. **AdditionalContextSection**: Flexible section for custom context
  - Can use static messages or provider functions
  - Perfect for RAG, current time, user preferences, etc.
  - Enables dynamic context injection

5. **ContextTemplate**: Orchestrates sections to build final context
  - Maintains ordered list of sections
  - Builds context by combining section outputs
  - Provides section management (add/remove/get)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ContextTemplate                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. SystemInstructionsSection                        │   │
│  │     → Returns: [system message]                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. AdditionalContextSection                         │   │
│  │     → Returns: [context message 1, context msg 2]    │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. ConversationHistorySection                       │   │
│  │     → Returns: [user msg, assistant msg, ...]        │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│              Combined messages → Model                      │
└─────────────────────────────────────────────────────────────┘
```

## Basic Usage

```python
from autonomy.agents.context import (
  ContextTemplate,
  SystemInstructionsSection,
  ConversationHistorySection,
  AdditionalContextSection,
)

# Create sections
instructions = [{"role": "system", "content": {...}}]
template = ContextTemplate(
  [
    SystemInstructionsSection(instructions),
    ConversationHistorySection(),  # No memory needed at construction
  ]
)

# Build context - memory is passed when building
messages = await template.build_context(memory, scope="user123", conversation="chat456")

# Modify template dynamically
template.add_section(AdditionalContextSection(name="rag"), index=1)
template.get_section("rag").add_message({...})
```

## Advanced Usage

### Custom Sections

```python
# Option 1: Inherit from ContextSection (provides set_enabled, etc.)
class KnowledgeRetrievalSection(ContextSection):
  def __init__(self, vector_db):
    super().__init__("knowledge_retrieval")
    self.vector_db = vector_db

  async def get_messages(self, memory, scope, conversation, context):
    # Memory is passed as first argument - use it to access conversation history
    # Retrieve relevant documents
    docs = await self.vector_db.search(query)
    return [format_doc_as_message(doc) for doc in docs]


# Option 2: Duck typing - no inheritance required
class SimpleSection:
  def __init__(self):
    self.name = "simple"
    self.enabled = True

  async def get_messages(self, memory, scope, conversation, context):
    # Memory is always passed, even if not used
    return [{"role": "system", "content": {"text": "Hello", "type": "text"}}]
```

### Filtering History

```python
def exclude_tool_calls(msg):
  return "tool_calls" not in msg


# No memory needed at construction - it's passed when get_messages is called
history = ConversationHistorySection(filter_fn=exclude_tool_calls, max_messages=50)
```

### Dynamic Context Providers

```python
async def current_time_provider(scope, conversation, context):
  from datetime import datetime

  time_str = datetime.now().isoformat()
  return [{"role": "system", "content": {"text": f"Current time: {time_str}", "type": "text"}}]


template.add_section(AdditionalContextSection(provider_fn=current_time_provider))
```

### Shared Context Between Sections

```python
# Section 1: Analyze and store data
class AnalysisSection(ContextSection):
  async def get_messages(self, memory, scope, conversation, params):
    params["sentiment"] = analyze_sentiment(conversation)
    return []


# Section 2: Use stored data
class ResponseSection(ContextSection):
  async def get_messages(self, memory, scope, conversation, params):
    sentiment = params.get("sentiment", "neutral")
    return [{"role": "system", "content": {"text": f"Sentiment: {sentiment}", "type": "text"}}]
```

## Common Patterns

### RAG (Retrieval-Augmented Generation)

```python
async def rag_provider(scope, conversation, params):
  # Get last user message
  query = get_last_user_message(conversation)
  # Search vector database
  results = await vector_db.search(query, top_k=3)
  # Format as messages
  return [format_result(r) for r in results]


template.add_section(AdditionalContextSection(provider_fn=rag_provider))
```

### Conversation Summarization

```python
class SummarySection(ContextSection):
  async def get_messages(self, memory, scope, conversation, params):
    # Access conversation history via memory parameter
    messages = await memory.get_messages_only(scope, conversation)
    if len(messages) > 20:
      summary = await generate_summary(messages)
      return [{"role": "system", "content": {"text": f"Summary: {summary}", "type": "text"}}]
    return []
```

### Time-Aware Context

```python
async def time_context(scope, conversation, params):
  from datetime import datetime

  now = datetime.now()
  return [{"role": "system", "content": {"text": f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}", "type": "text"}}]
```

## Design Principles

1. **Order Matters**: Sections are processed sequentially
2. **Independence**: Each section can be enabled/disabled independently
3. **Composability**: Sections can be mixed and matched
4. **Extensibility**: Easy to create custom sections
5. **Shared State**: Context dict enables section communication
6. **Error Tolerance**: Template continues if a section fails
7. **Transparency**: Extensive logging for debugging

## Performance Considerations

- Sections are called on every model invocation
- Keep section logic fast (use caching where appropriate)
- Use filters to reduce message count, not just for content filtering
- Provider functions should be async for I/O operations
- Consider message count limits for long conversations

## Debugging

Enable debug logging to see section contributions:
```python
import logging

logging.getLogger("context").setLevel(logging.DEBUG)
```

This will show:
- `[CONTEXT→BUILD]` - Template building process
- `[CONTEXT→section_name]` - Each section's contribution

## Examples

See `autonomy/examples/context_templates/` for complete working examples:
- `basic_example.py` - Simple context template usage
- `advanced_example.py` - Custom sections, filtering, and advanced patterns
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from ..logs import get_logger
from ..memory.memory import Memory, find_tool_pair_boundary, validate_tool_pairing
from ..models.model import Model

logger = get_logger("context")

# Filesystem defaults
FILESYSTEM_VISIBILITY_DEFAULT = "conversation"


class ContextSection:
  """
  Base class for context sections (optional - uses duck typing).

  Each section represents a logical component of the context sent to the model.
  Sections are processed in order and can contribute zero or more messages.

  Duck Typing:
    Any object with the following interface can be used as a section:
    - `name` attribute (str) - Section identifier
    - `enabled` attribute (bool) - Whether section is active
    - `get_messages(scope, conversation, params)` method - Returns message list

  This class is provided as a convenient base with default implementations,
  but inheritance is not required. You can create sections without inheriting
  from this class as long as they implement the required interface.

  Attributes:
    name: Identifier for this section (for logging and debugging)
    enabled: Whether this section is active

  Example (with inheritance):
    ```python
    class MySection(ContextSection):
      def __init__(self):
        super().__init__("my_section")

      async def get_messages(self, scope, conversation, params):
        return [...]
    ```

  Example (duck typing, no inheritance):
    ```python
    class MySection:
      def __init__(self):
        self.name = "my_section"
        self.enabled = True

      async def get_messages(self, scope, conversation, params):
        return [...]
    ```
  """

  def __init__(self, name: str, enabled: bool = True):
    """
    Initialize a context section.

    Args:
      name: Section identifier
      enabled: Whether section is active (default: True)
    """
    self.name = name
    self.enabled = enabled

  async def get_messages(self, memory: Memory, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
    """
    Retrieve messages for this section.

    This is the core method that sections must implement. Override this
    in subclasses or implement it in duck-typed sections.

    Args:
      memory: Memory instance for retrieving conversation history
      scope: User/tenant identifier for memory isolation
      conversation: Conversation thread identifier
      params: Shared parameters dictionary for passing data between sections

    Returns:
      List of message dicts to include in model input

    Example message format:
      {
        "role": "system" | "user" | "assistant",
        "content": {"text": "...", "type": "text"},
        "phase": "system" | "executing" | ...
      }
    """
    return []

  def set_enabled(self, enabled: bool):
    """Enable or disable this section."""
    self.enabled = enabled
    logger.debug(f"Section '{self.name}' {'enabled' if enabled else 'disabled'}")


class SystemInstructionsSection(ContextSection):
  """
  System instructions section.

  Provides the agent's system prompt/instructions. This typically appears
  first in the context and sets the behavioral parameters for the model.

  Example:
    ```python
    section = SystemInstructionsSection(
      [{"role": "system", "content": {"text": "You are a helpful assistant", "type": "text"}}]
    )
    ```
  """

  def __init__(self, instructions: List[dict], enabled: bool = True):
    """
    Initialize system instructions section.

    Args:
      instructions: List of system message dicts
      enabled: Whether section is active (default: True)
    """
    super().__init__("system_instructions", enabled)
    self.instructions = instructions

  async def get_messages(self, memory: Memory, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
    """Return system instructions."""
    if not self.enabled:
      return []

    logger.debug(f"[CONTEXT→{self.name}] Adding {len(self.instructions)} instruction message(s)")

    return self.instructions


class ConversationHistorySection(ContextSection):
  """
  Conversation history section.

  Retrieves message history from memory. This is the core conversational
  context showing what has been discussed so far.

  The section can optionally:
  - Limit message count
  - Filter messages by criteria
  - Transform messages before inclusion

  Example:
    ```python
    section = ConversationHistorySection()
    # Or with filtering:
    section = ConversationHistorySection(filter_fn=lambda msg: msg.get("role") != "system")
    ```
  """

  def __init__(
    self,
    enabled: bool = True,
    max_messages: Optional[int] = None,
    filter_fn: Optional[Callable[[dict], bool]] = None,
    transform_fn: Optional[Callable[[dict], dict]] = None,
  ):
    """
    Initialize conversation history section.

    Args:
      enabled: Whether section is active (default: True)
      max_messages: Optional limit on number of messages to include
      filter_fn: Optional function to filter messages (returns True to include)
      transform_fn: Optional function to transform each message before inclusion
    """
    super().__init__("conversation_history", enabled)
    self.max_messages = max_messages
    self.filter_fn = filter_fn
    self.transform_fn = transform_fn

  async def get_messages(self, memory: Memory, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
    """Retrieve conversation history from memory."""
    if not self.enabled:
      return []

    # Get messages from memory (without system instructions)
    messages = await memory.get_messages_only(scope, conversation)

    # Apply filter if provided
    if self.filter_fn:
      original_count = len(messages)
      messages = [msg for msg in messages if self.filter_fn(msg)]
      if len(messages) < original_count:
        logger.debug(f"[CONTEXT→{self.name}] Filtered {original_count - len(messages)} messages")

    # Apply message limit if provided
    if self.max_messages and len(messages) > self.max_messages:
      logger.debug(f"[CONTEXT→{self.name}] Limiting from {len(messages)} to {self.max_messages} messages")
      messages = messages[-self.max_messages :]  # Keep most recent

    # Apply transformation if provided
    if self.transform_fn:
      messages = [self.transform_fn(msg) for msg in messages]

    logger.debug(f"[CONTEXT→{self.name}] Adding {len(messages)} conversation message(s)")

    # Store message count in shared params for other sections
    params["conversation_message_count"] = len(messages)

    return messages


SUMMARIZATION_INSTRUCTIONS = """Summarize the following conversation history concisely.
Focus on:
- Key questions asked by the user
- Important facts and answers provided
- Topics discussed
- Any ongoing context that would be important for future questions

Keep the summary brief but comprehensive. Use bullet points for clarity.

Conversation to summarize:
{conversation}

Summary:"""


class SummarizedHistorySection(ContextSection):
  """
  Conversation history section with async (non-blocking) summarization.

  Unlike synchronous summarization, this section:
  1. NEVER blocks the request on summarization
  2. Returns cached summary immediately (even if slightly stale)
  3. Schedules background summarization for cache updates
  4. The next request benefits from the updated summary

  This eliminates summarization latency from the request path,
  resulting in significantly faster response times compared to sync summarization.

  How it works:
    - Messages accumulate normally until reaching `ceiling`
    - When `ceiling` is exceeded, older messages are summarized
    - After summarization, model sees ~`floor` messages (summary + recent)
    - The verbatim window starts RIGHT AFTER where the summary ends (no hidden messages)
    - Re-summarization triggers when `batch_size` (ceiling - floor) new messages accumulate

  Cold Start Behavior:
    - If no cached summary exists, returns all messages (like pre-threshold)
    - Background task generates summary for subsequent requests
    - After first summarization, all future requests use cached summary

  Staleness:
    - Summary may be 1-2 turns behind current conversation
    - Recent messages (last N) are always current
    - For knowledge-based agents, this is acceptable since they can re-search

  Example:
    ```python
    from autonomy import Model
    from autonomy.agents.context import SummarizedHistorySection, ContextTemplate

    section = SummarizedHistorySection(
      summary_model=Model("claude-haiku"),  # Use fast model for summaries
      floor=10,  # Min messages visible after summarization
      ceiling=20,  # Max messages before triggering summarization
    )

    template = ContextTemplate(
      [
        SystemInstructionsSection(instructions),
        section,
      ]
    )
    ```
  """

  def __init__(
    self,
    summary_model: Model,
    floor: int = 10,
    ceiling: int = 20,
    size: int = 2048,
    instructions: Optional[str] = None,
    enabled: bool = True,
    # Legacy parameter names (deprecated, use floor/ceiling/size instead)
    recent_count: Optional[int] = None,
    summarize_threshold: Optional[int] = None,
    max_summary_tokens: Optional[int] = None,
  ):
    """
    Initialize async summarized history section.

    Args:
      summary_model: Model to use for generating summaries (recommend fast model like Haiku)
      floor: Minimum messages visible after summarization (default: 10)
      ceiling: Maximum messages before triggering summarization (default: 20)
      size: Maximum tokens for summary generation (default: 2048)
      instructions: Custom prompt template that replaces the default SUMMARIZATION_INSTRUCTIONS.
        Must include {conversation} placeholder for the conversation text.
        Example: "Summarize this chat, preserving code:\n{conversation}\n\nSummary:"
      enabled: Whether section is active (default: True)

    Legacy Args (deprecated):
      recent_count: Use `floor` instead
      summarize_threshold: Use `ceiling` instead
      max_summary_tokens: Use `size` instead
    """
    super().__init__("conversation_history", enabled)
    self.summary_model = summary_model

    # Support legacy parameter names with deprecation
    if recent_count is not None:
      logger.warning("SummarizedHistorySection: 'recent_count' is deprecated, use 'floor' instead")
      floor = recent_count
    if summarize_threshold is not None:
      logger.warning("SummarizedHistorySection: 'summarize_threshold' is deprecated, use 'ceiling' instead")
      ceiling = summarize_threshold
    if max_summary_tokens is not None:
      logger.warning("SummarizedHistorySection: 'max_summary_tokens' is deprecated, use 'size' instead")
      size = max_summary_tokens

    self.floor = floor
    self.ceiling = ceiling
    self.size = size
    self.batch_size = ceiling - floor  # Re-summarize when this many new messages accumulate
    self.instructions = instructions

    # Validate configuration
    if self.floor >= self.ceiling:
      raise ValueError(f"floor ({self.floor}) must be less than ceiling ({self.ceiling})")
    if self.floor < 1:
      raise ValueError(f"floor ({self.floor}) must be at least 1")

    # Cache storage - keyed by "scope:conversation"
    self._summary_cache: Dict[str, str] = {}  # cached summary text
    self._summarized_count: Dict[str, int] = {}  # count of messages summarized
    self._pending_summarization: Dict[str, bool] = {}  # is summarization in progress?

    # Per-conversation locks to prevent duplicate summarization
    self._locks: Dict[str, asyncio.Lock] = {}

    # Metrics for debugging
    self._metrics = {
      "cache_hits": 0,
      "cache_misses": 0,
      "background_tasks_started": 0,
      "background_tasks_completed": 0,
      "background_tasks_failed": 0,
    }

  def _get_lock(self, cache_key: str) -> asyncio.Lock:
    """Get or create a lock for a conversation."""
    if cache_key not in self._locks:
      self._locks[cache_key] = asyncio.Lock()
    return self._locks[cache_key]

  def _format_messages_for_summary(self, messages: List[dict]) -> str:
    """Format messages into a string for summarization."""
    formatted = []
    for msg in messages:
      role = msg.get("role", "unknown")
      content = msg.get("content", "")

      # Handle different content formats
      if isinstance(content, dict):
        if "text" in content:
          content = content["text"]
        elif "content" in content:
          content = content["content"]
        else:
          content = str(content)
      elif isinstance(content, list):
        # Handle list of content blocks
        text_parts = []
        for part in content:
          if isinstance(part, dict) and "text" in part:
            text_parts.append(part["text"])
          elif isinstance(part, str):
            text_parts.append(part)
        content = " ".join(text_parts)

      # Skip tool calls and tool results for summary (keep it high-level)
      if role in ("tool", "function"):
        continue

      # Handle tool_calls in assistant messages
      if role == "assistant" and "tool_calls" in msg:
        tool_names = [tc.get("name", "unknown") for tc in msg.get("tool_calls", [])]
        formatted.append(f"Assistant: [Used tools: {', '.join(tool_names)}]")
        if content:
          formatted.append(f"Assistant: {content[:500]}")
      elif content:
        formatted.append(f"{role.capitalize()}: {content[:500]}")

    return "\n".join(formatted)

  async def _generate_summary(self, messages: List[dict]) -> str:
    """Generate a summary of the given messages."""
    conversation_text = self._format_messages_for_summary(messages)

    if not conversation_text.strip():
      return ""

    # Use custom instructions if provided (replaces entire prompt), otherwise use default
    if self.instructions:
      prompt = self.instructions.format(conversation=conversation_text)
    else:
      prompt = SUMMARIZATION_INSTRUCTIONS.format(conversation=conversation_text)

    try:
      response = await self.summary_model.complete_chat(
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=self.size,
      )

      # Extract text from response
      if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list) and len(content) > 0:
          if hasattr(content[0], "text"):
            return content[0].text
          return str(content[0])
        return str(content)
      return str(response)

    except Exception as e:
      logger.error(f"Summarization failed: {e}")
      # Return a simple fallback
      return f"[Previous conversation - {len(messages)} messages]"

  def _create_summary_message(
    self,
    summary: str,
    summarized_count: int,
    stale_count: Optional[int] = None,
  ) -> dict:
    """Create a summary message dict."""
    staleness_note = ""
    if stale_count and stale_count > 0:
      staleness_note = f" (summary may be {stale_count} messages behind)"

    return {
      "role": "system",
      "content": {
        "text": f"[CONVERSATION SUMMARY - {summarized_count} earlier messages{staleness_note}]\n{summary}\n[END SUMMARY - Recent messages follow]",
        "type": "text",
      },
    }

  async def _background_summarize(self, cache_key: str, messages: List[dict]):
    """
    Background task to generate and cache summary.

    This runs after the main request returns, so it doesn't add latency.
    The generated summary will be available for the next request.
    """
    lock = self._get_lock(cache_key)

    # Quick check if already in progress (without acquiring lock)
    if self._pending_summarization.get(cache_key):
      logger.debug(f"Summarization already in progress for {cache_key}")
      return

    async with lock:
      # Double-check after acquiring lock
      if self._pending_summarization.get(cache_key):
        return

      self._pending_summarization[cache_key] = True
      self._metrics["background_tasks_started"] += 1

      try:
        logger.info(f"Starting background summarization for {cache_key} ({len(messages)} messages)")
        summary = await self._generate_summary(messages)

        # Update cache
        self._summary_cache[cache_key] = summary
        self._summarized_count[cache_key] = len(messages)

        self._metrics["background_tasks_completed"] += 1
        logger.info(f"Background summarization completed for {cache_key}")

      except Exception as e:
        self._metrics["background_tasks_failed"] += 1
        logger.error(f"Background summarization failed for {cache_key}: {e}")

      finally:
        self._pending_summarization[cache_key] = False

  async def get_messages(
    self,
    memory: Memory,
    scope: str,
    conversation: str,
    params: Dict[str, Any],
  ) -> List[dict]:
    """
    Retrieve conversation history with async summarization.

    This method NEVER blocks on summarization. It returns immediately with:
    - Cached summary + recent messages (if cache exists)
    - All messages (if no cache yet - cold start)

    Background summarization is scheduled if the cache is stale.

    Algorithm:
    - Below ceiling: return all messages verbatim
    - Above ceiling: return [summary] + messages[summarized_count:]
    - Re-summarize when batch_size (ceiling - floor) new messages accumulate
    - Verbatim window starts RIGHT AFTER where summary ends (no hidden messages)
    """
    if not self.enabled:
      return []

    # Get messages from memory
    messages = await memory.get_messages_only(scope, conversation)
    total_messages = len(messages)
    params["conversation_message_count"] = total_messages

    # If below ceiling, return all messages (no summarization needed)
    if total_messages <= self.ceiling:
      logger.debug(f"[CONTEXT→{self.name}] Below ceiling ({total_messages}/{self.ceiling}), returning all messages")
      return messages

    cache_key = f"{scope}:{conversation}"

    # Check cache status
    cached_summary = self._summary_cache.get(cache_key)
    summarized_count = self._summarized_count.get(cache_key, 0)

    # Calculate how many messages should be summarized to maintain floor verbatim messages
    target_summarized = total_messages - self.floor

    # Determine if we need to (re-)summarize
    # Re-summarize when batch_size new messages have accumulated since last summary
    new_since_summary = target_summarized - summarized_count
    should_summarize = (
      # First time summarization: we have messages to summarize but no summary yet
      (target_summarized > 0 and summarized_count == 0)
      or
      # Re-summarization: batch_size new messages have accumulated
      (new_since_summary >= self.batch_size)
    )

    # Schedule background summarization if needed (non-blocking)
    if should_summarize and not self._pending_summarization.get(cache_key):
      messages_to_summarize = messages[:target_summarized]
      logger.debug(
        f"[CONTEXT→{self.name}] Scheduling summarization for {cache_key} "
        f"({len(messages_to_summarize)} messages, new_since_summary={new_since_summary})"
      )
      asyncio.create_task(self._background_summarize(cache_key, messages_to_summarize))

    # Return immediately with whatever we have
    if cached_summary:
      # Cache hit - return cached summary + ALL messages after the summarized portion
      # IMPORTANT: verbatim_start = summarized_count ensures NO hidden messages
      self._metrics["cache_hits"] += 1
      verbatim_start = summarized_count

      # Adjust verbatim_start to preserve tool_use/tool_result pairs
      # This prevents orphaned tool results that would cause Claude API errors
      adjusted_verbatim_start = find_tool_pair_boundary(messages, verbatim_start)
      if adjusted_verbatim_start != verbatim_start:
        logger.debug(
          f"[CONTEXT→{self.name}] Adjusted verbatim_start from {verbatim_start} to "
          f"{adjusted_verbatim_start} to preserve tool pairs"
        )
        verbatim_start = adjusted_verbatim_start

      stale_count = new_since_summary if new_since_summary > 0 else 0

      logger.debug(
        f"[CONTEXT→{self.name}] Cache hit for {cache_key} "
        f"(summarized={summarized_count}, verbatim={total_messages - verbatim_start}, stale_by={stale_count})"
      )

      summary_message = self._create_summary_message(
        cached_summary,
        summarized_count,
        stale_count if stale_count > 0 else None,
      )

      result_messages = [summary_message] + messages[verbatim_start:]

      # Validate tool pairing before returning
      # Only validate the conversation messages, not the summary
      is_valid, error = validate_tool_pairing(messages[verbatim_start:])
      if not is_valid:
        logger.error(f"[CONTEXT→{self.name}] Tool pairing validation failed for {cache_key}: {error}")
        # Fall back to returning all messages to avoid Claude API error
        # This is safer than returning broken tool pairs
        logger.warning(f"[CONTEXT→{self.name}] Falling back to all {total_messages} messages due to tool pairing issue")
        return messages

      return result_messages
    else:
      # Cache miss (cold start) - return all messages
      # Background task will generate summary for next request
      self._metrics["cache_misses"] += 1
      logger.debug(f"[CONTEXT→{self.name}] Cache miss for {cache_key}, returning all {total_messages} messages")
      return messages

  def get_metrics(self) -> Dict[str, int]:
    """Get summarization metrics for debugging."""
    return self._metrics.copy()

  def clear_cache(self, scope: Optional[str] = None, conversation: Optional[str] = None):
    """
    Clear cached summaries.

    Args:
      scope: If provided with conversation, clear only that conversation's cache
      conversation: If provided with scope, clear only that conversation's cache

    If neither provided, clears all caches.
    """
    if scope and conversation:
      cache_key = f"{scope}:{conversation}"
      self._summary_cache.pop(cache_key, None)
      self._summarized_count.pop(cache_key, None)
      logger.debug(f"[CONTEXT→{self.name}] Cleared cache for {cache_key}")
    else:
      self._summary_cache.clear()
      self._summarized_count.clear()
      logger.debug(f"[CONTEXT→{self.name}] Cleared all summary caches")

  def get_cache_info(self) -> Dict[str, Any]:
    """Get information about cached summaries."""
    return {
      "cached_conversations": len(self._summary_cache),
      "pending_summarizations": sum(1 for v in self._pending_summarization.values() if v),
      "conversations": list(self._summary_cache.keys()),
    }


class AdditionalContextSection(ContextSection):
  """
  Additional context section.

  Provides a flexible section for custom context that doesn't fit in other
  categories. This could be:
  - Retrieved knowledge from RAG systems
  - Relevant facts from long-term memory
  - Dynamic instructions based on conversation state
  - External data (weather, calendar, etc.)

  The section uses a provider function to generate messages on-demand.

  Example:
    ```python
    async def provide_weather(scope, conv, ctx):
      return [{"role": "system", "content": {"text": "Current weather: Sunny", "type": "text"}}]


    section = AdditionalContextSection(provider_fn=provide_weather)
    ```
  """

  def __init__(
    self,
    name: str = "additional_context",
    enabled: bool = True,
    provider_fn: Optional[Callable[[str, str, Dict[str, Any]], List[dict]]] = None,
  ):
    """
    Initialize additional context section.

    Args:
      name: Section identifier (default: "additional_context")
      enabled: Whether section is active (default: True)
      provider_fn: Optional async function that returns message dicts
                    Signature: async (scope, conversation, context) -> List[dict]
    """
    super().__init__(name, enabled)
    self.provider_fn = provider_fn
    self._static_messages: List[dict] = []

  def set_static_messages(self, messages: List[dict]):
    """
    Set static messages for this section.

    Args:
      messages: List of message dicts to include
    """
    self._static_messages = messages
    logger.debug(f"[CONTEXT→{self.name}] Set {len(messages)} static message(s)")

  def add_message(self, message: dict):
    """
    Add a single static message to this section.

    Args:
      message: Message dict to add
    """
    self._static_messages.append(message)
    logger.debug(f"[CONTEXT→{self.name}] Added 1 message (total: {len(self._static_messages)})")

  def clear_messages(self):
    """Clear all static messages."""
    count = len(self._static_messages)
    self._static_messages = []
    logger.debug(f"[CONTEXT→{self.name}] Cleared {count} message(s)")

  async def get_messages(self, memory: Memory, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
    """Retrieve additional context messages."""
    if not self.enabled:
      return []

    messages = []

    # Include static messages
    if self._static_messages:
      messages.extend(self._static_messages)

    # Call provider function if available
    if self.provider_fn:
      try:
        provided_messages = await self.provider_fn(scope, conversation, params)
        if provided_messages:
          messages.extend(provided_messages)
      except Exception as e:
        logger.error(f"[CONTEXT→{self.name}] Provider function failed: {e}")

    if messages:
      logger.debug(f"[CONTEXT→{self.name}] Adding {len(messages)} additional message(s)")

    return messages


class WorkspaceReminderSection(ContextSection):
  """
  Workspace reminder section.

  Periodically reminds the agent to check workspace files for context.
  Shows on first turn and then at configurable frequency.

  Example:
    ```python
    section = WorkspaceReminderSection(frequency=5)  # Every 5 turns
    ```
  """

  def __init__(self, frequency: int = 5, enabled: bool = True):
    """
    Initialize workspace reminder section.

    Args:
      frequency: Show reminder every N turns (default: 5)
      enabled: Whether section is active (default: True)
    """
    super().__init__("workspace_reminder", enabled)
    self.frequency = frequency

  async def get_messages(self, memory: Memory, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
    """Generate workspace reminder message."""
    if not self.enabled:
      return []

    message_count = params.get("conversation_message_count", 0)

    # Show on first turn and then periodically
    if message_count == 0 or (message_count > 0 and message_count % self.frequency == 0):
      logger.debug(f"[CONTEXT→{self.name}] Adding workspace reminder")
      return [
        {
          "role": "system",
          "content": {
            "text": "💡 Reminder: Use list_directory and read_file to check your workspace for context. Read memory.md first and then explore other files.",
            "type": "text",
          },
          "phase": "system",
        }
      ]

    return []


class FrameworkInstructionsSection(ContextSection):
  """
  Framework instructions section.

  Provides automatic documentation of built-in Autonomy framework capabilities
  including time tools, user input tools, and subagent delegation.

  This section dynamically generates instructions based on agent configuration:
  - Time tools (always included)
  - ask_user_for_input tool (conditional)
  - Subagent delegation tools (conditional)

  Example:
    ```python
    section = FrameworkInstructionsSection(
      enable_ask_for_user_input=True, subagent_configs={"researcher": {...}, "writer": {...}}
    )
    ```
  """

  def __init__(
    self,
    enable_ask_for_user_input: bool = False,
    enable_filesystem: bool = False,
    filesystem_visibility: str = FILESYSTEM_VISIBILITY_DEFAULT,
    subagent_configs: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
  ):
    """
    Initialize framework instructions section.

    Args:
      enable_ask_for_user_input: Whether ask_user_for_input tool is enabled
      enable_filesystem: Whether filesystem tools are enabled
      filesystem_visibility: Filesystem visibility level
      subagent_configs: Optional dict of configured subagents
      enabled: Whether section is active (default: True)
    """
    super().__init__("framework_instructions", enabled)
    self.enable_ask_for_user_input = enable_ask_for_user_input
    self.enable_filesystem = enable_filesystem
    self.filesystem_visibility = filesystem_visibility
    self.subagent_configs = subagent_configs or {}

  def _generate_framework_instructions(self) -> str:
    """
    Generate framework-level system instructions about built-in tools and capabilities.

    Returns:
      String containing framework instructions
    """
    instructions = []

    # Framework header
    instructions.append("## Autonomy Framework Capabilities")

    # Built-in time tools
    instructions.append(
      "### Time Tools\n"
      "You have access to built-in time tools:\n"
      "- `get_current_time_utc`: Returns current UTC time in ISO 8601 format\n"
      "- `get_current_time`: Returns current local time in ISO 8601 format"
    )

    # Filesystem tools (conditionally included)
    if self.enable_filesystem:
      visibility_descriptions = {
        "conversation": "Files are isolated per conversation. Each conversation has its own separate filesystem.",
        "scope": "Files are shared across conversations for the same user/tenant but isolated from other users.",
        "agent": "Files are shared across all users and conversations for this agent.",
        "all": "Files are shared across all agents and users in the system.",
      }

      visibility_info = visibility_descriptions.get(
        self.filesystem_visibility, "Files have custom visibility settings."
      )

      instructions.append(
        f"### Filesystem Tools\n"
        f"You have access to filesystem operations with **{self.filesystem_visibility}-level visibility**.\n"
        f"{visibility_info}\n\n"
        f"Available operations:\n"
        f"- `list_directory`: List files and directories\n"
        f"- `read_file`: Read file contents (supports partial reading with start/end lines)\n"
        f"- `write_file`: Create or overwrite files\n"
        f"- `edit_file`: Edit files by replacing specific strings\n"
        f"- `find_files`: Search for files by name pattern (glob)\n"
        f"- `search_in_files`: Search file contents using regex\n"
        f"- `remove_file`: Delete a file\n"
        f"- `remove_directory`: Delete a directory and its contents\n\n"
        f"All paths are relative to your filesystem root ('/').\n\n"
        f"**Workspace Best Practices:**\n\n"
        f"Your files persist. Use them actively to be more effective.\n\n"
        f"**Core Workflow:**\n"
        f"1. **Before responding to requests:**\n"
        f"   - ALWAYS read `memory.md` first for essential facts and key context\n"
        f'   - Search your files: `search_in_files("keywords", "/")`\n'
        f"   - Check if you've worked on this before\n"
        f"   - Build on existing knowledge\n\n"
        f"2. **After learning something important:**\n"
        f"   - ALWAYS update `memory.md` with key facts, preferences, and decisions\n"
        f"   - Keep it current with the most important information!\n"
        f'   - Create detailed notes: `write_file("/notes/topic.md", content)`\n'
        f"   - You can search for it later\n\n"
        f"3. **For complex work:**\n"
        f"   - Update `memory.md` with project context and key decisions\n"
        f'   - Create a plan: `write_file("/plans/project.md", plan)`\n'
        f'   - Track tasks: `write_file("/tasks.md", task_list)`\n'
        f"   - Update as you progress\n\n"
        f"**File Organization:**\n"
        f"- `/memory.md` - **REQUIRED** - Quick reference for essential facts, preferences, key decisions\n"
        f"- `/notes/` - Detailed learnings and observations (optional)\n"
        f"- `/plans/` - Project planning (optional)\n"
        f"- `/tasks.md` - Task tracking (optional)\n"
        f"- Or organize however makes sense to you\n\n"
        f"**Key Principles:**\n"
        f"- **memory.md is your anchor for essentials - read it first, update it with key info**\n"
        f"- **Use the full filesystem - memory.md for essentials, other files for details**\n"
        f"- Search before acting. Document as you go. Plan before executing.\n"
        f"- Build knowledge over time. Your workspace persists across conversations."
      )

    # Ask user for input tool (conditionally included)
    if self.enable_ask_for_user_input:
      instructions.append(
        "### User Input\n"
        "You can pause and ask the user for additional input using the `ask_user_for_input` tool.\n"
        "Use this when you need:\n"
        "- Clarification on ambiguous requests\n"
        "- Additional information not yet provided\n"
        "- User decisions or preferences\n"
        "- Confirmation before taking significant actions\n\n"
        "The conversation will pause until the user responds with their input."
      )

    # Subagent capabilities (conditionally included)
    if self.subagent_configs and len(self.subagent_configs) > 0:
      instructions.append("### Subagents\n" + "You can delegate work to specialized subagents:")

      subagent_list = []
      for name, config in self.subagent_configs.items():
        subagent_instr = config.get("instructions", "No description available")
        # Take first line or first 100 chars of instructions as description
        description = subagent_instr.split("\n")[0][:100]
        subagent_list.append(f"- `{name}`: {description}")

      instructions.append("\n".join(subagent_list))

      instructions.append(
        "\nAvailable subagent tools:\n"
        "- `start_subagent`: Start a new subagent instance\n"
        "- `delegate_to_subagent`: Delegate a task to a subagent and wait for result\n"
        "- `delegate_to_subagents_parallel`: Delegate tasks to multiple subagents in parallel\n"
        "- `list_subagents`: List all running subagent instances\n"
        "- `stop_subagent`: Stop a running subagent instance"
      )

    return "\n\n".join(instructions)

  async def get_messages(self, memory: Memory, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
    """Generate framework instructions message."""
    if not self.enabled:
      return []

    framework_text = self._generate_framework_instructions()

    logger.debug(f"[CONTEXT→{self.name}] Adding framework instructions")

    return [{"role": "system", "content": {"text": framework_text, "type": "text"}, "phase": "system"}]


class ContextTemplate:
  """
  Context template that orchestrates sections to build model input context.

  The template manages an ordered list of sections and combines their
  messages to create the final context sent to the language model.

  Flow:
    1. Initialize template with sections in desired order
    2. Call build_context(scope, conversation)
    3. Each enabled section contributes messages
    4. Messages are combined in section order
    5. Final message list is returned

  Example:
    ```python
    template = ContextTemplate(
      [
        SystemInstructionsSection(instructions),
        AdditionalContextSection(name="knowledge"),
        ConversationHistorySection(memory),
      ]
    )

    # Build context for a conversation
    messages = await template.build_context("user123", "conv456")

    # Disable a section temporarily
    template.get_section("knowledge").set_enabled(False)
    ```

  Benefits:
  - Clear separation of concerns
  - Easy to add/remove/reorder sections
  - Individual sections can be controlled
  - Shared context allows sections to communicate
  - Extensible for custom use cases
  """

  def __init__(self, sections: List[ContextSection]):
    """
    Initialize context template.

    Args:
      sections: Ordered list of context sections
    """
    self.sections = sections
    logger.debug(f"[CONTEXT] Initialized template with {len(sections)} section(s)")

  def get_section(self, name: str) -> Optional[ContextSection]:
    """
    Retrieve a section by name.

    Args:
      name: Section identifier

    Returns:
      ContextSection if found, None otherwise
    """
    for section in self.sections:
      if section.name == name:
        return section
    return None

  def add_section(self, section: ContextSection, index: Optional[int] = None):
    """
    Add a section to the template.

    Args:
      section: Section to add
      index: Optional position to insert at (default: append to end)
    """
    if index is not None:
      self.sections.insert(index, section)
      logger.debug(f"[CONTEXT] Inserted section '{section.name}' at index {index}")
    else:
      self.sections.append(section)
      logger.debug(f"[CONTEXT] Added section '{section.name}'")

  def remove_section(self, name: str) -> bool:
    """
    Remove a section by name.

    Args:
      name: Section identifier

    Returns:
      True if section was removed, False if not found
    """
    for i, section in enumerate(self.sections):
      if section.name == name:
        self.sections.pop(i)
        logger.debug(f"[CONTEXT] Removed section '{name}'")
        return True
    return False

  async def build_context(
    self, memory: Memory, scope: str, conversation: str, params: Optional[Dict[str, Any]] = None
  ) -> List[dict]:
    """
    Build context by combining all enabled sections.

    Args:
      memory: Memory instance for retrieving conversation history
      scope: User/tenant identifier for memory isolation
      conversation: Conversation thread identifier
      params: Optional dict of parameters to share with sections

    Returns:
      List of message dicts for model input
    """
    logger.debug(f"[CONTEXT→BUILD] Building context for {scope}/{conversation}")

    # Shared parameters dictionary for sections to communicate
    section_params: Dict[str, Any] = params.copy() if params else {}

    all_messages = []
    section_counts = []

    for section in self.sections:
      if not section.enabled:
        logger.debug(f"[CONTEXT→BUILD] Skipping disabled section '{section.name}'")
        continue

      try:
        messages = await section.get_messages(memory, scope, conversation, section_params)
        message_count = len(messages)
        all_messages.extend(messages)
        section_counts.append(f"{section.name}={message_count}")

      except Exception as e:
        logger.error(f"[CONTEXT→BUILD] Section '{section.name}' failed: {e}")
        # Continue with other sections even if one fails

    logger.debug(f"[CONTEXT→BUILD] Built context with {len(all_messages)} total messages ({', '.join(section_counts)})")

    return all_messages

  def get_section_names(self) -> List[str]:
    """
    Get names of all sections in order.

    Returns:
      List of section names
    """
    return [section.name for section in self.sections]

  def __repr__(self) -> str:
    """String representation of template."""
    section_names = ", ".join(self.get_section_names())
    return f"ContextTemplate([{section_names}])"


def create_default_template(
  instructions: List[dict],
  enable_ask_for_user_input: bool = False,
  enable_filesystem: bool = False,
  filesystem_visibility: str = FILESYSTEM_VISIBILITY_DEFAULT,
  subagent_configs: Optional[Dict[str, Any]] = None,
  enable_workspace_reminder: bool = False,
  workspace_reminder_frequency: int = 5,
) -> ContextTemplate:
  """
  Create a default context template with framework instructions.

  Creates a template with the standard structure:
  1. System instructions (user-provided)
  2. Framework instructions (built-in tools and capabilities)
  3. Workspace reminder (optional, if enabled)
  4. Conversation history

  Args:
    instructions: System instruction messages
    enable_ask_for_user_input: Whether ask_user_for_input tool is enabled
    enable_filesystem: Whether filesystem tools are enabled
    filesystem_visibility: Filesystem visibility level
    subagent_configs: Optional dict of configured subagents
    enable_workspace_reminder: Whether to add periodic workspace reminder (default: False)
    workspace_reminder_frequency: How often to show reminder in turns (default: 5)

  Returns:
    ContextTemplate configured with default sections
  """
  sections = [
    SystemInstructionsSection(instructions),
    FrameworkInstructionsSection(
      enable_ask_for_user_input=enable_ask_for_user_input,
      enable_filesystem=enable_filesystem,
      filesystem_visibility=filesystem_visibility,
      subagent_configs=subagent_configs,
    ),
  ]

  # Add workspace reminder if explicitly enabled
  if enable_workspace_reminder:
    sections.append(WorkspaceReminderSection(frequency=workspace_reminder_frequency))

  # Conversation history always comes last
  sections.append(ConversationHistorySection())

  return ContextTemplate(sections)

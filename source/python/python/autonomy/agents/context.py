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
   - Defines the interface: `get_messages(scope, conversation, context)`
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
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. SystemInstructionsSection                        │  │
│  │     → Returns: [system message]                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. AdditionalContextSection                         │  │
│  │     → Returns: [context message 1, context msg 2]    │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  3. ConversationHistorySection                       │  │
│  │     → Returns: [user msg, assistant msg, ...]        │  │
│  └──────────────────────────────────────────────────────┘  │
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
    ConversationHistorySection(memory),
  ]
)

# Build context
messages = await template.build_context(scope="user123", conversation="chat456")

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

  async def get_messages(self, scope, conversation, context):
    # Retrieve relevant documents
    docs = await self.vector_db.search(query)
    return [format_doc_as_message(doc) for doc in docs]


# Option 2: Duck typing - no inheritance required
class SimpleSection:
  def __init__(self):
    self.name = "simple"
    self.enabled = True

  async def get_messages(self, scope, conversation, context):
    return [{"role": "system", "content": {"text": "Hello", "type": "text"}}]
```

### Filtering History

```python
def exclude_tool_calls(msg):
  return "tool_calls" not in msg


history = ConversationHistorySection(memory, filter_fn=exclude_tool_calls, max_messages=50)
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
  async def get_messages(self, scope, conversation, params):
    params["sentiment"] = analyze_sentiment(conversation)
    return []


# Section 2: Use stored data
class ResponseSection(ContextSection):
  async def get_messages(self, scope, conversation, params):
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
  async def get_messages(self, scope, conversation, params):
    msg_count = params.get("conversation_message_count", 0)
    if msg_count > 20:
      summary = await generate_summary(scope, conversation)
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

from typing import Dict, List, Optional, Callable, Any
from ..logs import get_logger
from ..memory.memory import Memory

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

  async def get_messages(self, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
    """
    Retrieve messages for this section.

    This is the core method that sections must implement. Override this
    in subclasses or implement it in duck-typed sections.

    Args:
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

  async def get_messages(self, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
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
    section = ConversationHistorySection(memory)
    # Or with filtering:
    section = ConversationHistorySection(memory, filter_fn=lambda msg: msg.get("role") != "system")
    ```
  """

  def __init__(
    self,
    memory: Memory,
    enabled: bool = True,
    max_messages: Optional[int] = None,
    filter_fn: Optional[Callable[[dict], bool]] = None,
    transform_fn: Optional[Callable[[dict], dict]] = None,
  ):
    """
    Initialize conversation history section.

    Args:
      memory: Memory instance to retrieve messages from
      enabled: Whether section is active (default: True)
      max_messages: Optional limit on number of messages to include
      filter_fn: Optional function to filter messages (returns True to include)
      transform_fn: Optional function to transform each message before inclusion
    """
    super().__init__("conversation_history", enabled)
    self.memory = memory
    self.max_messages = max_messages
    self.filter_fn = filter_fn
    self.transform_fn = transform_fn

  async def get_messages(self, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
    """Retrieve conversation history from memory."""
    if not self.enabled:
      return []

    # Get messages from memory (without system instructions)
    messages = await self.memory.get_messages_only(scope, conversation)

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

  async def get_messages(self, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
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

  async def get_messages(self, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
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

  async def get_messages(self, scope: str, conversation: str, params: Dict[str, Any]) -> List[dict]:
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

  async def build_context(self, scope: str, conversation: str, params: Optional[Dict[str, Any]] = None) -> List[dict]:
    """
    Build context by combining all enabled sections.

    Args:
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
        messages = await section.get_messages(scope, conversation, section_params)
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
  memory: Memory,
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
    memory: Memory instance
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
  sections.append(ConversationHistorySection(memory))

  return ContextTemplate(sections)

"""
Unit tests for the context template system.

Tests cover:
- Basic template construction and building
- Section ordering and management
- Filtering and transformation
- Shared context between sections
- Dynamic section control
- Custom sections
- Async summarized history section
- Context summary configuration and helper functions
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from autonomy.agents.context import (
  ContextTemplate,
  ContextSection,
  SystemInstructionsSection,
  ConversationHistorySection,
  AdditionalContextSection,
  SummarizedHistorySection,
  create_default_template,
)
from autonomy.agents.agent import (
  ContextSummaryConfig,
  _compute_memory_config,
  _create_summarized_template,
  MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT,
  MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT,
)
from autonomy.memory.memory import Memory
from autonomy.models.model import Model


class MockMemory:
  """Mock memory for testing."""

  def __init__(self):
    self.instructions = [{"role": "system", "content": {"text": "You are a test assistant", "type": "text"}}]
    self.messages = []

  async def get_messages_only(self, scope: str, conversation: str):
    """Return mock messages."""
    return self.messages.copy()


@pytest.mark.asyncio
async def test_system_instructions_section():
  """Test SystemInstructionsSection basic functionality."""
  instructions = [{"role": "system", "content": {"text": "Test instruction", "type": "text"}}]
  memory = MockMemory()

  section = SystemInstructionsSection(instructions)
  assert section.name == "system_instructions"
  assert section.enabled is True

  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert len(messages) == 1
  assert messages[0]["role"] == "system"
  assert messages[0]["content"]["text"] == "Test instruction"


@pytest.mark.asyncio
async def test_system_instructions_section_disabled():
  """Test that disabled section returns no messages."""
  instructions = [{"role": "system", "content": {"text": "Test instruction", "type": "text"}}]
  memory = MockMemory()

  section = SystemInstructionsSection(instructions, enabled=False)
  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert len(messages) == 0


@pytest.mark.asyncio
async def test_conversation_history_section():
  """Test ConversationHistorySection basic functionality."""
  memory = MockMemory()
  memory.messages = [
    {"role": "user", "content": {"text": "Hello", "type": "text"}},
    {"role": "assistant", "content": {"text": "Hi there", "type": "text"}},
  ]

  section = ConversationHistorySection()
  context = {}
  messages = await section.get_messages(memory, "scope1", "conv1", context)

  assert len(messages) == 2
  assert messages[0]["role"] == "user"
  assert messages[1]["role"] == "assistant"
  assert context["conversation_message_count"] == 2


@pytest.mark.asyncio
async def test_conversation_history_section_max_messages():
  """Test message count limiting."""
  memory = MockMemory()
  memory.messages = [{"role": "user", "content": {"text": f"Message {i}", "type": "text"}} for i in range(10)]

  section = ConversationHistorySection(max_messages=5)
  messages = await section.get_messages(memory, "scope1", "conv1", {})

  # Should keep only the most recent 5 messages
  assert len(messages) == 5
  assert messages[0]["content"]["text"] == "Message 5"
  assert messages[4]["content"]["text"] == "Message 9"


@pytest.mark.asyncio
async def test_context_section_base_class():
  """Test that ContextSection base class works with duck typing."""
  memory = MockMemory()

  # Can instantiate directly (no longer abstract)
  section = ContextSection("test")
  assert section.name == "test"
  assert section.enabled is True

  # Default implementation returns empty list
  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert messages == []


@pytest.mark.asyncio
async def test_conversation_history_section_filter():
  """Test message filtering."""
  memory = MockMemory()
  memory.messages = [
    {"role": "user", "content": {"text": "Keep this", "type": "text"}},
    {"role": "system", "content": {"text": "Filter this", "type": "text"}},
    {"role": "user", "content": {"text": "Keep this too", "type": "text"}},
  ]

  def filter_system(msg):
    return msg.get("role") != "system"

  section = ConversationHistorySection(filter_fn=filter_system)
  messages = await section.get_messages(memory, "scope1", "conv1", {})

  assert len(messages) == 2
  assert all(msg["role"] == "user" for msg in messages)


@pytest.mark.asyncio
async def test_conversation_history_section_transform():
  """Test message transformation."""
  memory = MockMemory()
  memory.messages = [
    {"role": "user", "content": {"text": "hello", "type": "text"}},
  ]

  def uppercase_transform(msg):
    transformed = msg.copy()
    if "content" in transformed:
      transformed["content"] = {"text": transformed["content"]["text"].upper(), "type": "text"}
    return transformed

  section = ConversationHistorySection(transform_fn=uppercase_transform)
  messages = await section.get_messages(memory, "scope1", "conv1", {})

  assert messages[0]["content"]["text"] == "HELLO"


@pytest.mark.asyncio
async def test_additional_context_section_static():
  """Test AdditionalContextSection with static messages."""
  memory = MockMemory()
  section = AdditionalContextSection(name="test_context")

  # Add static messages
  section.add_message({"role": "system", "content": {"text": "Context 1", "type": "text"}})
  section.add_message({"role": "system", "content": {"text": "Context 2", "type": "text"}})

  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert len(messages) == 2
  assert messages[0]["content"]["text"] == "Context 1"
  assert messages[1]["content"]["text"] == "Context 2"

  # Test clear
  section.clear_messages()
  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert len(messages) == 0


@pytest.mark.asyncio
async def test_additional_context_section_provider():
  """Test AdditionalContextSection with provider function."""
  memory = MockMemory()

  async def test_provider(scope, conversation, context):
    return [
      {"role": "system", "content": {"text": f"Scope: {scope}", "type": "text"}},
      {"role": "system", "content": {"text": f"Conv: {conversation}", "type": "text"}},
    ]

  section = AdditionalContextSection(provider_fn=test_provider)
  messages = await section.get_messages(memory, "user123", "chat456", {})

  assert len(messages) == 2
  assert messages[0]["content"]["text"] == "Scope: user123"
  assert messages[1]["content"]["text"] == "Conv: chat456"


@pytest.mark.asyncio
async def test_additional_context_section_combined():
  """Test AdditionalContextSection with both static and provider."""
  memory = MockMemory()

  async def test_provider(scope, conversation, context):
    return [{"role": "system", "content": {"text": "From provider", "type": "text"}}]

  section = AdditionalContextSection(provider_fn=test_provider)
  section.add_message({"role": "system", "content": {"text": "Static", "type": "text"}})

  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert len(messages) == 2
  assert messages[0]["content"]["text"] == "Static"
  assert messages[1]["content"]["text"] == "From provider"


@pytest.mark.asyncio
async def test_context_template_build():
  """Test building context from multiple sections."""
  memory = MockMemory()
  memory.messages = [
    {"role": "user", "content": {"text": "Hello", "type": "text"}},
  ]

  instructions = [{"role": "system", "content": {"text": "Instructions", "type": "text"}}]

  template = ContextTemplate(
    [
      SystemInstructionsSection(instructions),
      ConversationHistorySection(),
    ]
  )

  messages = await template.build_context(memory, "scope1", "conv1")

  assert len(messages) == 2
  assert messages[0]["role"] == "system"
  assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_context_template_section_order():
  """Test that sections are processed in order."""
  memory = MockMemory()
  results = []

  class OrderTrackingSection(ContextSection):
    def __init__(self, name, order_num):
      super().__init__(name)
      self.order_num = order_num

    async def get_messages(self, memory, scope, conversation, context):
      results.append(self.order_num)
      return []

  template = ContextTemplate(
    [
      OrderTrackingSection("first", 1),
      OrderTrackingSection("second", 2),
      OrderTrackingSection("third", 3),
    ]
  )

  await template.build_context(memory, "scope1", "conv1")
  assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_context_template_shared_context():
  """Test that sections can share data via context dict."""
  memory = MockMemory()

  class WriterSection(ContextSection):
    async def get_messages(self, memory, scope, conversation, params):
      params["shared_value"] = 42
      return []

  class ReaderSection(ContextSection):
    async def get_messages(self, memory, scope, conversation, params):
      value = params.get("shared_value", 0)
      return [{"role": "system", "content": {"text": f"Value: {value}", "type": "text"}}]

  template = ContextTemplate(
    [
      WriterSection("writer"),
      ReaderSection("reader"),
    ]
  )

  messages = await template.build_context(memory, "scope1", "conv1")
  assert len(messages) == 1
  assert messages[0]["content"]["text"] == "Value: 42"


@pytest.mark.asyncio
async def test_context_template_get_section():
  """Test getting sections by name."""
  section1 = SystemInstructionsSection([])
  section2 = AdditionalContextSection(name="custom")

  template = ContextTemplate([section1, section2])

  retrieved = template.get_section("system_instructions")
  assert retrieved is section1

  retrieved = template.get_section("custom")
  assert retrieved is section2

  retrieved = template.get_section("nonexistent")
  assert retrieved is None


@pytest.mark.asyncio
async def test_context_template_add_section():
  """Test adding sections to template."""
  template = ContextTemplate(
    [
      SystemInstructionsSection([]),
    ]
  )

  assert len(template.sections) == 1

  # Add at end
  new_section = AdditionalContextSection(name="new")
  template.add_section(new_section)
  assert len(template.sections) == 2
  assert template.sections[1] is new_section

  # Add at specific index
  insert_section = AdditionalContextSection(name="inserted")
  template.add_section(insert_section, index=1)
  assert len(template.sections) == 3
  assert template.sections[1] is insert_section


@pytest.mark.asyncio
async def test_context_template_remove_section():
  """Test removing sections from template."""
  template = ContextTemplate(
    [
      SystemInstructionsSection([]),
      AdditionalContextSection(name="to_remove"),
    ]
  )

  assert len(template.sections) == 2

  # Remove existing section
  result = template.remove_section("to_remove")
  assert result is True
  assert len(template.sections) == 1

  # Try to remove non-existent section
  result = template.remove_section("nonexistent")
  assert result is False
  assert len(template.sections) == 1


@pytest.mark.asyncio
async def test_context_template_get_section_names():
  """Test getting section names."""
  template = ContextTemplate(
    [
      SystemInstructionsSection([]),
      AdditionalContextSection(name="custom1"),
      AdditionalContextSection(name="custom2"),
    ]
  )

  names = template.get_section_names()
  assert names == ["system_instructions", "custom1", "custom2"]


@pytest.mark.asyncio
async def test_context_template_disabled_section():
  """Test that disabled sections don't contribute messages."""
  memory = MockMemory()
  instructions = [{"role": "system", "content": {"text": "Instructions", "type": "text"}}]

  section1 = SystemInstructionsSection(instructions)
  section2 = AdditionalContextSection(name="extra")
  section2.add_message({"role": "system", "content": {"text": "Extra", "type": "text"}})
  section2.set_enabled(False)

  template = ContextTemplate([section1, section2])
  messages = await template.build_context(memory, "scope1", "conv1")

  # Only section1 should contribute
  assert len(messages) == 1
  assert messages[0]["content"]["text"] == "Instructions"


@pytest.mark.asyncio
async def test_context_template_section_error_handling():
  """Test that template continues if a section fails."""
  memory = MockMemory()

  class FailingSection(ContextSection):
    async def get_messages(self, memory, scope, conversation, context):
      raise ValueError("Test error")

  instructions = [{"role": "system", "content": {"text": "Instructions", "type": "text"}}]

  template = ContextTemplate(
    [
      SystemInstructionsSection(instructions),
      FailingSection("failing"),
      AdditionalContextSection(name="after_failure"),
    ]
  )

  # Add message to the last section
  template.sections[2].add_message({"role": "system", "content": {"text": "After failure", "type": "text"}})

  messages = await template.build_context(memory, "scope1", "conv1")

  # Should get messages from sections that didn't fail
  assert len(messages) == 2
  assert messages[0]["content"]["text"] == "Instructions"
  assert messages[1]["content"]["text"] == "After failure"


@pytest.mark.asyncio
async def test_create_default_template():
  """Test creating default template."""
  memory = MockMemory()
  template = create_default_template(memory.instructions)

  assert len(template.sections) == 3
  assert template.get_section_names() == ["system_instructions", "framework_instructions", "conversation_history"]


def test_context_template_repr():
  """Test string representation of template."""
  template = ContextTemplate(
    [
      SystemInstructionsSection([]),
      AdditionalContextSection(name="custom"),
    ]
  )

  repr_str = repr(template)
  assert "ContextTemplate" in repr_str
  assert "system_instructions" in repr_str
  assert "custom" in repr_str


@pytest.mark.asyncio
async def test_section_set_enabled():
  """Test enabling and disabling sections."""
  memory = MockMemory()
  section = SystemInstructionsSection([{"role": "system", "content": {"text": "Test", "type": "text"}}])

  # Initially enabled
  assert section.enabled is True
  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert len(messages) == 1

  # Disable
  section.set_enabled(False)
  assert section.enabled is False
  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert len(messages) == 0

  # Re-enable
  section.set_enabled(True)
  assert section.enabled is True
  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert len(messages) == 1


@pytest.mark.asyncio
async def test_duck_typing_custom_section():
  """Test that duck typing works - sections don't need to inherit from ContextSection."""
  memory = MockMemory()

  # Create a custom section class that doesn't inherit from ContextSection
  class DuckTypedSection:
    def __init__(self, name):
      self.name = name
      self.enabled = True

    async def get_messages(self, memory, scope, conversation, context):
      return [{"role": "system", "content": {"text": f"Duck typed: {scope}", "type": "text"}}]

  # Should work fine in a template
  duck_section = DuckTypedSection("duck")
  template = ContextTemplate([duck_section])

  messages = await template.build_context(memory, "user1", "conv1")
  assert len(messages) == 1
  assert messages[0]["content"]["text"] == "Duck typed: user1"


# ============================================================================
# SummarizedHistorySection Tests
# ============================================================================


class MockModel:
  """Mock model for testing SummarizedHistorySection."""

  def __init__(self, response_text: str = "Test summary"):
    self.response_text = response_text
    self.call_count = 0

  async def complete_chat(self, messages, stream=False, max_tokens=None):
    """Mock complete_chat method."""
    self.call_count += 1
    # Simulate async delay
    await asyncio.sleep(0.01)
    # Return mock response
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = self.response_text
    mock_response.content = [mock_content]
    return mock_response


@pytest.mark.asyncio
async def test_summarized_history_section_init():
  """Test SummarizedHistorySection initialization."""
  mock_model = MockModel()

  section = SummarizedHistorySection(
    summary_model=mock_model,
    floor=10,
    ceiling=20,
    size=500,
  )

  assert section.name == "conversation_history"
  assert section.enabled is True
  assert section.floor == 10
  assert section.ceiling == 20
  assert section.size == 500
  assert section.batch_size == 10  # ceiling - floor
  assert section._summary_cache == {}
  assert section._summarized_count == {}


@pytest.mark.asyncio
async def test_summarized_history_section_init_legacy_params():
  """Test SummarizedHistorySection initialization with deprecated parameter names."""
  mock_model = MockModel()

  # Using legacy parameter names should still work
  section = SummarizedHistorySection(
    summary_model=mock_model,
    recent_count=10,
    summarize_threshold=20,
    max_summary_tokens=500,
  )

  assert section.name == "conversation_history"
  assert section.enabled is True
  assert section.floor == 10  # mapped from recent_count
  assert section.ceiling == 20  # mapped from summarize_threshold
  assert section.size == 500  # mapped from max_summary_tokens


@pytest.mark.asyncio
async def test_summarized_history_section_init_validation():
  """Test SummarizedHistorySection validation errors."""
  mock_model = MockModel()

  # floor >= ceiling should raise error
  with pytest.raises(ValueError, match="floor.*must be less than ceiling"):
    SummarizedHistorySection(
      summary_model=mock_model,
      floor=20,
      ceiling=10,
    )

  # floor < 1 should raise error
  with pytest.raises(ValueError, match="floor.*must be at least 1"):
    SummarizedHistorySection(
      summary_model=mock_model,
      floor=0,
      ceiling=10,
    )


@pytest.mark.asyncio
async def test_summarized_history_section_below_threshold():
  """Test that messages below ceiling are returned as-is without summarization."""
  mock_model = MockModel()
  memory = MockMemory()
  memory.messages = [{"role": "user", "content": {"text": f"Message {i}", "type": "text"}} for i in range(15)]

  section = SummarizedHistorySection(
    summary_model=mock_model,
    floor=10,
    ceiling=20,  # 15 < 20, so no summarization
  )

  context = {}
  messages = await section.get_messages(memory, "scope1", "conv1", context)

  assert len(messages) == 15
  assert context["conversation_message_count"] == 15
  assert mock_model.call_count == 0  # No summarization call


@pytest.mark.asyncio
async def test_summarized_history_section_cold_start():
  """Test cold start behavior - returns all messages when no cache exists."""
  mock_model = MockModel("Test summary for cold start")
  memory = MockMemory()
  memory.messages = [{"role": "user", "content": {"text": f"Message {i}", "type": "text"}} for i in range(25)]

  section = SummarizedHistorySection(
    summary_model=mock_model,
    recent_count=10,
    summarize_threshold=20,  # 25 > 20, summarization triggered
  )

  context = {}
  messages = await section.get_messages(memory, "scope1", "conv1", context)

  # Cold start: returns all messages (no blocking on summarization)
  assert len(messages) == 25
  assert context["conversation_message_count"] == 25
  # Background task may be started but we don't wait for it
  assert section._metrics["cache_misses"] == 1


@pytest.mark.asyncio
async def test_summarized_history_section_cache_hit():
  """Test that cached summaries are returned immediately."""
  mock_model = MockModel("Cached summary content")
  memory = MockMemory()
  memory.messages = [{"role": "user", "content": {"text": f"Message {i}", "type": "text"}} for i in range(25)]

  section = SummarizedHistorySection(
    summary_model=mock_model,
    recent_count=10,
    summarize_threshold=20,
  )

  # Pre-populate cache (simulating previous summarization)
  cache_key = "scope1:conv1"
  section._summary_cache[cache_key] = "Pre-cached summary"
  section._summarized_count[cache_key] = 15  # Summarized 15 old messages

  context = {}
  messages = await section.get_messages(memory, "scope1", "conv1", context)

  # Should return summary message + 10 recent messages
  assert len(messages) == 11
  assert messages[0]["role"] == "system"
  assert "CONVERSATION SUMMARY" in messages[0]["content"]["text"]
  assert "Pre-cached summary" in messages[0]["content"]["text"]
  assert section._metrics["cache_hits"] == 1


@pytest.mark.asyncio
async def test_summarized_history_section_disabled():
  """Test that disabled section returns no messages."""
  mock_model = MockModel()
  memory = MockMemory()
  memory.messages = [{"role": "user", "content": {"text": "Hello", "type": "text"}}]

  section = SummarizedHistorySection(
    summary_model=mock_model,
    enabled=False,
  )

  messages = await section.get_messages(memory, "scope1", "conv1", {})
  assert len(messages) == 0


@pytest.mark.asyncio
async def test_summarized_history_section_format_messages():
  """Test message formatting for summarization."""
  mock_model = MockModel()

  section = SummarizedHistorySection(summary_model=mock_model)

  messages = [
    {"role": "user", "content": {"text": "Hello", "type": "text"}},
    {"role": "assistant", "content": {"text": "Hi there", "type": "text"}},
    {"role": "user", "content": "Plain string content"},
    {"role": "tool", "content": "Tool output"},  # Should be skipped
  ]

  formatted = section._format_messages_for_summary(messages)

  assert "User: Hello" in formatted
  assert "Assistant: Hi there" in formatted
  assert "User: Plain string content" in formatted
  # Tool messages should be excluded
  assert "Tool output" not in formatted


@pytest.mark.asyncio
async def test_summarized_history_section_format_tool_calls():
  """Test formatting of assistant messages with tool calls."""
  mock_model = MockModel()

  section = SummarizedHistorySection(summary_model=mock_model)

  messages = [
    {
      "role": "assistant",
      "content": {"text": "Let me search for that", "type": "text"},
      "tool_calls": [{"name": "search"}, {"name": "retrieve"}],
    },
  ]

  formatted = section._format_messages_for_summary(messages)

  assert "Used tools: search, retrieve" in formatted
  assert "Let me search for that" in formatted


@pytest.mark.asyncio
async def test_summarized_history_section_create_summary_message():
  """Test summary message creation."""
  mock_model = MockModel()

  section = SummarizedHistorySection(summary_model=mock_model)

  # Without staleness
  message = section._create_summary_message("Test summary", 10)
  assert message["role"] == "system"
  assert "CONVERSATION SUMMARY - 10 earlier messages" in message["content"]["text"]
  assert "Test summary" in message["content"]["text"]
  assert "END SUMMARY" in message["content"]["text"]

  # With staleness
  message_stale = section._create_summary_message("Test summary", 10, stale_count=3)
  assert "may be 3 messages behind" in message_stale["content"]["text"]


@pytest.mark.asyncio
async def test_summarized_history_section_get_metrics():
  """Test metrics retrieval."""
  mock_model = MockModel()

  section = SummarizedHistorySection(summary_model=mock_model)

  metrics = section.get_metrics()

  assert "cache_hits" in metrics
  assert "cache_misses" in metrics
  assert "background_tasks_started" in metrics
  assert "background_tasks_completed" in metrics
  assert "background_tasks_failed" in metrics


@pytest.mark.asyncio
async def test_summarized_history_section_clear_cache():
  """Test cache clearing."""
  mock_model = MockModel()

  section = SummarizedHistorySection(summary_model=mock_model)

  # Populate cache
  section._summary_cache["scope1:conv1"] = "Summary 1"
  section._summary_cache["scope1:conv2"] = "Summary 2"
  section._summarized_count["scope1:conv1"] = 10
  section._summarized_count["scope1:conv2"] = 15

  # Clear specific conversation
  section.clear_cache(scope="scope1", conversation="conv1")
  assert "scope1:conv1" not in section._summary_cache
  assert "scope1:conv2" in section._summary_cache

  # Clear all
  section.clear_cache()
  assert len(section._summary_cache) == 0
  assert len(section._summarized_count) == 0


@pytest.mark.asyncio
async def test_summarized_history_section_get_cache_info():
  """Test cache info retrieval."""
  mock_model = MockModel()

  section = SummarizedHistorySection(summary_model=mock_model)

  # Empty cache
  info = section.get_cache_info()
  assert info["cached_conversations"] == 0
  assert info["pending_summarizations"] == 0
  assert info["conversations"] == []

  # Populated cache
  section._summary_cache["scope1:conv1"] = "Summary 1"
  section._summary_cache["scope2:conv2"] = "Summary 2"

  info = section.get_cache_info()
  assert info["cached_conversations"] == 2
  assert "scope1:conv1" in info["conversations"]
  assert "scope2:conv2" in info["conversations"]


@pytest.mark.asyncio
async def test_summarized_history_section_background_summarization():
  """Test background summarization completes and updates cache."""
  mock_model = MockModel("Background summary result")
  memory = MockMemory()
  memory.messages = [{"role": "user", "content": {"text": f"Message {i}", "type": "text"}} for i in range(25)]

  section = SummarizedHistorySection(
    summary_model=mock_model,
    recent_count=10,
    summarize_threshold=20,
  )

  # First call - cold start, triggers background summarization
  context = {}
  messages = await section.get_messages(memory, "scope1", "conv1", context)
  assert len(messages) == 25  # Returns all messages on cold start

  # Wait for background task to complete
  await asyncio.sleep(0.1)

  # Second call - should use cached summary
  context2 = {}
  messages2 = await section.get_messages(memory, "scope1", "conv1", context2)

  # Should return summary + recent messages
  assert len(messages2) == 11  # 1 summary + 10 recent
  assert "CONVERSATION SUMMARY" in messages2[0]["content"]["text"]
  assert section._metrics["cache_hits"] >= 1


@pytest.mark.asyncio
async def test_summarized_history_section_stale_cache():
  """Test stale cache detection and background update.

  The new algorithm ensures NO HIDDEN MESSAGES by starting the verbatim
  window right after where the summary ends (at summarized_count index).
  """
  mock_model = MockModel("Updated summary")
  memory = MockMemory()

  section = SummarizedHistorySection(
    summary_model=mock_model,
    floor=5,
    ceiling=10,
  )

  # Pre-populate cache with old summary
  cache_key = "scope1:conv1"
  section._summary_cache[cache_key] = "Old summary"
  section._summarized_count[cache_key] = 10  # Summarized first 10 messages

  # Now memory has 20 messages total
  memory.messages = [{"role": "user", "content": {"text": f"Message {i}", "type": "text"}} for i in range(20)]

  context = {}
  messages = await section.get_messages(memory, "scope1", "conv1", context)

  # NEW ALGORITHM: verbatim window starts right after summary (no hidden messages)
  # Summary covers messages 0-9 (10 messages)
  # Verbatim includes messages 10-19 (10 messages)
  # Total: 1 summary + 10 verbatim = 11 messages
  assert len(messages) == 11  # 1 summary + messages[10:20]
  assert "Old summary" in messages[0]["content"]["text"]
  # Staleness should be indicated (target_summarized=15, summarized_count=10, so 5 messages behind)
  assert "may be" in messages[0]["content"]["text"]  # staleness note


@pytest.mark.asyncio
async def test_summarized_history_section_in_template():
  """Test using SummarizedHistorySection in a ContextTemplate."""
  mock_model = MockModel("Template test summary")
  memory = MockMemory()
  memory.messages = [{"role": "user", "content": {"text": f"Message {i}", "type": "text"}} for i in range(15)]

  instructions = [{"role": "system", "content": {"text": "Test instructions", "type": "text"}}]

  section = SummarizedHistorySection(
    summary_model=mock_model,
    recent_count=10,
    summarize_threshold=20,  # Below threshold
  )

  template = ContextTemplate(
    [
      SystemInstructionsSection(instructions),
      section,
    ]
  )

  messages = await template.build_context(memory, "scope1", "conv1")

  # Should have system instructions + all conversation messages (below threshold)
  assert len(messages) == 16  # 1 instruction + 15 messages
  assert messages[0]["role"] == "system"
  assert messages[0]["content"]["text"] == "Test instructions"


@pytest.mark.asyncio
async def test_summarized_history_section_generate_summary_error():
  """Test error handling in summary generation."""
  mock_model = MockModel()
  # Make complete_chat raise an error
  mock_model.complete_chat = AsyncMock(side_effect=Exception("API error"))

  section = SummarizedHistorySection(summary_model=mock_model)

  messages = [
    {"role": "user", "content": {"text": "Hello", "type": "text"}},
    {"role": "assistant", "content": {"text": "Hi", "type": "text"}},
  ]

  # Should return fallback message on error
  summary = await section._generate_summary(messages)
  assert "[Previous conversation" in summary
  assert "2 messages" in summary


@pytest.mark.asyncio
async def test_summarized_history_section_list_content():
  """Test formatting messages with list content."""
  mock_model = MockModel()

  section = SummarizedHistorySection(summary_model=mock_model)

  messages = [
    {
      "role": "user",
      "content": [
        {"text": "Part 1", "type": "text"},
        {"text": "Part 2", "type": "text"},
      ],
    },
    {
      "role": "assistant",
      "content": [
        "Plain string part",
        {"text": "Dict part", "type": "text"},
      ],
    },
  ]

  formatted = section._format_messages_for_summary(messages)

  assert "Part 1" in formatted
  assert "Part 2" in formatted
  assert "Plain string part" in formatted
  assert "Dict part" in formatted


# =============================================================================
# CONTEXT SUMMARY CONFIGURATION TESTS
# =============================================================================


def test_compute_memory_config_with_context_summary_true():
  """Test _compute_memory_config when context_summary=True."""
  max_messages, max_tokens = _compute_memory_config(
    context_summary=True,
    max_messages_in_short_term_memory=100,  # Should be ignored
    max_tokens_in_short_term_memory=50000,  # Should be ignored
  )

  # With context_summary=True, ceiling defaults to 20
  # internal_max_messages = ceiling * 2 = 40
  assert max_messages == 40
  assert max_tokens is None  # No token limit when using summarization


def test_compute_memory_config_with_context_summary_config():
  """Test _compute_memory_config with custom ContextSummaryConfig."""
  config: ContextSummaryConfig = {
    "floor": 15,
    "ceiling": 30,
  }

  max_messages, max_tokens = _compute_memory_config(
    context_summary=config,
    max_messages_in_short_term_memory=100,
    max_tokens_in_short_term_memory=50000,
  )

  # internal_max_messages = ceiling * 2 = 60
  assert max_messages == 60
  assert max_tokens is None


def test_compute_memory_config_without_context_summary():
  """Test _compute_memory_config when context_summary is disabled."""
  max_messages, max_tokens = _compute_memory_config(
    context_summary=None,
    max_messages_in_short_term_memory=100,
    max_tokens_in_short_term_memory=50000,
  )

  assert max_messages == 100
  assert max_tokens == 50000


def test_compute_memory_config_with_context_summary_false():
  """Test _compute_memory_config when context_summary=False."""
  max_messages, max_tokens = _compute_memory_config(
    context_summary=False,
    max_messages_in_short_term_memory=200,
    max_tokens_in_short_term_memory=80000,
  )

  assert max_messages == 200
  assert max_tokens == 80000


def test_compute_memory_config_defaults():
  """Test _compute_memory_config with default values."""
  max_messages, max_tokens = _compute_memory_config(
    context_summary=None,
    max_messages_in_short_term_memory=None,
    max_tokens_in_short_term_memory=None,
  )

  assert max_messages == MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT
  assert max_tokens == MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT


def test_create_summarized_template_with_true():
  """Test _create_summarized_template with context_summary=True."""
  mock_model = MockModel()
  instructions = [{"role": "system", "content": "Test instructions"}]

  template = _create_summarized_template(
    instructions=instructions,
    context_summary=True,
    model=mock_model,
  )

  # Should have 3 sections: system, framework, summarized history
  assert len(template.sections) == 3
  assert template.sections[0].name == "system_instructions"
  assert template.sections[1].name == "framework_instructions"
  assert template.sections[2].name == "conversation_history"

  # Verify it's a SummarizedHistorySection with defaults
  history_section = template.sections[2]
  assert isinstance(history_section, SummarizedHistorySection)
  assert history_section.floor == 10
  assert history_section.ceiling == 20
  assert history_section.size == 2048


def test_create_summarized_template_with_custom_config():
  """Test _create_summarized_template with custom ContextSummaryConfig."""
  mock_model = MockModel()
  custom_model = MockModel()
  instructions = [{"role": "system", "content": "Test instructions"}]

  config: ContextSummaryConfig = {
    "floor": 15,
    "ceiling": 30,
    "size": 1024,
    "model": custom_model,
    "instructions": "Preserve code snippets",
  }

  template = _create_summarized_template(
    instructions=instructions,
    context_summary=config,
    model=mock_model,  # Should be overridden by config.model
  )

  history_section = template.sections[2]
  assert isinstance(history_section, SummarizedHistorySection)
  assert history_section.floor == 15
  assert history_section.ceiling == 30
  assert history_section.size == 1024
  assert history_section.summary_model == custom_model
  assert history_section.instructions == "Preserve code snippets"


def test_create_summarized_template_with_features():
  """Test _create_summarized_template with agent features enabled."""
  mock_model = MockModel()
  instructions = [{"role": "system", "content": "Test instructions"}]

  template = _create_summarized_template(
    instructions=instructions,
    context_summary=True,
    model=mock_model,
    enable_ask_for_user_input=True,
    enable_filesystem=True,
    filesystem_visibility="scope",
    subagent_configs={"helper": {"role": "helper", "instructions": "Help"}},
  )

  # Framework section should have the features configured
  framework_section = template.sections[1]
  assert framework_section.enable_ask_for_user_input is True
  assert framework_section.enable_filesystem is True


@pytest.mark.asyncio
async def test_summarized_history_section_custom_instructions():
  """Test that custom instructions replace the entire prompt template."""
  mock_model = MockModel("Custom summary result")

  # Custom prompt template with {conversation} placeholder
  custom_prompt = """You are a code assistant. Summarize this conversation preserving all code:

{conversation}

Technical Summary:"""

  section = SummarizedHistorySection(
    summary_model=mock_model,
    floor=5,
    ceiling=10,
    instructions=custom_prompt,
  )

  messages = [
    {"role": "user", "content": {"text": "How do I print in Python?", "type": "text"}},
    {"role": "assistant", "content": {"text": "Use print('hello')", "type": "text"}},
  ]

  # Generate summary to verify custom prompt is used
  summary = await section._generate_summary(messages)

  # Verify the model was called (mock returns "Custom summary result")
  assert mock_model.call_count == 1

  # The prompt passed to the model should be our custom template with conversation inserted
  # Since we're using a mock, we just verify summarization completed successfully
  assert summary == "Custom summary result"


def test_context_summary_config_typing():
  """Test that ContextSummaryConfig accepts valid configurations."""
  # Empty config (all defaults)
  config1: ContextSummaryConfig = {}
  assert config1 == {}

  # Partial config
  config2: ContextSummaryConfig = {"floor": 5, "ceiling": 15}
  assert config2["floor"] == 5
  assert config2["ceiling"] == 15

  # Full config
  mock_model = MockModel()
  config3: ContextSummaryConfig = {
    "floor": 10,
    "ceiling": 25,
    "size": 1500,
    "model": mock_model,
    "instructions": "Keep technical details",
  }
  assert config3["floor"] == 10
  assert config3["size"] == 1500

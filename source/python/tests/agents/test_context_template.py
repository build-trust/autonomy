"""
Unit tests for the context template system.

Tests cover:
- Basic template construction and building
- Section ordering and management
- Filtering and transformation
- Shared context between sections
- Dynamic section control
- Custom sections
"""

import asyncio
import pytest
from autonomy.agents.context import (
  ContextTemplate,
  ContextSection,
  SystemInstructionsSection,
  ConversationHistorySection,
  AdditionalContextSection,
  create_default_template,
)
from autonomy.memory.memory import Memory


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

  section = SystemInstructionsSection(instructions)
  assert section.name == "system_instructions"
  assert section.enabled is True

  messages = await section.get_messages("scope1", "conv1", {})
  assert len(messages) == 1
  assert messages[0]["role"] == "system"
  assert messages[0]["content"]["text"] == "Test instruction"


@pytest.mark.asyncio
async def test_system_instructions_section_disabled():
  """Test that disabled section returns no messages."""
  instructions = [{"role": "system", "content": {"text": "Test instruction", "type": "text"}}]

  section = SystemInstructionsSection(instructions, enabled=False)
  messages = await section.get_messages("scope1", "conv1", {})
  assert len(messages) == 0


@pytest.mark.asyncio
async def test_conversation_history_section():
  """Test ConversationHistorySection basic functionality."""
  memory = MockMemory()
  memory.messages = [
    {"role": "user", "content": {"text": "Hello", "type": "text"}},
    {"role": "assistant", "content": {"text": "Hi there", "type": "text"}},
  ]

  section = ConversationHistorySection(memory)
  context = {}
  messages = await section.get_messages("scope1", "conv1", context)

  assert len(messages) == 2
  assert messages[0]["role"] == "user"
  assert messages[1]["role"] == "assistant"
  assert context["conversation_message_count"] == 2


@pytest.mark.asyncio
async def test_conversation_history_section_max_messages():
  """Test message count limiting."""
  memory = MockMemory()
  memory.messages = [{"role": "user", "content": {"text": f"Message {i}", "type": "text"}} for i in range(10)]

  section = ConversationHistorySection(memory, max_messages=5)
  messages = await section.get_messages("scope1", "conv1", {})

  # Should keep only the most recent 5 messages
  assert len(messages) == 5
  assert messages[0]["content"]["text"] == "Message 5"
  assert messages[4]["content"]["text"] == "Message 9"


@pytest.mark.asyncio
async def test_context_section_base_class():
  """Test that ContextSection base class works with duck typing."""
  # Can instantiate directly (no longer abstract)
  section = ContextSection("test")
  assert section.name == "test"
  assert section.enabled is True

  # Default implementation returns empty list
  messages = await section.get_messages("scope1", "conv1", {})
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

  section = ConversationHistorySection(memory, filter_fn=filter_system)
  messages = await section.get_messages("scope1", "conv1", {})

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

  section = ConversationHistorySection(memory, transform_fn=uppercase_transform)
  messages = await section.get_messages("scope1", "conv1", {})

  assert messages[0]["content"]["text"] == "HELLO"


@pytest.mark.asyncio
async def test_additional_context_section_static():
  """Test AdditionalContextSection with static messages."""
  section = AdditionalContextSection(name="test_context")

  # Add static messages
  section.add_message({"role": "system", "content": {"text": "Context 1", "type": "text"}})
  section.add_message({"role": "system", "content": {"text": "Context 2", "type": "text"}})

  messages = await section.get_messages("scope1", "conv1", {})
  assert len(messages) == 2
  assert messages[0]["content"]["text"] == "Context 1"
  assert messages[1]["content"]["text"] == "Context 2"

  # Test clear
  section.clear_messages()
  messages = await section.get_messages("scope1", "conv1", {})
  assert len(messages) == 0


@pytest.mark.asyncio
async def test_additional_context_section_provider():
  """Test AdditionalContextSection with provider function."""

  async def test_provider(scope, conversation, context):
    return [
      {"role": "system", "content": {"text": f"Scope: {scope}", "type": "text"}},
      {"role": "system", "content": {"text": f"Conv: {conversation}", "type": "text"}},
    ]

  section = AdditionalContextSection(provider_fn=test_provider)
  messages = await section.get_messages("user123", "chat456", {})

  assert len(messages) == 2
  assert messages[0]["content"]["text"] == "Scope: user123"
  assert messages[1]["content"]["text"] == "Conv: chat456"


@pytest.mark.asyncio
async def test_additional_context_section_combined():
  """Test AdditionalContextSection with both static and provider."""

  async def test_provider(scope, conversation, context):
    return [{"role": "system", "content": {"text": "From provider", "type": "text"}}]

  section = AdditionalContextSection(provider_fn=test_provider)
  section.add_message({"role": "system", "content": {"text": "Static", "type": "text"}})

  messages = await section.get_messages("scope1", "conv1", {})
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
      ConversationHistorySection(memory),
    ]
  )

  messages = await template.build_context("scope1", "conv1")

  assert len(messages) == 2
  assert messages[0]["role"] == "system"
  assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_context_template_section_order():
  """Test that sections are processed in order."""
  results = []

  class OrderTrackingSection(ContextSection):
    def __init__(self, name, order_num):
      super().__init__(name)
      self.order_num = order_num

    async def get_messages(self, scope, conversation, context):
      results.append(self.order_num)
      return []

  template = ContextTemplate(
    [
      OrderTrackingSection("first", 1),
      OrderTrackingSection("second", 2),
      OrderTrackingSection("third", 3),
    ]
  )

  await template.build_context("scope1", "conv1")
  assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_context_template_shared_context():
  """Test that sections can share data via context dict."""

  class WriterSection(ContextSection):
    async def get_messages(self, scope, conversation, params):
      params["shared_value"] = 42
      return []

  class ReaderSection(ContextSection):
    async def get_messages(self, scope, conversation, params):
      value = params.get("shared_value", 0)
      return [{"role": "system", "content": {"text": f"Value: {value}", "type": "text"}}]

  template = ContextTemplate(
    [
      WriterSection("writer"),
      ReaderSection("reader"),
    ]
  )

  messages = await template.build_context("scope1", "conv1")
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
  instructions = [{"role": "system", "content": {"text": "Instructions", "type": "text"}}]

  section1 = SystemInstructionsSection(instructions)
  section2 = AdditionalContextSection(name="extra")
  section2.add_message({"role": "system", "content": {"text": "Extra", "type": "text"}})
  section2.set_enabled(False)

  template = ContextTemplate([section1, section2])
  messages = await template.build_context("scope1", "conv1")

  # Only section1 should contribute
  assert len(messages) == 1
  assert messages[0]["content"]["text"] == "Instructions"


@pytest.mark.asyncio
async def test_context_template_section_error_handling():
  """Test that template continues if a section fails."""

  class FailingSection(ContextSection):
    async def get_messages(self, scope, conversation, context):
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

  messages = await template.build_context("scope1", "conv1")

  # Should get messages from sections that didn't fail
  assert len(messages) == 2
  assert messages[0]["content"]["text"] == "Instructions"
  assert messages[1]["content"]["text"] == "After failure"


@pytest.mark.asyncio
async def test_create_default_template():
  """Test creating default template."""
  memory = MockMemory()
  template = create_default_template(memory, memory.instructions)

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
  section = SystemInstructionsSection([{"role": "system", "content": {"text": "Test", "type": "text"}}])

  # Initially enabled
  assert section.enabled is True
  messages = await section.get_messages("scope1", "conv1", {})
  assert len(messages) == 1

  # Disable
  section.set_enabled(False)
  assert section.enabled is False
  messages = await section.get_messages("scope1", "conv1", {})
  assert len(messages) == 0

  # Re-enable
  section.set_enabled(True)
  assert section.enabled is True
  messages = await section.get_messages("scope1", "conv1", {})
  assert len(messages) == 1


@pytest.mark.asyncio
async def test_duck_typing_custom_section():
  """Test that duck typing works - sections don't need to inherit from ContextSection."""

  # Create a custom section class that doesn't inherit from ContextSection
  class DuckTypedSection:
    def __init__(self, name):
      self.name = name
      self.enabled = True

    async def get_messages(self, scope, conversation, context):
      return [{"role": "system", "content": {"text": f"Duck typed: {scope}", "type": "text"}}]

  # Should work fine in a template
  duck_section = DuckTypedSection("duck")
  template = ContextTemplate([duck_section])

  messages = await template.build_context("user1", "conv1")
  assert len(messages) == 1
  assert messages[0]["content"]["text"] == "Duck typed: user1"

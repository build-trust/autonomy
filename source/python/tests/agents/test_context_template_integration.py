"""
Integration tests for context templates with actual Agent instances.

These tests verify that context templates work correctly when integrated
with the full Agent system, including memory and message conversion.

NOTE: These tests require the Rust integration layer and are currently skipped.
The core context template functionality is thoroughly tested in test_context_template.py.
"""

import asyncio
import pytest
from autonomy import Node
from autonomy.agents.agent import Agent, system_message
from autonomy.agents.context import (
  ContextTemplate,
  ContextSection,
  SystemInstructionsSection,
  ConversationHistorySection,
  AdditionalContextSection,
)

pytestmark = pytest.mark.skip(reason="Requires Rust integration - core functionality tested in test_context_template.py")


class CustomTestContextSection(ContextSection):
  """Test section that tracks calls."""

  def __init__(self):
    super().__init__("test_section")
    self.call_count = 0
    self.last_scope = None
    self.last_conversation = None

  async def get_messages(self, scope, conversation, context):
    self.call_count += 1
    self.last_scope = scope
    self.last_conversation = conversation
    return [
      {
        "role": "system",
        "content": {
          "text": f"Test context (call #{self.call_count})",
          "type": "text"
        },
        "phase": "system"
      }
    ]


@pytest.mark.asyncio
async def test_agent_uses_context_template():
  """Test that agent actually uses the context template."""
  node = Node()

  # Create agent with default template
  agent = await Agent.start(
    node=node,
    name="test-agent",
    instructions="You are a test assistant.",
  )

  # Verify default template was created
  assert agent.context_template is not None
  assert len(agent.context_template.sections) == 2
  assert agent.context_template.get_section_names() == [
    "system_instructions",
    "conversation_history"
  ]

  await Agent.stop(node, "test-agent")


@pytest.mark.asyncio
async def test_agent_with_custom_section():
  """Test agent with custom context section."""
  node = Node()

  agent = await Agent.start(
    node=node,
    name="test-agent-custom",
    instructions="You are a test assistant.",
  )

  # Add custom section
  test_section = CustomTestContextSection()
  agent.context_template.add_section(test_section, index=1)

  assert len(agent.context_template.sections) == 3
  assert agent.context_template.get_section_names() == [
    "system_instructions",
    "test_section",
    "conversation_history"
  ]

  # Verify section hasn't been called yet
  assert test_section.call_count == 0

  # Get context (this should trigger the section)
  messages = await agent.determine_input_context("user1", "conv1")

  # Verify section was called
  assert test_section.call_count == 1
  assert test_section.last_scope == "test-agent-custom/user1"
  assert test_section.last_conversation == "conv1"

  # Verify messages include test section content
  message_texts = [
    msg.content if isinstance(msg.content, str)
    else msg.content.get("text", "") if hasattr(msg, "content") and isinstance(msg.content, dict)
    else str(msg.get("content", {}).get("text", ""))
    for msg in messages
  ]
  # The test section should have contributed a message
  assert any("Test context" in str(msg) for msg in messages)

  await Agent.stop(node, "test-agent-custom")


@pytest.mark.asyncio
async def test_agent_section_ordering():
  """Test that sections are processed in order."""
  node = Node()

  agent = await Agent.start(
    node=node,
    name="test-agent-order",
    instructions="You are a test assistant.",
  )

  # Clear and rebuild template with known order
  call_order = []

  class OrderSection(ContextSection):
    def __init__(self, order_num):
      super().__init__(f"order_{order_num}")
      self.order_num = order_num

    async def get_messages(self, scope, conversation, context):
      call_order.append(self.order_num)
      return []

  agent.context_template = ContextTemplate([
    OrderSection(1),
    OrderSection(2),
    OrderSection(3),
  ])

  # Build context
  await agent.determine_input_context("user1", "conv1")

  # Verify order
  assert call_order == [1, 2, 3]

  await Agent.stop(node, "test-agent-order")


@pytest.mark.asyncio
async def test_agent_dynamic_section_control():
  """Test enabling/disabling sections dynamically."""
  node = Node()

  agent = await Agent.start(
    node=node,
    name="test-agent-dynamic",
    instructions="You are a test assistant.",
  )

  # Add a test section
  test_section = CustomTestContextSection()
  agent.context_template.add_section(test_section, index=1)

  # First call with section enabled
  messages1 = await agent.determine_input_context("user1", "conv1")
  assert test_section.call_count == 1

  # Disable section
  test_section.set_enabled(False)

  # Second call with section disabled
  messages2 = await agent.determine_input_context("user1", "conv1")
  # Call count shouldn't increase when disabled
  assert test_section.call_count == 1

  # Re-enable section
  test_section.set_enabled(True)

  # Third call with section re-enabled
  messages3 = await agent.determine_input_context("user1", "conv1")
  assert test_section.call_count == 2

  await Agent.stop(node, "test-agent-dynamic")


@pytest.mark.asyncio
async def test_agent_with_provider_function():
  """Test agent with provider function in additional context."""
  node = Node()

  call_count = 0

  async def time_provider(scope, conversation, context):
    nonlocal call_count
    call_count += 1
    return [
      {
        "role": "system",
        "content": {
          "text": f"Call {call_count}",
          "type": "text"
        },
        "phase": "system"
      }
    ]

  agent = await Agent.start(
    node=node,
    name="test-agent-provider",
    instructions="You are a test assistant.",
  )

  # Add section with provider
  provider_section = AdditionalContextSection(
    name="time_provider",
    provider_fn=time_provider
  )
  agent.context_template.add_section(provider_section, index=1)

  # First call
  await agent.determine_input_context("user1", "conv1")
  assert call_count == 1

  # Second call
  await agent.determine_input_context("user1", "conv1")
  assert call_count == 2

  await Agent.stop(node, "test-agent-provider")


@pytest.mark.asyncio
async def test_agent_shared_context():
  """Test that sections can share context."""
  node = Node()

  shared_data = {}

  class WriterSection(ContextSection):
    async def get_messages(self, scope, conversation, context):
      context["test_value"] = 42
      shared_data["context_obj"] = id(context)
      return []

  class ReaderSection(ContextSection):
    async def get_messages(self, scope, conversation, context):
      shared_data["read_value"] = context.get("test_value", None)
      shared_data["same_obj"] = id(context) == shared_data.get("context_obj")
      return []

  agent = await Agent.start(
    node=node,
    name="test-agent-shared",
    instructions="You are a test assistant.",
  )

  agent.context_template = ContextTemplate([
    WriterSection("writer"),
    ReaderSection("reader"),
  ])

  await agent.determine_input_context("user1", "conv1")

  # Verify context was shared
  assert shared_data["read_value"] == 42
  assert shared_data["same_obj"] is True

  await Agent.stop(node, "test-agent-shared")


@pytest.mark.asyncio
async def test_agent_memory_integration():
  """Test that context template correctly integrates with memory."""
  node = Node()

  agent = await Agent.start(
    node=node,
    name="test-agent-memory",
    instructions="You are a test assistant.",
  )

  # Add a message to memory
  scope = "user1"
  conversation = "conv1"
  test_message = {
    "role": "user",
    "content": {"text": "Test message", "type": "text"},
    "phase": "executing"
  }

  await agent.memory.add_message(
    agent._memory_scope(scope),
    conversation,
    test_message
  )

  # Get context
  messages = await agent.determine_input_context(scope, conversation)

  # Convert messages to dicts for easier inspection
  message_dicts = []
  for msg in messages:
    if hasattr(msg, '__dict__'):
      message_dicts.append(msg.__dict__)
    else:
      message_dicts.append(msg)

  # Should include both system instructions and the test message
  assert len(messages) >= 2  # At least instructions + test message

  await Agent.stop(node, "test-agent-memory")

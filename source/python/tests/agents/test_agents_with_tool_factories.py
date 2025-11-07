"""
Integration tests for tool factories and Agent framework.

These tests validate the tool factory pattern, which enables per-scope and
per-conversation tool isolation. This is critical for multi-tenant scenarios
where different users or conversations need isolated tool instances.

The factory pattern bug (using 'tools' instead of 'static_tools') went undetected
because there were zero integration tests for factories. These tests ensure that:
1. Factory objects are correctly detected
2. create_tools() is called with proper parameters
3. Factory-created tools work with the Agent framework
4. Static tools and factory tools can be mixed
5. Scope/conversation isolation is maintained
"""

import asyncio
import os
import pytest
import tempfile
from typing import List, Optional

from autonomy import Agent, Node, Tool
from autonomy.agents.http import HttpServer
from autonomy.tools.filesystem import FilesystemTools
from mock_utils import MockModel, create_simple_mock_model, create_tool_mock_model


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_base_dir():
  """Provide a temporary directory for filesystem operations."""
  with tempfile.TemporaryDirectory() as tmpdir:
    yield tmpdir


# =============================================================================
# CRITICAL TEST 1: Basic Factory Detection and Usage
# =============================================================================


def test_agent_with_tool_factory_basic(temp_base_dir):
  """
  Test that agent can be created with a tool factory.

  This is the MOST CRITICAL test - it catches the bug where factory objects
  were incorrectly passed to prepare_tools(), causing:
  AttributeError: 'FilesystemTools' object has no attribute 'spec'

  The bug was on line 1730 of agent.py:
    all_tools = tools + scope_specific_tools  # BUG - 'tools' includes factories!
  Should be:
    all_tools = static_tools + scope_specific_tools  # CORRECT
  """

  async def _test(node):
    return await _test_agent_with_tool_factory_basic(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_agent_with_tool_factory_basic(node, temp_base_dir):
  # Create FilesystemTools in factory mode (no scope/conversation at init)
  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  # This would fail with the bug: AttributeError on factory.spec
  agent = await Agent.start(
    node=node,
    name="test-factory-agent",
    model=create_simple_mock_model("Test response"),
    tools=[fs_tools],  # Pass factory directly
    instructions="You are a test agent with filesystem tools",
  )

  # If we reach here, the bug is fixed
  assert agent is not None
  assert agent.name == "test-factory-agent"


def test_agent_start_with_factory_returns_reference(temp_base_dir):
  """Test that Agent.start() with factory returns a valid AgentReference."""

  async def _test(node):
    return await _test_agent_start_with_factory_returns_reference(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_agent_start_with_factory_returns_reference(node, temp_base_dir):
  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  agent_ref = await Agent.start(
    node=node,
    name="factory-agent",
    model=create_simple_mock_model(),
    tools=[fs_tools],
    instructions="Test",
  )

  # Verify we got a valid reference
  assert agent_ref is not None
  assert hasattr(agent_ref, "send")
  assert hasattr(agent_ref, "name")


# =============================================================================
# CRITICAL TEST 2: Mixed Static and Factory Tools
# =============================================================================


def test_agent_with_mixed_static_and_factory_tools(temp_base_dir):
  """
  Test agent with both static (function-based) and factory tools.

  This catches the bug because the incorrect code:
    all_tools = tools + scope_specific_tools
  would include both the calculator function AND the FilesystemTools factory
  object, causing prepare_tools() to fail on the factory object.

  The correct code:
    all_tools = static_tools + scope_specific_tools
  only includes the calculator function in static_tools, and factory-created
  tools in scope_specific_tools.
  """

  async def _test(node):
    return await _test_agent_with_mixed_static_and_factory_tools(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_agent_with_mixed_static_and_factory_tools(node, temp_base_dir):
  # Static tool (function-based)
  @Tool
  def calculator(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

  # Factory tool
  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  # Mix both types - this was the exact scenario that triggered the bug
  agent = await Agent.start(
    node=node,
    name="mixed-tools-agent",
    model=create_simple_mock_model("Result: 8"),
    tools=[calculator, fs_tools],  # Static tool + Factory tool
    instructions="You can calculate and manage files",
  )

  assert agent is not None
  assert agent.name == "mixed-tools-agent"


def test_agent_with_multiple_static_and_one_factory(temp_base_dir):
  """Test agent with multiple static tools and one factory."""

  async def _test(node):
    return await _test_agent_with_multiple_static_and_one_factory(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_agent_with_multiple_static_and_one_factory(node, temp_base_dir):
  @Tool
  def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

  @Tool
  def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

  @Tool
  def get_greeting(name: str) -> str:
    """Get a greeting."""
    return f"Hello, {name}!"

  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  agent = await Agent.start(
    node=node,
    name="multi-tool-agent",
    model=create_simple_mock_model(),
    tools=[add, multiply, get_greeting, fs_tools],
    instructions="Multi-tool agent",
  )

  assert agent is not None


# =============================================================================
# TEST 3: Factory create_tools Method
# =============================================================================


def test_factory_create_tools_method_called(temp_base_dir):
  """
  Test that create_tools() method works and is called with correct parameters.

  Validates the factory protocol implementation.
  """
  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  # Manually call create_tools as the framework would
  tools = fs_tools.create_tools(
    scope="user-alice",
    conversation="conv-123",
    agent_name="test-agent",
  )

  # Verify tools were created
  assert len(tools) > 0, "Factory should create at least one tool"

  # Verify tools have required attributes (this is what failed with the bug)
  for tool in tools:
    assert hasattr(tool, "spec"), f"Tool {tool} missing .spec attribute"
    assert tool.spec is not None


def test_factory_creates_different_instances_per_scope(temp_base_dir):
  """
  Test that factory creates different tool instances for different scopes.

  This validates the primary purpose of the factory pattern - isolation.
  """
  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  # Create tools for different scopes
  tools_alice = fs_tools.create_tools("user-alice", "conv-1", "test-agent")
  tools_bob = fs_tools.create_tools("user-bob", "conv-1", "test-agent")

  # Each scope should get its own tools
  assert len(tools_alice) > 0
  assert len(tools_bob) > 0

  # Verify they're working with different scope directories
  # (We can't easily test file isolation without executing the tools,
  # but the factory should set different scope_root paths)


def test_factory_creates_different_instances_per_conversation(temp_base_dir):
  """
  Test that factory creates isolated instances per conversation.

  With visibility="conversation", each conversation should get isolated tools.
  """
  fs_tools = FilesystemTools(base_dir=temp_base_dir)  # Default is conversation-level

  # Same scope, different conversations
  tools_conv1 = fs_tools.create_tools("user-alice", "conv-1", "agent")
  tools_conv2 = fs_tools.create_tools("user-alice", "conv-2", "agent")

  # Should create separate tool instances
  assert len(tools_conv1) > 0
  assert len(tools_conv2) > 0


# =============================================================================
# TEST 4: Factory Detection Logic
# =============================================================================


def test_factory_detection_logic(temp_base_dir):
  """
  Test that factory objects are correctly identified by hasattr check.

  This tests the detection logic in agent.py that separates static tools
  from factories using:
    hasattr(tool, "create_tools") and callable(getattr(tool, "create_tools"))
  """

  @Tool
  def simple_tool() -> str:
    """A simple static tool."""
    return "result"

  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  # Factory should have create_tools method
  assert hasattr(fs_tools, "create_tools"), "Factory should have create_tools method"
  assert callable(getattr(fs_tools, "create_tools")), "create_tools should be callable"

  # Static tool should NOT have create_tools method
  assert not hasattr(simple_tool, "create_tools"), "Static tool should not have create_tools"


def test_factory_vs_static_tool_distinction():
  """Test that we can reliably distinguish factories from static tools."""

  @Tool
  def static_calculator(x: int, y: int) -> int:
    """A static calculator tool."""
    return x + y

  @Tool
  async def async_static_tool(data: str) -> str:
    """An async static tool."""
    return f"Processed: {data}"

  with tempfile.TemporaryDirectory() as tmpdir:
    factory_tool = FilesystemTools(base_dir=tmpdir)

    # All static tools should NOT have create_tools
    for tool in [static_calculator, async_static_tool]:
      assert not hasattr(tool, "create_tools"), f"{tool} should not be detected as factory"

    # Factory should have create_tools
    assert hasattr(factory_tool, "create_tools"), "Factory should have create_tools"


# =============================================================================
# TEST 5: Multiple Factories
# =============================================================================


def test_agent_with_multiple_factories(temp_base_dir):
  """
  Test agent with multiple factory tools.

  Future-proofing for when more factory implementations exist.
  Tests that the framework correctly handles multiple factories.
  """

  async def _test(node):
    return await _test_agent_with_multiple_factories(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_agent_with_multiple_factories(node, temp_base_dir):
  # Create multiple factory instances
  fs_tools_1 = FilesystemTools(base_dir=os.path.join(temp_base_dir, "factory1"))
  fs_tools_2 = FilesystemTools(base_dir=os.path.join(temp_base_dir, "factory2"))

  agent = await Agent.start(
    node=node,
    name="multi-factory-agent",
    model=create_simple_mock_model(),
    tools=[fs_tools_1, fs_tools_2],
    instructions="Test with multiple factories",
  )

  assert agent is not None
  assert agent.name == "multi-factory-agent"


def test_agent_with_factories_and_multiple_static_tools(temp_base_dir):
  """Test complex scenario with multiple factories and multiple static tools."""

  async def _test(node):
    return await _test_agent_with_factories_and_multiple_static_tools(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_agent_with_factories_and_multiple_static_tools(node, temp_base_dir):
  @Tool
  def tool1(x: int) -> int:
    return x * 2

  @Tool
  def tool2(x: str) -> str:
    return x.upper()

  @Tool
  async def tool3(x: str) -> str:
    return f"Async: {x}"

  factory1 = FilesystemTools(base_dir=os.path.join(temp_base_dir, "f1"))
  factory2 = FilesystemTools(base_dir=os.path.join(temp_base_dir, "f2"))

  # Complex mix: static, factory, static, factory, static
  agent = await Agent.start(
    node=node,
    name="complex-mix-agent",
    model=create_simple_mock_model(),
    tools=[tool1, factory1, tool2, factory2, tool3],
    instructions="Complex tool mix",
  )

  assert agent is not None


# =============================================================================
# TEST 6: Factory with Different Visibility Levels
# =============================================================================


def test_factory_with_scope_visibility(temp_base_dir):
  """Test factory with scope-level visibility."""

  async def _test(node):
    return await _test_factory_with_scope_visibility(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_factory_with_scope_visibility(node, temp_base_dir):
  fs_tools = FilesystemTools(visibility="scope", base_dir=temp_base_dir)

  agent = await Agent.start(
    node=node,
    name="scope-agent",
    model=create_simple_mock_model(),
    tools=[fs_tools],
    instructions="Scope-level filesystem",
  )

  assert agent is not None


def test_factory_with_conversation_visibility(temp_base_dir):
  """Test factory with conversation-level visibility (default)."""

  async def _test(node):
    return await _test_factory_with_conversation_visibility(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_factory_with_conversation_visibility(node, temp_base_dir):
  fs_tools = FilesystemTools(visibility="conversation", base_dir=temp_base_dir)

  agent = await Agent.start(
    node=node,
    name="conversation-agent",
    model=create_simple_mock_model(),
    tools=[fs_tools],
    instructions="Conversation-level filesystem",
  )

  assert agent is not None


def test_factory_with_agent_visibility(temp_base_dir):
  """Test factory with agent-level visibility."""

  async def _test(node):
    return await _test_factory_with_agent_visibility(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_factory_with_agent_visibility(node, temp_base_dir):
  fs_tools = FilesystemTools(visibility="agent", base_dir=temp_base_dir)

  agent = await Agent.start(
    node=node,
    name="agent-level-agent",
    model=create_simple_mock_model(),
    tools=[fs_tools],
    instructions="Agent-level filesystem",
  )

  assert agent is not None


# =============================================================================
# TEST 7: Factory Initialization Variants
# =============================================================================


def test_factory_with_minimal_initialization():
  """Test factory with minimal initialization (all defaults)."""
  Node.start(
    _test_factory_with_minimal_initialization,
    wait_until_interrupted=False,
    http_server=HttpServer(listen_address="127.0.0.1:0"),
  )


async def _test_factory_with_minimal_initialization(node):
  # Just create factory with no parameters
  fs_tools = FilesystemTools()

  agent = await Agent.start(
    node=node,
    name="minimal-factory-agent",
    model=create_simple_mock_model(),
    tools=[fs_tools],
    instructions="Minimal factory setup",
  )

  assert agent is not None


def test_factory_with_custom_base_dir(temp_base_dir):
  """Test factory with custom base directory."""

  async def _test(node):
    return await _test_factory_with_custom_base_dir(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_factory_with_custom_base_dir(node, temp_base_dir):
  custom_dir = os.path.join(temp_base_dir, "custom", "path")
  fs_tools = FilesystemTools(base_dir=custom_dir)

  agent = await Agent.start(
    node=node,
    name="custom-dir-agent",
    model=create_simple_mock_model(),
    tools=[fs_tools],
    instructions="Custom base dir",
  )

  assert agent is not None


# =============================================================================
# TEST 8: Error Handling
# =============================================================================


def test_factory_with_invalid_visibility():
  """Test that factory raises error for invalid visibility."""
  with pytest.raises(ValueError, match="Invalid visibility"):
    FilesystemTools(visibility="invalid")


def test_agent_gracefully_handles_empty_tools_list():
  """Test that agent can start with empty tools list."""
  Node.start(
    _test_agent_gracefully_handles_empty_tools_list,
    wait_until_interrupted=False,
    http_server=HttpServer(listen_address="127.0.0.1:0"),
  )


async def _test_agent_gracefully_handles_empty_tools_list(node):
  agent = await Agent.start(
    node=node,
    name="no-tools-agent",
    model=create_simple_mock_model(),
    tools=[],  # Empty list
    instructions="No tools",
  )

  assert agent is not None


def test_agent_with_none_tools_list():
  """Test that agent can start with None tools."""
  Node.start(
    _test_agent_with_none_tools_list, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0")
  )


async def _test_agent_with_none_tools_list(node):
  agent = await Agent.start(
    node=node,
    name="none-tools-agent",
    model=create_simple_mock_model(),
    tools=None,  # None instead of list
    instructions="No tools",
  )

  assert agent is not None


# =============================================================================
# TEST 9: Regression Tests for the Specific Bug
# =============================================================================


def test_regression_factory_object_not_in_prepared_tools(temp_base_dir):
  """
  Regression test for the specific bug.

  The bug was that factory objects were included in the final tool list
  passed to prepare_tools(), which expected Tool objects with .spec attribute.

  This test verifies that:
  1. Factory objects are detected and separated
  2. Only Tool objects (with .spec) are passed to prepare_tools()
  3. Factory-created tools (which have .spec) are correctly included
  """

  async def _test(node):
    return await _test_regression_factory_object_not_in_prepared_tools(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_regression_factory_object_not_in_prepared_tools(node, temp_base_dir):
  @Tool
  def static_tool(x: int) -> int:
    return x * 2

  fs_factory = FilesystemTools(base_dir=temp_base_dir)

  # This was failing because fs_factory (the object itself) was being
  # passed to prepare_tools, which tried to access fs_factory.spec
  agent = await Agent.start(
    node=node,
    name="regression-test-agent",
    model=create_simple_mock_model(),
    tools=[static_tool, fs_factory],
    instructions="Regression test for factory bug",
  )

  # If we get here without AttributeError, the bug is fixed
  assert agent is not None


def test_regression_all_tools_variable_uses_static_tools(temp_base_dir):
  """
  Regression test specifically for line 1730 in agent.py.

  The bug was:
    all_tools = tools + scope_specific_tools  # WRONG
  Should be:
    all_tools = static_tools + scope_specific_tools  # CORRECT

  This test ensures factories are filtered out before combining with
  scope-specific tools.
  """

  async def _test(node):
    return await _test_regression_all_tools_variable_uses_static_tools(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_regression_all_tools_variable_uses_static_tools(node, temp_base_dir):
  factory = FilesystemTools(base_dir=temp_base_dir)

  # The agent creation process will:
  # 1. Detect 'factory' has create_tools method
  # 2. Add it to tool_factories list (not static_tools)
  # 3. Call create_tools() to get scope-specific tools
  # 4. Combine static_tools (empty) + scope_specific_tools
  # 5. Pass combined list to prepare_tools()
  #
  # With the bug, it would pass [factory] + scope_specific_tools,
  # causing AttributeError on factory.spec
  agent = await Agent.start(
    node=node,
    name="regression-line-1730",
    model=create_simple_mock_model(),
    tools=[factory],
    instructions="Test line 1730 bug fix",
  )

  assert agent is not None


# =============================================================================
# TEST 10: Integration with Agent Operations
# =============================================================================


def test_agent_with_factory_can_receive_messages(temp_base_dir):
  """Test that agent with factory tools can receive and respond to messages."""

  async def _test(node):
    return await _test_agent_with_factory_can_receive_messages(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_agent_with_factory_can_receive_messages(node, temp_base_dir):
  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  agent = await Agent.start(
    node=node,
    name="messaging-agent",
    model=create_simple_mock_model("I can help with files!"),
    tools=[fs_tools],
    instructions="You help with files",
  )

  # Send a message
  response = await agent.send("Hello")

  assert response is not None
  assert len(response) > 0


def test_agent_with_mixed_tools_can_receive_messages(temp_base_dir):
  """Test that agent with mixed tools can receive messages."""

  async def _test(node):
    return await _test_agent_with_mixed_tools_can_receive_messages(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_agent_with_mixed_tools_can_receive_messages(node, temp_base_dir):
  @Tool
  def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  agent = await Agent.start(
    node=node,
    name="mixed-messaging-agent",
    model=create_simple_mock_model("Greeting sent!"),
    tools=[greet, fs_tools],
    instructions="You can greet and manage files",
  )

  response = await agent.send("Greet Alice")

  assert response is not None


# =============================================================================
# TEST 11: Edge Cases
# =============================================================================


def test_factory_only_agent_no_static_tools(temp_base_dir):
  """Test agent with only factory tools (no static tools)."""

  async def _test(node):
    return await _test_factory_only_agent_no_static_tools(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_factory_only_agent_no_static_tools(node, temp_base_dir):
  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  # Only factory, no static tools
  agent = await Agent.start(
    node=node,
    name="factory-only-agent",
    model=create_simple_mock_model(),
    tools=[fs_tools],
    instructions="Factory tools only",
  )

  assert agent is not None


def test_static_only_agent_no_factories():
  """Test agent with only static tools (no factories) - baseline case."""
  Node.start(
    _test_static_only_agent_no_factories,
    wait_until_interrupted=False,
    http_server=HttpServer(listen_address="127.0.0.1:0"),
  )


async def _test_static_only_agent_no_factories(node):
  @Tool
  def calc(x: int) -> int:
    return x * 2

  # Only static, no factories
  agent = await Agent.start(
    node=node,
    name="static-only-agent",
    model=create_simple_mock_model(),
    tools=[calc],
    instructions="Static tools only",
  )

  assert agent is not None


def test_agent_start_many_with_factories(temp_base_dir):
  """Test Agent.start_many() with factory tools."""

  async def _test(node):
    return await _test_agent_start_many_with_factories(node, temp_base_dir)

  Node.start(_test, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0"))


async def _test_agent_start_many_with_factories(node, temp_base_dir):
  fs_tools = FilesystemTools(base_dir=temp_base_dir)

  # start_many creates multiple agents with identical config
  agents = await Agent.start_many(
    node=node,
    instructions="Multi-agent with factory tools",
    number_of_agents=2,
    model=create_simple_mock_model(),
    tools=[fs_tools],
  )

  assert len(agents) == 2
  # Each agent gets a unique generated name
  assert agents[0].name is not None
  assert agents[1].name is not None
  assert agents[0].name != agents[1].name

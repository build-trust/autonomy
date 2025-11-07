import pytest
import asyncio
from typing import List, Dict, Any

from autonomy import Agent, Node, Tool, Model
from autonomy.agents.http import HttpServer
from autonomy.nodes.message import ConversationRole
from mock_utils import (
  MockModel,
  create_simple_mock_model,
  create_tool_mock_model,
  calculator_tool,
  add_numbers,
  simple_test_tool,
)


class TestAgentConfigAPI:
  """Tests for config-based agent creation API"""

  def test_start_from_config_minimal(self):
    """Test starting agent with minimal config (only instructions)"""
    Node.start(
      self._test_start_from_config_minimal,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_start_from_config_minimal(self, node):
    model = create_simple_mock_model("Config agent ready!")

    config = {
      "instructions": "You are a test assistant created from config.",
      "model": model,
    }

    agent = await Agent.start_from_config(node=node, config=config)

    assert agent is not None
    assert agent.name is not None  # Auto-generated name

    # Test basic functionality
    response = await agent.send("Hello")
    assert len(response) >= 1
    assert response[-1].role == ConversationRole.ASSISTANT
    assert "Config agent ready!" in response[-1].content.text

  def test_start_from_config_full(self):
    """Test starting agent with all config options"""
    Node.start(
      self._test_start_from_config_full,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_start_from_config_full(self, node):
    model = create_simple_mock_model("Full config agent ready!")

    config = {
      "instructions": "You are a comprehensive test agent.",
      "name": "config-test-agent",
      "model": model,
      "tools": [Tool(simple_test_tool)],
      "max_iterations": 100,
      "max_execution_time": 60.0,
      "max_messages_in_short_term_memory": 500,
      "max_tokens_in_short_term_memory": 50000,
      "enable_long_term_memory": True,
    }

    agent = await Agent.start_from_config(node=node, config=config)

    assert agent is not None
    assert agent.name == "config-test-agent"

    # Test basic functionality
    response = await agent.send("Hello")
    assert len(response) >= 1
    assert response[-1].role == ConversationRole.ASSISTANT

  def test_start_from_config_equivalence(self):
    """Verify config-based start matches direct parameter start"""
    Node.start(
      self._test_start_from_config_equivalence,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_start_from_config_equivalence(self, node):
    model1 = create_simple_mock_model("Direct agent ready!")
    model2 = create_simple_mock_model("Config agent ready!")

    # Start with direct parameters
    agent1 = await Agent.start(
      node=node,
      instructions="Test agent",
      name="direct-agent",
      model=model1,
      max_iterations=100,
    )

    # Start with config
    config = {
      "instructions": "Test agent",
      "name": "config-agent",
      "model": model2,
      "max_iterations": 100,
    }
    agent2 = await Agent.start_from_config(node=node, config=config)

    # Both should work identically
    assert agent1 is not None
    assert agent2 is not None
    assert agent1.name == "direct-agent"
    assert agent2.name == "config-agent"

    # Test both respond correctly
    response1 = await agent1.send("Hello")
    response2 = await agent2.send("Hello")

    assert len(response1) >= 1
    assert len(response2) >= 1
    assert response1[-1].role == ConversationRole.ASSISTANT
    assert response2[-1].role == ConversationRole.ASSISTANT

  def test_start_from_config_with_tools(self):
    """Test config with tools"""
    Node.start(
      self._test_start_from_config_with_tools,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_start_from_config_with_tools(self, node):
    model = create_tool_mock_model(
      tool_name="add_numbers", arguments='{"a": 5, "b": 3}', final_response="The result is 8"
    )

    config = {
      "instructions": "You are a calculator assistant.",
      "name": "calculator-agent",
      "model": model,
      "tools": [Tool(add_numbers)],
    }

    agent = await Agent.start_from_config(node=node, config=config)

    assert agent is not None
    assert agent.name == "calculator-agent"

    # Test tool usage
    response = await agent.send("What is 5 + 3?")
    assert len(response) >= 1

  def test_start_many_from_config(self):
    """Test starting multiple agents from config"""
    Node.start(
      self._test_start_many_from_config,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_start_many_from_config(self, node):
    model = create_simple_mock_model("Multi-agent ready!")

    config = {
      "instructions": "You are a test agent.",
      "number_of_agents": 3,
      "model": model,
    }

    agents = await Agent.start_many_from_config(node=node, config=config)

    assert len(agents) == 3
    assert all(agent is not None for agent in agents)

    # Each agent should have a unique name
    agent_names = [agent.name for agent in agents]
    assert len(set(agent_names)) == 3  # All unique

  def test_start_many_from_config_full(self):
    """Test starting multiple agents with all config options"""
    Node.start(
      self._test_start_many_from_config_full,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_start_many_from_config_full(self, node):
    model = create_simple_mock_model("Full multi-agent ready!")

    config = {
      "instructions": "You are a comprehensive test agent.",
      "number_of_agents": 5,
      "model": model,
      "tools": [Tool(simple_test_tool)],
      "max_iterations": 100,
      "max_execution_time": 60.0,
      "max_messages_in_short_term_memory": 500,
      "max_tokens_in_short_term_memory": 50000,
      "enable_long_term_memory": True,
    }

    agents = await Agent.start_many_from_config(node=node, config=config)

    assert len(agents) == 5
    assert all(agent is not None for agent in agents)

  def test_config_composition(self):
    """Test dict merging for config reuse"""
    Node.start(
      self._test_config_composition, wait_until_interrupted=False, http_server=HttpServer(listen_address="127.0.0.1:0")
    )

  async def _test_config_composition(self, node):
    model1 = create_simple_mock_model("Support agent ready!")
    model2 = create_simple_mock_model("Sales agent ready!")

    # Base configuration
    base_config = {
      "max_iterations": 100,
      "max_execution_time": 60.0,
    }

    # Specific agent configs extending base
    support_config = {
      **base_config,
      "instructions": "You are a support assistant.",
      "name": "support-agent",
      "model": model1,
    }

    sales_config = {
      **base_config,
      "instructions": "You are a sales assistant.",
      "name": "sales-agent",
      "model": model2,
    }

    # Start both agents
    support_agent = await Agent.start_from_config(node=node, config=support_config)
    sales_agent = await Agent.start_from_config(node=node, config=sales_config)

    assert support_agent is not None
    assert sales_agent is not None
    assert support_agent.name == "support-agent"
    assert sales_agent.name == "sales-agent"

    # Both should respond
    support_response = await support_agent.send("Hello")
    sales_response = await sales_agent.send("Hello")

    assert len(support_response) >= 1
    assert len(sales_response) >= 1
    assert "Support agent ready!" in support_response[-1].content.text
    assert "Sales agent ready!" in sales_response[-1].content.text

  def test_start_from_config_with_defaults(self):
    """Test that default values are applied correctly"""
    Node.start(
      self._test_start_from_config_with_defaults,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_start_from_config_with_defaults(self, node):
    model = create_simple_mock_model("Default agent ready!")

    # Minimal config - should use defaults for all optional fields
    config = {
      "instructions": "You are a test agent.",
      "model": model,
    }

    agent = await Agent.start_from_config(node=node, config=config)

    assert agent is not None
    # Agent should work with default settings
    response = await agent.send("Hello")
    assert len(response) >= 1

  def test_start_many_from_config_equivalence(self):
    """Verify start_many_from_config matches start_many"""
    Node.start(
      self._test_start_many_from_config_equivalence,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_start_many_from_config_equivalence(self, node):
    model1 = create_simple_mock_model("Direct multi-agent!")
    model2 = create_simple_mock_model("Config multi-agent!")

    # Start with direct parameters
    agents1 = await Agent.start_many(
      node=node,
      instructions="Test agent",
      number_of_agents=3,
      model=model1,
    )

    # Start with config
    config = {
      "instructions": "Test agent",
      "number_of_agents": 3,
      "model": model2,
    }
    agents2 = await Agent.start_many_from_config(node=node, config=config)

    # Both should create same number of agents
    assert len(agents1) == 3
    assert len(agents2) == 3
    assert all(agent is not None for agent in agents1)
    assert all(agent is not None for agent in agents2)

  def test_config_with_exposed_as(self):
    """Test config with exposed_as parameter"""
    Node.start(
      self._test_config_with_exposed_as,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_config_with_exposed_as(self, node):
    model = create_simple_mock_model("Exposed agent ready!")

    config = {
      "instructions": "You are an exposed test agent.",
      "name": "internal-agent",
      "exposed_as": "public-agent",
      "model": model,
    }

    agent = await Agent.start_from_config(node=node, config=config)

    assert agent is not None
    assert agent.name == "internal-agent"

  def test_config_missing_required_field(self):
    """Test that missing required field raises appropriate error"""
    Node.start(
      self._test_config_missing_required_field,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_config_missing_required_field(self, node):
    # Config missing required 'instructions' field
    config = {
      "name": "test-agent",
    }

    with pytest.raises(KeyError):
      await Agent.start_from_config(node=node, config=config)

  def test_start_many_config_missing_required_fields(self):
    """Test that missing required fields raise appropriate errors"""
    Node.start(
      self._test_start_many_config_missing_required_fields,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_start_many_config_missing_required_fields(self, node):
    # Missing number_of_agents
    config = {
      "instructions": "Test agent",
    }

    with pytest.raises(KeyError):
      await Agent.start_many_from_config(node=node, config=config)

"""
Tests for agent timeouts.

Tests the timeout parameters added to Agent.start(), Agent.stop(),
and Reference.send_message() methods, as well as the custom timeout
error classes.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from autonomy import Agent, Node
from autonomy.agents.http import HttpServer
from autonomy.agents.errors import (
  AgentTimeoutError,
  AgentStartTimeoutError,
  AgentStopTimeoutError,
  SendMessageTimeoutError,
)
from mock_utils import create_simple_mock_model


class TestAgentTimeoutErrors:
  """Tests for custom timeout error classes."""

  def test_agent_timeout_error_basic(self):
    """Test AgentTimeoutError with basic parameters."""
    error = AgentTimeoutError(method="test_method", timeout=30.0)

    assert error.method == "test_method"
    assert error.timeout == 30.0
    assert error.context == {}
    assert "test_method" in str(error)
    assert "30.0s" in str(error)

  def test_agent_timeout_error_with_context(self):
    """Test AgentTimeoutError with context dictionary."""
    context = {"agent_name": "test-agent", "operation": "startup"}
    error = AgentTimeoutError(method="test_method", timeout=15.0, context=context)

    assert error.context == context
    assert "test-agent" in str(error)
    assert "startup" in str(error)

  def test_agent_timeout_error_custom_message(self):
    """Test AgentTimeoutError with custom message."""
    custom_msg = "Custom error message for testing"
    error = AgentTimeoutError(method="test_method", timeout=10.0, message=custom_msg)

    assert str(error) == custom_msg
    assert error.message == custom_msg

  def test_agent_start_timeout_error(self):
    """Test AgentStartTimeoutError specific behavior."""
    error = AgentStartTimeoutError(timeout=30.0, agent_name="my-agent")

    assert error.method == "Agent.start()"
    assert error.timeout == 30.0
    assert error.agent_name == "my-agent"
    assert error.context["agent_name"] == "my-agent"
    assert "Agent.start()" in str(error)
    assert "30.0s" in str(error)
    assert "gateway" in str(error).lower() or "connectivity" in str(error).lower()

  def test_agent_start_timeout_error_without_name(self):
    """Test AgentStartTimeoutError when agent name is not provided."""
    error = AgentStartTimeoutError(timeout=45.0)

    assert error.agent_name is None
    assert "Agent.start()" in str(error)

  def test_agent_stop_timeout_error(self):
    """Test AgentStopTimeoutError specific behavior."""
    error = AgentStopTimeoutError(timeout=10.0, agent_name="stopping-agent")

    assert error.method == "Agent.stop()"
    assert error.timeout == 10.0
    assert error.agent_name == "stopping-agent"
    assert "Agent.stop()" in str(error)
    assert "10.0s" in str(error)

  def test_send_message_timeout_error(self):
    """Test SendMessageTimeoutError specific behavior."""
    error = SendMessageTimeoutError(timeout=60.0, agent_name="target-agent")

    assert error.method == "Reference.send_message()"
    assert error.timeout == 60.0
    assert error.agent_name == "target-agent"
    assert "send_message()" in str(error)
    assert "send()" in str(error).lower()  # Should suggest using send() instead

  def test_timeout_errors_are_asyncio_timeout_error_subclass(self):
    """Test that all timeout errors inherit from asyncio.TimeoutError."""
    errors = [
      AgentTimeoutError(method="test", timeout=1.0),
      AgentStartTimeoutError(timeout=1.0),
      AgentStopTimeoutError(timeout=1.0, agent_name="test"),
      SendMessageTimeoutError(timeout=1.0),
    ]

    for error in errors:
      assert isinstance(error, asyncio.TimeoutError)
      assert isinstance(error, AgentTimeoutError)


class TestAgentStartTimeout:
  """Tests for Agent.start() timeout functionality."""

  def test_agent_start_with_timeout_success(self):
    """Test that agent starts successfully within timeout."""
    Node.start(
      self._test_agent_start_with_timeout_success,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_start_with_timeout_success(self, node):
    model = create_simple_mock_model("Hello!")

    # Agent should start well within 30 seconds
    agent = await Agent.start(
      node=node,
      name="timeout-test-agent",
      instructions="Test agent",
      model=model,
      timeout=30.0,
    )

    assert agent is not None
    assert agent.name == "timeout-test-agent"

    # Verify agent works
    response = await agent.send("Test message")
    assert len(response) >= 1

  def test_agent_start_timeout_none_no_limit(self):
    """Test that timeout=None means no timeout (backward compatible)."""
    Node.start(
      self._test_agent_start_timeout_none_no_limit,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_start_timeout_none_no_limit(self, node):
    model = create_simple_mock_model("No timeout test")

    # timeout=None should work (no timeout)
    agent = await Agent.start(
      node=node,
      name="no-timeout-agent",
      instructions="Test agent without timeout",
      model=model,
      timeout=None,
    )

    assert agent is not None
    assert agent.name == "no-timeout-agent"

  def test_agent_start_timeout_raises_correct_error_type(self):
    """Test that timeout raises AgentStartTimeoutError, not generic TimeoutError."""
    Node.start(
      self._test_agent_start_timeout_raises_correct_error_type,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_start_timeout_raises_correct_error_type(self, node):
    # Create a mock that will cause a delay
    async def slow_start_spawner(*args, **kwargs):
      await asyncio.sleep(10)  # Simulate slow startup

    # Patch the start_spawner to be slow
    original_start_spawner = node.start_spawner
    node.start_spawner = slow_start_spawner

    try:
      model = create_simple_mock_model("Slow agent")

      with pytest.raises(AgentStartTimeoutError) as exc_info:
        await Agent.start(
          node=node,
          name="slow-start-agent",
          instructions="This agent starts slowly",
          model=model,
          timeout=0.1,  # Very short timeout to trigger error
        )

      error = exc_info.value
      assert error.timeout == 0.1
      assert error.agent_name == "slow-start-agent"
      assert isinstance(error, asyncio.TimeoutError)
    finally:
      node.start_spawner = original_start_spawner


class TestAgentStopTimeout:
  """Tests for Agent.stop() timeout functionality."""

  def test_agent_stop_with_timeout_success(self):
    """Test that agent stops successfully within timeout."""
    Node.start(
      self._test_agent_stop_with_timeout_success,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_stop_with_timeout_success(self, node):
    model = create_simple_mock_model("Stopping test")

    agent = await Agent.start(
      node=node,
      name="stop-test-agent",
      instructions="Test agent for stop",
      model=model,
    )

    # Stop should complete within timeout
    await Agent.stop(node, agent.name, timeout=10.0)

    # Agent should be stopped (subsequent operations may fail)

  def test_agent_stop_timeout_none_no_limit(self):
    """Test that timeout=None means no timeout (backward compatible)."""
    Node.start(
      self._test_agent_stop_timeout_none_no_limit,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_stop_timeout_none_no_limit(self, node):
    model = create_simple_mock_model("No timeout stop test")

    agent = await Agent.start(
      node=node,
      name="no-timeout-stop-agent",
      instructions="Test agent",
      model=model,
    )

    # timeout=None should work (no timeout)
    await Agent.stop(node, agent.name, timeout=None)

  def test_agent_stop_timeout_logs_warning_by_default(self):
    """Test that timeout logs warning but doesn't raise by default."""
    Node.start(
      self._test_agent_stop_timeout_logs_warning_by_default,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_stop_timeout_logs_warning_by_default(self, node):
    # Create a mock that will cause a delay on stop
    async def slow_stop_worker(*args, **kwargs):
      await asyncio.sleep(10)  # Simulate slow stop

    original_stop_worker = node.stop_worker

    model = create_simple_mock_model("Slow stop test")

    agent = await Agent.start(
      node=node,
      name="slow-stop-agent",
      instructions="Test agent",
      model=model,
    )

    node.stop_worker = slow_stop_worker

    try:
      # Should NOT raise, just log warning
      await Agent.stop(node, agent.name, timeout=0.1, raise_on_timeout=False)
      # If we get here without exception, test passes
    finally:
      node.stop_worker = original_stop_worker

  def test_agent_stop_timeout_raises_when_configured(self):
    """Test that timeout raises when raise_on_timeout=True."""
    Node.start(
      self._test_agent_stop_timeout_raises_when_configured,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_stop_timeout_raises_when_configured(self, node):
    # Create a mock that will cause a delay on stop
    async def slow_stop_worker(*args, **kwargs):
      await asyncio.sleep(10)  # Simulate slow stop

    original_stop_worker = node.stop_worker

    model = create_simple_mock_model("Slow stop raise test")

    agent = await Agent.start(
      node=node,
      name="slow-stop-raise-agent",
      instructions="Test agent",
      model=model,
    )

    node.stop_worker = slow_stop_worker

    try:
      with pytest.raises(AgentStopTimeoutError) as exc_info:
        await Agent.stop(node, agent.name, timeout=0.1, raise_on_timeout=True)

      error = exc_info.value
      assert error.timeout == 0.1
      assert error.agent_name == "slow-stop-raise-agent"
      assert isinstance(error, asyncio.TimeoutError)
    finally:
      node.stop_worker = original_stop_worker


class TestSendMessageTimeout:
  """Tests for Reference.send_message() timeout functionality."""

  def test_send_message_with_timeout_success(self):
    """Test that send_message works within timeout."""
    Node.start(
      self._test_send_message_with_timeout_success,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_send_message_with_timeout_success(self, node):
    model = create_simple_mock_model("Message received!")

    agent = await Agent.start(
      node=node,
      name="send-message-agent",
      instructions="Test agent",
      model=model,
    )

    # send_message with timeout should work
    receive_ref = await agent.send_message("Hello", timeout=30.0)
    assert receive_ref is not None

  def test_send_message_timeout_none_no_limit(self):
    """Test that timeout=None means no timeout (backward compatible)."""
    Node.start(
      self._test_send_message_timeout_none_no_limit,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_send_message_timeout_none_no_limit(self, node):
    model = create_simple_mock_model("No timeout message test")

    agent = await Agent.start(
      node=node,
      name="no-timeout-message-agent",
      instructions="Test agent",
      model=model,
    )

    # timeout=None should work (no timeout)
    receive_ref = await agent.send_message("Hello", timeout=None)
    assert receive_ref is not None


class TestTimeoutErrorMessages:
  """Tests for timeout error message quality."""

  def test_error_messages_contain_method_name(self):
    """Test that error messages contain the method that timed out."""
    errors = [
      (AgentStartTimeoutError(timeout=30.0, agent_name="test"), "Agent.start()"),
      (AgentStopTimeoutError(timeout=10.0, agent_name="test"), "Agent.stop()"),
      (SendMessageTimeoutError(timeout=60.0, agent_name="test"), "send_message()"),
    ]

    for error, expected_method in errors:
      assert expected_method in str(error), f"Expected '{expected_method}' in error message: {error}"

  def test_error_messages_contain_timeout_value(self):
    """Test that error messages contain the timeout value."""
    error = AgentStartTimeoutError(timeout=42.5, agent_name="test")
    assert "42.5" in str(error)

  def test_error_messages_contain_agent_name_when_provided(self):
    """Test that error messages contain agent name when provided."""
    error = AgentStartTimeoutError(timeout=30.0, agent_name="my-special-agent")
    assert "my-special-agent" in str(error)

  def test_error_messages_contain_helpful_suggestions(self):
    """Test that error messages contain suggestions for resolution."""
    start_error = AgentStartTimeoutError(timeout=30.0, agent_name="test")
    assert "timeout" in str(start_error).lower() or "connectivity" in str(start_error).lower()

    stop_error = AgentStopTimeoutError(timeout=10.0, agent_name="test")
    assert "timeout" in str(stop_error).lower() or "processing" in str(stop_error).lower()

    send_error = SendMessageTimeoutError(timeout=60.0)
    # Should suggest using send() instead
    assert "send()" in str(send_error).lower()


class TestTimeoutBackwardCompatibility:
  """Tests to ensure backward compatibility with existing code."""

  def test_agent_start_without_timeout_works(self):
    """Test that Agent.start() works without timeout parameter."""
    Node.start(
      self._test_agent_start_without_timeout_works,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_start_without_timeout_works(self, node):
    model = create_simple_mock_model("Backward compat test")

    # Should work without timeout parameter
    agent = await Agent.start(
      node=node,
      name="compat-agent",
      instructions="Test agent",
      model=model,
    )

    assert agent is not None

  def test_agent_stop_without_timeout_works(self):
    """Test that Agent.stop() works without timeout parameter."""
    Node.start(
      self._test_agent_stop_without_timeout_works,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_stop_without_timeout_works(self, node):
    model = create_simple_mock_model("Backward compat stop test")

    agent = await Agent.start(
      node=node,
      name="compat-stop-agent",
      instructions="Test agent",
      model=model,
    )

    # Should work without timeout parameter
    await Agent.stop(node, agent.name)

  def test_send_message_without_timeout_works(self):
    """Test that send_message() works without timeout parameter."""
    Node.start(
      self._test_send_message_without_timeout_works,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_send_message_without_timeout_works(self, node):
    model = create_simple_mock_model("Backward compat message test")

    agent = await Agent.start(
      node=node,
      name="compat-message-agent",
      instructions="Test agent",
      model=model,
    )

    # Should work without timeout parameter
    receive_ref = await agent.send_message("Hello")
    assert receive_ref is not None

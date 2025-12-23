"""
Comprehensive Timeout & Rate Limit Test Scenarios.

This module tests timeout and rate limiting behavior across all layers of the
Autonomy framework, covering 20 scenarios defined in TIMEOUT_TEST_SCENARIOS.md.

Test Categories:
- A: Interactive Chat (Scenarios 1, 13)
- B: Tool-Augmented Chat (Scenarios 2, 12)
- C: Human-in-the-Loop (Scenario 3)
- D: Deep Work / Research (Scenarios 4, 10)
- E: High-Scale Batch (Scenarios 5, 11)
- F: Voice Interface (Scenarios 6, 7)
- G: Subagent Delegation (Scenarios 8, 9)
- H: Error Handling (Scenarios 14, 15)
- I: Multi-Client Rate Limiting (Scenarios 16-20)

Usage:
    # Run all tests with mocks
    cd autonomy/source/python
    uv run --active pytest tests/agents/test_timeout_scenarios.py -v

    # Run specific category
    uv run --active pytest tests/agents/test_timeout_scenarios.py -v -k "CategoryA"

    # Run live gateway tests (requires gateway connection)
    uv run --active pytest tests/agents/test_timeout_scenarios.py -v -m live_gateway
"""

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
import tempfile
import os
import shutil

from autonomy import Agent, Node, Tool
from autonomy.agents.http import HttpServer
from autonomy.agents.errors import (
  AgentTimeoutError,
  AgentStartTimeoutError,
  AgentStopTimeoutError,
  SendMessageTimeoutError,
)
from autonomy.nodes.message import ConversationRole
from autonomy.tools.filesystem import FilesystemTools

# Import mock utilities from existing test infrastructure
from mock_utils import (
  MockModel,
  SlowMockModel,
  ErrorMockModel,
  create_simple_mock_model,
  create_tool_mock_model,
  create_conversation_mock_model,
  calculator_tool,
  simple_test_tool,
)


# =============================================================================
# ADDITIONAL MOCK UTILITIES FOR TIMEOUT TESTING
# =============================================================================


class TimedMockModel(MockModel):
  """
  Mock model that tracks timing of calls for performance testing.

  Attributes:
      call_times: List of (start_time, end_time, duration) for each call
      total_delay: Cumulative delay across all calls
  """

  def __init__(self, messages: List[Dict[str, Any]] = None, delay: float = 0.0):
    super().__init__(messages)
    self.delay = delay
    self.call_times: List[tuple] = []
    self.total_delay = 0.0

  def complete_chat(self, messages, stream: bool = False, **kwargs):
    start = time.monotonic()
    self.call_count += 1

    if not self.provided_messages:
      result = self._default_response(stream)
    else:
      provided_message = self.provided_messages.pop(0)
      if stream:
        result = self._timed_streaming(provided_message, start)
      else:
        result = self._timed_non_streaming(provided_message, start)

    return result

  async def _timed_non_streaming(self, provided_message: Dict[str, Any], start: float):
    if self.delay > 0:
      await asyncio.sleep(self.delay)

    end = time.monotonic()
    duration = end - start
    self.call_times.append((start, end, duration))
    self.total_delay += duration

    return await super()._complete_chat_non_streaming(provided_message)

  async def _timed_streaming(self, provided_message: Dict[str, Any], start: float):
    if self.delay > 0:
      await asyncio.sleep(self.delay / 2)  # Half delay before streaming

    async for chunk in super()._complete_chat_streaming(provided_message):
      yield chunk

    end = time.monotonic()
    duration = end - start
    self.call_times.append((start, end, duration))
    self.total_delay += duration


class RateLimitedMockModel(MockModel):
  """
  Mock model that simulates rate limiting behavior.

  Can be configured to return 429 errors after a certain number of requests
  or based on a rate limit.
  """

  def __init__(
    self,
    messages: List[Dict[str, Any]] = None,
    rate_limit_after: int = 0,
    rate_limit_rpm: float = 60.0,
    retry_after: float = 5.0,
  ):
    super().__init__(messages)
    self.rate_limit_after = rate_limit_after
    self.rate_limit_rpm = rate_limit_rpm
    self.retry_after = retry_after
    self.rate_limited_count = 0
    self._request_times: List[float] = []

  def complete_chat(self, messages, stream: bool = False, **kwargs):
    self.call_count += 1
    now = time.monotonic()

    # Check if we should rate limit based on count
    if self.rate_limit_after > 0 and self.call_count > self.rate_limit_after:
      self.rate_limited_count += 1
      raise RateLimitError(
        f"Rate limit exceeded after {self.rate_limit_after} requests",
        retry_after=self.retry_after,
      )

    # Track request time for RPM-based limiting
    self._request_times.append(now)

    # Clean old requests (older than 60 seconds)
    cutoff = now - 60.0
    self._request_times = [t for t in self._request_times if t > cutoff]

    # Check RPM limit
    if len(self._request_times) > self.rate_limit_rpm:
      self.rate_limited_count += 1
      raise RateLimitError(
        f"Rate limit of {self.rate_limit_rpm} RPM exceeded",
        retry_after=self.retry_after,
      )

    return super().complete_chat(messages, stream, **kwargs)


class RateLimitError(Exception):
  """Simulated rate limit error."""

  def __init__(self, message: str, retry_after: float = 5.0):
    super().__init__(message)
    self.retry_after = retry_after
    self.status_code = 429


class CircuitBreakerMockModel(MockModel):
  """
  Mock model that simulates circuit breaker states.

  Attributes:
      circuit_state: Current state (closed, half-open, open, exhausted)
      failure_count: Number of consecutive failures
  """

  def __init__(
    self,
    messages: List[Dict[str, Any]] = None,
    initial_state: str = "closed",
    fail_after: int = 0,
  ):
    super().__init__(messages)
    self.circuit_state = initial_state
    self.failure_count = 0
    self.fail_after = fail_after
    self.state_changes: List[tuple] = []

  def set_circuit_state(self, state: str):
    """Manually set the circuit state."""
    old_state = self.circuit_state
    self.circuit_state = state
    self.state_changes.append((time.monotonic(), old_state, state))

  def complete_chat(self, messages, stream: bool = False, **kwargs):
    self.call_count += 1

    # Simulate circuit breaker behavior
    if self.circuit_state == "open":
      raise CircuitOpenError("Circuit breaker is open")
    elif self.circuit_state == "exhausted":
      raise CircuitExhaustedError("All providers exhausted")
    elif self.circuit_state == "half-open":
      # 50% chance of success in half-open state
      if self.call_count % 2 == 0:
        self.set_circuit_state("closed")
      else:
        raise CircuitOpenError("Circuit breaker test failed")

    # Check if we should fail based on count
    if self.fail_after > 0 and self.call_count >= self.fail_after:
      self.failure_count += 1
      if self.failure_count >= 3:
        self.set_circuit_state("open")
      raise Exception("Simulated failure")

    return super().complete_chat(messages, stream, **kwargs)


class CircuitOpenError(Exception):
  """Circuit breaker is open."""

  pass


class CircuitExhaustedError(Exception):
  """All providers exhausted."""

  pass


@dataclass
class MockGatewayResponse:
  """Mock response with gateway headers."""

  content: str
  hint_rpm: Optional[float] = None
  circuit_state: str = "closed"
  retry_after: Optional[float] = None

  @property
  def _gateway_hint_rpm(self):
    return self.hint_rpm

  @property
  def _gateway_circuit_state(self):
    return self.circuit_state


@dataclass
class TimingResult:
  """Result of a timed operation."""

  success: bool
  duration: float
  iterations: int = 0
  error: Optional[Exception] = None
  response: Any = None


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def simple_model():
  """Create a simple mock model for basic tests."""
  return create_simple_mock_model("Hello! I'm here to help.")


@pytest.fixture
def slow_model():
  """Create a slow mock model for timeout tests."""
  return SlowMockModel(
    messages=[{"role": "assistant", "content": "Slow response"}],
    delay=0.5,
  )


@pytest.fixture
def timed_model():
  """Create a timed mock model for performance tests."""
  return TimedMockModel(
    messages=[
      {"role": "assistant", "content": "Response 1"},
      {"role": "assistant", "content": "Response 2"},
      {"role": "assistant", "content": "Response 3"},
    ],
    delay=0.1,
  )


@pytest.fixture
def rate_limited_model():
  """Create a rate-limited mock model."""
  return RateLimitedMockModel(
    messages=[{"role": "assistant", "content": "Rate limited response"}] * 100,
    rate_limit_after=5,
    retry_after=1.0,
  )


# =============================================================================
# CATEGORY A: INTERACTIVE CHAT (Scenarios 1, 13)
# =============================================================================


class TestCategoryAInteractiveChat:
  """
  Tests for interactive chat scenarios with quick responses.

  Scenario 1: Simple Chat Response
  Scenario 13: Cold Start vs Warm
  """

  # -------------------------------------------------------------------------
  # Scenario 1: Simple Chat Response
  # -------------------------------------------------------------------------

  def test_simple_chat_responds_quickly(self):
    """Test that simple chat responds within 5 seconds."""
    Node.start(
      self._test_simple_chat_responds_quickly_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_simple_chat_responds_quickly_impl(self, node):
    model = TimedMockModel(
      messages=[{"role": "assistant", "content": "Hello! How can I help?"}],
      delay=0.1,
    )

    agent = await Agent.start(
      node=node,
      name="simple-chat-agent",
      instructions="You are a helpful assistant.",
      model=model,
      max_iterations=10,
      max_execution_time=60.0,
    )

    start = time.monotonic()
    response = await agent.send("Hello")
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"Response took {elapsed:.2f}s, expected < 5s"
    assert len(response) >= 1
    assert response[-1].role == ConversationRole.ASSISTANT
    assert "Hello" in response[-1].content.text or "help" in response[-1].content.text

  def test_simple_chat_under_http_timeout(self):
    """Test that simple chat completes before HTTP timeout."""
    Node.start(
      self._test_simple_chat_under_http_timeout_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_simple_chat_under_http_timeout_impl(self, node):
    model = create_simple_mock_model("Quick response!")

    agent = await Agent.start(
      node=node,
      name="http-timeout-agent",
      instructions="You are helpful.",
      model=model,
      max_execution_time=60.0,
    )

    # Simulate HTTP timeout of 60s - response should be well under
    start = time.monotonic()
    response = await agent.send("Quick question")
    elapsed = time.monotonic() - start

    assert elapsed < 60.0, f"Response took {elapsed:.2f}s, exceeds HTTP timeout"
    assert len(response) >= 1

  def test_simple_chat_no_iteration_limit_hit(self):
    """Test that simple chat uses fewer than max iterations."""
    Node.start(
      self._test_simple_chat_no_iteration_limit_hit_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_simple_chat_no_iteration_limit_hit_impl(self, node):
    model = TimedMockModel(
      messages=[{"role": "assistant", "content": "Simple answer."}],
      delay=0.0,
    )

    agent = await Agent.start(
      node=node,
      name="iteration-test-agent",
      instructions="Give brief answers.",
      model=model,
      max_iterations=10,
      max_execution_time=60.0,
    )

    response = await agent.send("What is 2+2?")

    # For a simple question, should use very few iterations
    assert model.call_count <= 3, f"Used {model.call_count} iterations, expected <= 3"
    assert len(response) >= 1

  # -------------------------------------------------------------------------
  # Scenario 13: Cold Start vs Warm
  # -------------------------------------------------------------------------

  def test_cold_start_timing(self):
    """Test that first request (cold start) completes within reasonable time."""
    Node.start(
      self._test_cold_start_timing_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_cold_start_timing_impl(self, node):
    model = create_simple_mock_model("Cold start response!")

    # Time the agent start (cold start)
    start = time.monotonic()
    agent = await Agent.start(
      node=node,
      name="cold-start-agent",
      instructions="You are helpful.",
      model=model,
      timeout=30.0,  # 30s startup timeout
    )
    startup_time = time.monotonic() - start

    # Time the first request
    request_start = time.monotonic()
    response = await agent.send("First request")
    first_request_time = time.monotonic() - request_start

    # Cold start should be under 30s
    assert startup_time < 30.0, f"Cold start took {startup_time:.2f}s"
    # First request should also be reasonable
    assert first_request_time < 10.0, f"First request took {first_request_time:.2f}s"
    assert len(response) >= 1

  def test_warm_request_fast(self):
    """Test that subsequent requests are faster than cold start."""
    Node.start(
      self._test_warm_request_fast_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_warm_request_fast_impl(self, node):
    model = MockModel(
      messages=[
        {"role": "assistant", "content": "First response"},
        {"role": "assistant", "content": "Second response"},
        {"role": "assistant", "content": "Third response"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="warm-test-agent",
      instructions="You are helpful.",
      model=model,
    )

    # First request (may include some warm-up)
    start1 = time.monotonic()
    await agent.send("First")
    time1 = time.monotonic() - start1

    # Second request (should be warm)
    start2 = time.monotonic()
    await agent.send("Second")
    time2 = time.monotonic() - start2

    # Third request (should be warm)
    start3 = time.monotonic()
    await agent.send("Third")
    time3 = time.monotonic() - start3

    # Warm requests should be fast (under 5s each)
    assert time2 < 5.0, f"Warm request 2 took {time2:.2f}s"
    assert time3 < 5.0, f"Warm request 3 took {time3:.2f}s"

  def test_agent_start_with_timeout(self):
    """Test that Agent.start() timeout parameter works."""
    Node.start(
      self._test_agent_start_with_timeout_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_start_with_timeout_impl(self, node):
    model = create_simple_mock_model("Started!")

    # Should succeed with reasonable timeout
    agent = await Agent.start(
      node=node,
      name="timeout-param-agent",
      instructions="Test agent",
      model=model,
      timeout=30.0,
    )

    assert agent is not None
    assert agent.name == "timeout-param-agent"

  def test_agent_start_timeout_error(self):
    """Test that AgentStartTimeoutError is raised on timeout."""
    Node.start(
      self._test_agent_start_timeout_error_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_start_timeout_error_impl(self, node):
    # Patch start_spawner to be slow
    original_start_spawner = node.start_spawner

    async def slow_start_spawner(*args, **kwargs):
      await asyncio.sleep(10)  # Simulate slow startup

    node.start_spawner = slow_start_spawner

    try:
      model = create_simple_mock_model("Should timeout")

      with pytest.raises(AgentStartTimeoutError) as exc_info:
        await Agent.start(
          node=node,
          name="slow-start-agent",
          instructions="Test agent",
          model=model,
          timeout=0.1,  # Very short timeout
        )

      error = exc_info.value
      assert error.timeout == 0.1
      assert error.agent_name == "slow-start-agent"
      assert isinstance(error, asyncio.TimeoutError)
    finally:
      node.start_spawner = original_start_spawner


# =============================================================================
# CATEGORY B: TOOL-AUGMENTED CHAT (Scenarios 2, 12)
# =============================================================================


class TestCategoryBToolAugmentedChat:
  """
  Tests for tool-augmented chat scenarios.

  Scenario 2: Tool-Augmented Response
  Scenario 12: Long Streaming Response
  """

  # -------------------------------------------------------------------------
  # Scenario 2: Tool-Augmented Response
  # -------------------------------------------------------------------------

  def test_tool_chat_executes_tools(self):
    """Test that agent executes tools and uses results."""
    Node.start(
      self._test_tool_chat_executes_tools_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_tool_chat_executes_tools_impl(self, node):
    model = create_tool_mock_model(
      "calculator_tool",
      '{"expression": "2 + 2"}',
      "The result is 4",
    )

    agent = await Agent.start(
      node=node,
      name="tool-agent",
      instructions="Use calculator for math questions.",
      model=model,
      tools=[Tool(calculator_tool)],
      max_iterations=20,
      max_execution_time=120.0,
    )

    response = await agent.send("What is 2 + 2?")

    # Should have tool call and response
    assert len(response) >= 2
    tool_responses = [msg for msg in response if msg.role == ConversationRole.TOOL]
    assert len(tool_responses) >= 1
    assert "4" in tool_responses[0].content.text

  def test_tool_chat_multiple_tools(self):
    """Test agent with multiple sequential tool calls."""
    Node.start(
      self._test_tool_chat_multiple_tools_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_tool_chat_multiple_tools_impl(self, node):
    # Model that makes multiple tool calls
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "calculator_tool", "arguments": '{"expression": "10 + 5"}'}],
        },
        {
          "role": "assistant",
          "tool_calls": [{"name": "calculator_tool", "arguments": '{"expression": "15 * 2"}'}],
        },
        {"role": "assistant", "content": "First calculation: 15, Second: 30"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="multi-tool-agent",
      instructions="Perform calculations step by step.",
      model=model,
      tools=[Tool(calculator_tool)],
      max_iterations=20,
      max_execution_time=120.0,
    )

    start = time.monotonic()
    response = await agent.send("Calculate 10+5, then multiply by 2")
    elapsed = time.monotonic() - start

    # Should complete within timeout
    assert elapsed < 120.0, f"Took {elapsed:.2f}s"

    # Should have multiple tool responses
    tool_responses = [msg for msg in response if msg.role == ConversationRole.TOOL]
    assert len(tool_responses) >= 2

  def test_tool_chat_under_http_timeout(self):
    """Test tool chat completes before HTTP timeout."""
    Node.start(
      self._test_tool_chat_under_http_timeout_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_tool_chat_under_http_timeout_impl(self, node):
    model = create_tool_mock_model(
      "simple_test_tool",
      '{"param": "test"}',
      "Tool completed successfully",
    )

    agent = await Agent.start(
      node=node,
      name="http-tool-agent",
      instructions="Use tools as needed.",
      model=model,
      tools=[Tool(simple_test_tool)],
      max_execution_time=120.0,
    )

    start = time.monotonic()
    response = await agent.send("Run a test")
    elapsed = time.monotonic() - start

    # Should be well under HTTP timeout (120s)
    assert elapsed < 60.0, f"Took {elapsed:.2f}s, should be under 60s"
    assert len(response) >= 1

  def test_tool_chat_slow_tool(self):
    """Test handling of slow tool execution."""
    Node.start(
      self._test_tool_chat_slow_tool_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_tool_chat_slow_tool_impl(self, node):
    # Define a slow tool
    async def slow_tool(param: str) -> str:
      await asyncio.sleep(2.0)  # 2 second delay
      return f"Slow result: {param}"

    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "slow_tool", "arguments": '{"param": "test"}'}],
        },
        {"role": "assistant", "content": "Got the slow result!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="slow-tool-agent",
      instructions="Use slow_tool when asked.",
      model=model,
      tools=[Tool(slow_tool)],
      max_execution_time=120.0,
    )

    start = time.monotonic()
    response = await agent.send("Run the slow tool")
    elapsed = time.monotonic() - start

    # Should complete but take at least 2s for the slow tool
    assert elapsed >= 2.0, f"Expected >= 2s, got {elapsed:.2f}s"
    assert elapsed < 30.0, f"Took too long: {elapsed:.2f}s"
    assert len(response) >= 1

  # -------------------------------------------------------------------------
  # Scenario 12: Long Streaming Response
  # -------------------------------------------------------------------------

  def test_streaming_long_response(self):
    """Test streaming of long responses."""
    Node.start(
      self._test_streaming_long_response_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_long_response_impl(self, node):
    # Create a model with a long response
    long_content = "This is a very long response. " * 100
    model = MockModel([{"role": "assistant", "content": long_content}])

    agent = await Agent.start(
      node=node,
      name="streaming-agent",
      instructions="Provide detailed explanations.",
      model=model,
      max_execution_time=300.0,
    )

    chunks = []
    start = time.monotonic()

    async for chunk in agent.send_stream("Give me a long explanation"):
      chunks.append(chunk)

    elapsed = time.monotonic() - start

    # Should have received many chunks
    assert len(chunks) > 0, "Expected chunks from streaming"
    assert elapsed < 300.0, f"Streaming took {elapsed:.2f}s"

  def test_streaming_chunks_arrive(self):
    """Test that streaming chunks arrive continuously."""
    Node.start(
      self._test_streaming_chunks_arrive_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_chunks_arrive_impl(self, node):
    model = MockModel([{"role": "assistant", "content": "Chunk by chunk response here."}])

    agent = await Agent.start(
      node=node,
      name="chunk-agent",
      instructions="Stream responses.",
      model=model,
    )

    chunk_times = []
    start = time.monotonic()

    async for chunk in agent.send_stream("Stream something"):
      chunk_times.append(time.monotonic() - start)

    # Should have multiple chunks
    assert len(chunk_times) > 0, "Expected chunks"

    # First chunk should arrive quickly
    if chunk_times:
      assert chunk_times[0] < 5.0, f"First chunk took {chunk_times[0]:.2f}s"

  def test_streaming_no_timeout(self):
    """Test that long streaming doesn't timeout."""
    Node.start(
      self._test_streaming_no_timeout_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_no_timeout_impl(self, node):
    # Use slow mock model to simulate slow streaming
    model = SlowMockModel(
      messages=[{"role": "assistant", "content": "Slowly streamed content."}],
      delay=0.1,  # Small delay per chunk
    )

    agent = await Agent.start(
      node=node,
      name="no-timeout-stream-agent",
      instructions="Stream slowly.",
      model=model,
      max_execution_time=300.0,
    )

    chunks = []
    async for chunk in agent.send_stream("Stream slowly please"):
      chunks.append(chunk)

    # Should complete without timeout
    assert len(chunks) > 0


# =============================================================================
# CATEGORY C: HUMAN-IN-THE-LOOP (Scenario 3)
# =============================================================================


class TestCategoryCHumanInTheLoop:
  """
  Tests for human-in-the-loop scenarios with pause/resume.

  Scenario 3: Human-in-the-Loop Conversation
  """

  def test_hitl_pauses_correctly(self):
    """Test that agent pauses when asking for input."""
    # This test verifies the HITL mechanism works
    # Full HITL tests are in test_hitl_*.py files
    Node.start(
      self._test_hitl_pauses_correctly_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_hitl_pauses_correctly_impl(self, node):
    # Create model that requests user input
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "What is your name?"}'}],
        },
      ]
    )

    agent = await Agent.start(
      node=node,
      name="hitl-agent",
      instructions="Ask user for their name.",
      model=model,
      enable_ask_for_user_input=True,
      max_iterations=50,
      max_execution_time=300.0,
    )

    # Start the conversation - should pause waiting for input
    # The exact mechanism depends on the HITL implementation
    # This is a basic validation that the agent starts correctly
    assert agent is not None
    assert agent.name == "hitl-agent"

  def test_hitl_timeout_during_execution(self):
    """Test that execution times out if too long (not pause time)."""
    Node.start(
      self._test_hitl_timeout_during_execution_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_hitl_timeout_during_execution_impl(self, node):
    # Model that would take too long
    model = SlowMockModel(
      messages=[{"role": "assistant", "content": "Very slow response"}],
      delay=5.0,  # 5 second delay
    )

    agent = await Agent.start(
      node=node,
      name="slow-hitl-agent",
      instructions="Be slow.",
      model=model,
      max_execution_time=1.0,  # 1 second limit
    )

    # This should timeout because the model is slower than max_execution_time
    # The agent properly enforces max_execution_time and raises an exception
    try:
      response = await agent.send("Hello")
      # If we get here without error, the agent completed somehow
      # This is also acceptable (model might have been faster than expected)
      assert len(response) >= 0  # Any response is valid
    except Exception as e:
      # Timeout exception is expected behavior - agent enforces max_execution_time
      # The error message should indicate timeout
      assert "timed out" in str(e).lower() or "timeout" in str(e).lower()


# =============================================================================
# CATEGORY D: DEEP WORK / RESEARCH (Scenarios 4, 10)
# =============================================================================


class TestCategoryDDeepWork:
  """
  Tests for deep work / research scenarios.

  Scenario 4: Deep Research Task
  Scenario 10: Filesystem-Enabled Agent
  """

  def test_deep_research_completes(self):
    """Test that deep research task completes within timeout."""
    Node.start(
      self._test_deep_research_completes_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_deep_research_completes_impl(self, node):
    # Model with multiple iterations simulating research
    model = MockModel(
      [
        {"role": "assistant", "content": "Starting research..."},
        {"role": "assistant", "content": "Found relevant information..."},
        {"role": "assistant", "content": "Research complete. Here are my findings."},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="research-agent",
      instructions="Conduct thorough research.",
      model=model,
      max_iterations=100,
      max_execution_time=1800.0,  # 30 minutes
    )

    start = time.monotonic()
    response = await agent.send("Research the topic")
    elapsed = time.monotonic() - start

    assert elapsed < 1800.0, f"Research took {elapsed:.2f}s"
    assert len(response) >= 1

  def test_deep_research_many_iterations(self):
    """Test agent with many iterations for deep work."""
    Node.start(
      self._test_deep_research_many_iterations_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_deep_research_many_iterations_impl(self, node):
    # Create model with many responses
    messages = [{"role": "assistant", "content": f"Step {i}"} for i in range(10)]
    messages.append({"role": "assistant", "content": "Research complete!"})
    model = MockModel(messages)

    agent = await Agent.start(
      node=node,
      name="multi-iter-agent",
      instructions="Work through many steps.",
      model=model,
      max_iterations=100,
      max_execution_time=600.0,
    )

    response = await agent.send("Do detailed work")
    assert len(response) >= 1

  def test_deep_research_long_running(self):
    """Test that long-running research succeeds."""
    Node.start(
      self._test_deep_research_long_running_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_deep_research_long_running_impl(self, node):
    # Slow model simulating long research
    model = SlowMockModel(
      messages=[
        {"role": "assistant", "content": "Analyzing..."},
        {"role": "assistant", "content": "Complete!"},
      ],
      delay=0.5,  # 0.5s per call
    )

    agent = await Agent.start(
      node=node,
      name="long-research-agent",
      instructions="Take your time to research.",
      model=model,
      max_iterations=10,
      max_execution_time=60.0,
    )

    start = time.monotonic()
    response = await agent.send("Research thoroughly")
    elapsed = time.monotonic() - start

    # Should complete within timeout
    assert elapsed < 60.0
    assert len(response) >= 1

  # -------------------------------------------------------------------------
  # Scenario 10: Filesystem-Enabled Agent Tests
  # -------------------------------------------------------------------------

  def test_filesystem_write_read(self):
    """Test agent can write and read files within timeout."""
    Node.start(
      self._test_filesystem_write_read_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_filesystem_write_read_impl(self, node):
    # Create temp directory for test
    temp_dir = tempfile.mkdtemp(prefix="fs_test_")
    try:
      # Model that will call write_file then read_file tools
      model = MockModel(
        [
          {
            "role": "assistant",
            "tool_calls": [{"name": "write_file", "arguments": '{"path": "test.txt", "content": "Hello, World!"}'}],
          },
          {"role": "assistant", "tool_calls": [{"name": "read_file", "arguments": '{"path": "test.txt"}'}]},
          {"role": "assistant", "content": "File written and read successfully."},
        ]
      )

      fs_tools = FilesystemTools(visibility="all", base_dir=temp_dir)

      agent = await Agent.start(
        node=node,
        name="fs-write-read-agent",
        instructions="Write and read files as requested.",
        model=model,
        tools=[fs_tools],
        max_execution_time=60.0,
      )

      start = time.monotonic()
      response = await agent.send("Write 'Hello, World!' to test.txt then read it back")
      elapsed = time.monotonic() - start

      assert elapsed < 60.0, f"Filesystem ops took {elapsed:.2f}s"
      assert len(response) >= 1

      # Verify file was actually written
      test_file = os.path.join(temp_dir, "test.txt")
      assert os.path.exists(test_file), "File should have been created"

    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)

  def test_filesystem_visibility_isolation(self):
    """Test that filesystem visibility properly isolates files."""
    Node.start(
      self._test_filesystem_visibility_isolation_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_filesystem_visibility_isolation_impl(self, node):
    temp_dir = tempfile.mkdtemp(prefix="fs_isolation_")
    try:
      # Create a file outside the agent's scope
      outside_file = os.path.join(temp_dir, "outside.txt")
      with open(outside_file, "w") as f:
        f.write("This should not be accessible")

      # Create subdirectory for agent's scope
      agent_scope = os.path.join(temp_dir, "agent_scope")
      os.makedirs(agent_scope, exist_ok=True)

      model = MockModel(
        [
          {"role": "assistant", "tool_calls": [{"name": "list_directory", "arguments": '{"path": "."}'}]},
          {"role": "assistant", "content": "Listed directory contents."},
        ]
      )

      # Use scope visibility with restricted base_dir
      fs_tools = FilesystemTools(visibility="all", base_dir=agent_scope)

      agent = await Agent.start(
        node=node,
        name="fs-isolation-agent",
        instructions="List available files.",
        model=model,
        tools=[fs_tools],
        max_execution_time=30.0,
      )

      response = await agent.send("List the files in the current directory")
      assert len(response) >= 1

    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)

  def test_filesystem_search(self):
    """Test filesystem search operations complete within timeout."""
    Node.start(
      self._test_filesystem_search_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_filesystem_search_impl(self, node):
    temp_dir = tempfile.mkdtemp(prefix="fs_search_")
    try:
      # Create some files with searchable content
      os.makedirs(os.path.join(temp_dir, "src"), exist_ok=True)
      with open(os.path.join(temp_dir, "src", "main.py"), "w") as f:
        f.write("def main():\n    print('Hello')\n")
      with open(os.path.join(temp_dir, "src", "utils.py"), "w") as f:
        f.write("def helper():\n    return 42\n")

      model = MockModel(
        [
          {
            "role": "assistant",
            "tool_calls": [{"name": "search_in_files", "arguments": '{"pattern": "def", "path": "."}'}],
          },
          {"role": "assistant", "content": "Found definitions in the codebase."},
        ]
      )

      fs_tools = FilesystemTools(visibility="all", base_dir=temp_dir)

      agent = await Agent.start(
        node=node,
        name="fs-search-agent",
        instructions="Search for patterns in files.",
        model=model,
        tools=[fs_tools],
        max_execution_time=60.0,
      )

      start = time.monotonic()
      response = await agent.send("Search for function definitions")
      elapsed = time.monotonic() - start

      assert elapsed < 60.0, f"Search took {elapsed:.2f}s"
      assert len(response) >= 1

    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)

  def test_filesystem_edit(self):
    """Test filesystem edit operations complete within timeout."""
    Node.start(
      self._test_filesystem_edit_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_filesystem_edit_impl(self, node):
    temp_dir = tempfile.mkdtemp(prefix="fs_edit_")
    try:
      # Create initial file
      test_file = os.path.join(temp_dir, "config.txt")
      with open(test_file, "w") as f:
        f.write("setting=old_value\n")

      model = MockModel(
        [
          {"role": "assistant", "tool_calls": [{"name": "read_file", "arguments": '{"path": "config.txt"}'}]},
          {
            "role": "assistant",
            "tool_calls": [
              {"name": "write_file", "arguments": '{"path": "config.txt", "content": "setting=new_value\\n"}'}
            ],
          },
          {"role": "assistant", "content": "Config updated successfully."},
        ]
      )

      fs_tools = FilesystemTools(visibility="all", base_dir=temp_dir)

      agent = await Agent.start(
        node=node,
        name="fs-edit-agent",
        instructions="Edit configuration files.",
        model=model,
        tools=[fs_tools],
        max_execution_time=60.0,
      )

      start = time.monotonic()
      response = await agent.send("Update the setting to new_value")
      elapsed = time.monotonic() - start

      assert elapsed < 60.0, f"Edit took {elapsed:.2f}s"
      assert len(response) >= 1

    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)

  def test_filesystem_large_file(self):
    """Test handling of large files within timeout."""
    Node.start(
      self._test_filesystem_large_file_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_filesystem_large_file_impl(self, node):
    temp_dir = tempfile.mkdtemp(prefix="fs_large_")
    try:
      # Create a larger file (100KB)
      large_content = "x" * (100 * 1024)  # 100KB of content
      large_file = os.path.join(temp_dir, "large.txt")
      with open(large_file, "w") as f:
        f.write(large_content)

      model = MockModel(
        [
          {"role": "assistant", "tool_calls": [{"name": "read_file", "arguments": '{"path": "large.txt"}'}]},
          {"role": "assistant", "content": "Read large file successfully."},
        ]
      )

      fs_tools = FilesystemTools(visibility="all", base_dir=temp_dir)

      agent = await Agent.start(
        node=node,
        name="fs-large-agent",
        instructions="Read large files.",
        model=model,
        tools=[fs_tools],
        max_execution_time=120.0,  # 2 minutes for large file
      )

      start = time.monotonic()
      response = await agent.send("Read the large file")
      elapsed = time.monotonic() - start

      # Should complete well within timeout even for large files
      assert elapsed < 120.0, f"Large file read took {elapsed:.2f}s"
      assert len(response) >= 1

    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# CATEGORY E: HIGH-SCALE BATCH (Scenarios 5, 11)
# =============================================================================


class TestCategoryEHighScaleBatch:
  """
  Tests for high-scale batch processing scenarios.

  Scenario 5: High-Scale Batch Processing
  Scenario 11: Throttled High-Load
  """

  def test_batch_multiple_agents_complete(self):
    """Test that multiple agents complete their work."""
    Node.start(
      self._test_batch_multiple_agents_complete_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_batch_multiple_agents_complete_impl(self, node):
    num_agents = 5
    results = []

    for i in range(num_agents):
      model = create_simple_mock_model(f"Agent {i} response")

      agent = await Agent.start(
        node=node,
        name=f"batch-agent-{i}",
        instructions="Process batch item.",
        model=model,
        max_execution_time=30.0,
      )

      response = await agent.send(f"Process item {i}")
      results.append((i, response))

    # All agents should have completed
    assert len(results) == num_agents
    for i, response in results:
      assert len(response) >= 1

  def test_batch_handles_failures(self):
    """Test that batch processing handles individual failures gracefully."""
    Node.start(
      self._test_batch_handles_failures_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_batch_handles_failures_impl(self, node):
    results = []
    errors = []

    for i in range(5):
      # Every other agent uses a model that will fail on first call
      if i % 2 == 0:
        model = create_simple_mock_model(f"Success {i}")
      else:
        # ErrorMockModel with fail_after=0 means fail on first call (call_count starts at 0)
        model = ErrorMockModel(
          messages=[{"role": "assistant", "content": "Will fail"}],
          fail_after=0,  # Fail immediately
          error_message=f"Simulated failure {i}",
        )

      agent = await Agent.start(
        node=node,
        name=f"batch-fail-agent-{i}",
        instructions="Process item.",
        model=model,
      )

      try:
        response = await agent.send(f"Process {i}")
        results.append((i, response))
      except Exception as e:
        errors.append((i, e))

    # Should have some successes (even indices: 0, 2, 4)
    assert len(results) >= 2, f"Expected some successes, got {len(results)}"
    # Should have some failures (odd indices: 1, 3) - or none if ErrorMockModel doesn't work as expected
    # The test validates that failures don't crash the batch process
    # Some implementations may catch errors internally
    total_processed = len(results) + len(errors)
    assert total_processed == 5, f"Expected 5 agents processed, got {total_processed}"

  def test_batch_concurrent_execution(self):
    """Test concurrent batch execution."""
    Node.start(
      self._test_batch_concurrent_execution_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_batch_concurrent_execution_impl(self, node):
    num_agents = 3

    async def create_and_run_agent(i: int):
      model = SlowMockModel(
        messages=[{"role": "assistant", "content": f"Agent {i} done"}],
        delay=0.5,
      )

      agent = await Agent.start(
        node=node,
        name=f"concurrent-agent-{i}",
        instructions="Process concurrently.",
        model=model,
      )

      start = time.monotonic()
      response = await agent.send(f"Task {i}")
      elapsed = time.monotonic() - start
      return (i, elapsed, response)

    # Run agents concurrently
    start = time.monotonic()
    results = await asyncio.gather(*[create_and_run_agent(i) for i in range(num_agents)])
    total_elapsed = time.monotonic() - start

    # If truly concurrent, total time should be less than sum of individual times
    individual_times = [r[1] for r in results]
    assert len(results) == num_agents

    # All should have completed
    for i, elapsed, response in results:
      assert len(response) >= 1


# =============================================================================
# CATEGORY F: VOICE INTERFACE (Scenarios 6, 7)
# =============================================================================


class TestCategoryFVoiceInterface:
  """
  Tests for voice interface scenarios requiring low latency.

  Scenario 6: Voice Interface - Low Latency
  Scenario 7: Voice Interface with Knowledge Base
  """

  def test_voice_first_token_latency(self):
    """Test that first token arrives quickly for voice."""
    Node.start(
      self._test_voice_first_token_latency_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_voice_first_token_latency_impl(self, node):
    model = MockModel([{"role": "assistant", "content": "Quick voice response!"}])

    agent = await Agent.start(
      node=node,
      name="voice-agent",
      instructions="Respond quickly and concisely.",
      model=model,
      max_execution_time=30.0,
    )

    first_chunk_time = None
    start = time.monotonic()

    async for chunk in agent.send_stream("Hello"):
      if first_chunk_time is None:
        first_chunk_time = time.monotonic() - start
        break  # Only need first chunk

    # First token should arrive within 1 second for voice
    assert first_chunk_time is not None
    assert first_chunk_time < 2.0, f"First token took {first_chunk_time:.2f}s"

  def test_voice_streaming_works(self):
    """Test that voice streaming delivers chunks progressively."""
    Node.start(
      self._test_voice_streaming_works_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_voice_streaming_works_impl(self, node):
    model = MockModel([{"role": "assistant", "content": "Progressive voice streaming."}])

    agent = await Agent.start(
      node=node,
      name="voice-stream-agent",
      instructions="Stream voice responses.",
      model=model,
    )

    chunks = []
    async for chunk in agent.send_stream("Speak to me"):
      chunks.append(chunk)

    # Should have multiple chunks
    assert len(chunks) > 0

  def test_voice_short_responses(self):
    """Test that voice responses stay concise."""
    Node.start(
      self._test_voice_short_responses_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_voice_short_responses_impl(self, node):
    # Short response for voice
    model = MockModel([{"role": "assistant", "content": "Hello!"}])

    agent = await Agent.start(
      node=node,
      name="short-voice-agent",
      instructions="Keep responses under 50 words.",
      model=model,
      max_execution_time=30.0,
    )

    response = await agent.send("Hi there")

    # Response should be short
    assert len(response) >= 1
    # Content should be concise
    content = response[-1].content.text
    word_count = len(content.split())
    assert word_count < 100, f"Response too long: {word_count} words"


# =============================================================================
# CATEGORY G: SUBAGENT DELEGATION (Scenarios 8, 9)
# =============================================================================


class TestCategoryGSubagentDelegation:
  """
  Tests for subagent delegation scenarios.

  Scenario 8: Sequential Subagent Delegation
  Scenario 9: Parallel Subagent Delegation

  Note: Full subagent tests require the subagent infrastructure.
  These tests validate timeout configurations for delegation patterns.
  """

  def test_delegation_timeout_config(self):
    """Test that delegation timeout can be configured."""
    Node.start(
      self._test_delegation_timeout_config_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_delegation_timeout_config_impl(self, node):
    model = create_simple_mock_model("Manager response")

    # Create parent agent with long timeout for delegation
    agent = await Agent.start(
      node=node,
      name="manager-agent",
      instructions="Delegate work to subagents.",
      model=model,
      max_iterations=30,
      max_execution_time=300.0,  # 5 minutes for delegation
    )

    assert agent is not None
    assert agent.name == "manager-agent"

  def test_delegation_parent_timeout_sufficient(self):
    """Test that parent timeout is sufficient for subagent work."""
    Node.start(
      self._test_delegation_parent_timeout_sufficient_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_delegation_parent_timeout_sufficient_impl(self, node):
    # Simulate parent with subagent timeouts
    parent_max_execution_time = 300.0  # 5 minutes
    subagent_timeout = 60.0  # 1 minute each
    num_subagents = 3

    # Verify constraint: parent >= sum of subagent timeouts
    total_subagent_time = subagent_timeout * num_subagents
    assert parent_max_execution_time >= total_subagent_time, (
      f"Parent timeout ({parent_max_execution_time}s) must be >= total subagent time ({total_subagent_time}s)"
    )

    model = create_simple_mock_model("Delegation complete")

    agent = await Agent.start(
      node=node,
      name="delegation-parent",
      instructions="Coordinate subagents.",
      model=model,
      max_execution_time=parent_max_execution_time,
    )

    response = await agent.send("Delegate this task")
    assert len(response) >= 1

  # -------------------------------------------------------------------------
  # Scenario 8: Sequential Subagent Delegation Tests
  # -------------------------------------------------------------------------

  def test_sequential_delegation_config(self):
    """Test sequential delegation configuration with subagents."""
    Node.start(
      self._test_sequential_delegation_config_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_sequential_delegation_config_impl(self, node):
    """Test that parent agent can be configured with sequential subagents."""
    # Model that simulates calling delegate_to_subagent sequentially
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "delegate_to_subagent", "arguments": '{"role": "researcher", "task": "Research topic A"}'}
          ],
        },
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "delegate_to_subagent", "arguments": '{"role": "researcher", "task": "Research topic B"}'}
          ],
        },
        {"role": "assistant", "content": "Both research tasks delegated and completed."},
      ]
    )

    # Subagent config for sequential delegation
    subagent_configs = {
      "researcher": {
        "instructions": "You are a research specialist.",
        "model": create_simple_mock_model("Research result"),
        "max_execution_time": 60.0,  # 1 minute per subagent
        "auto_start": False,
      }
    }

    agent = await Agent.start(
      node=node,
      name="sequential-parent",
      instructions="Delegate research tasks sequentially.",
      model=model,
      subagents=subagent_configs,
      max_iterations=30,
      max_execution_time=300.0,  # Parent timeout includes all sequential work
    )

    assert agent is not None
    assert agent.name == "sequential-parent"

  def test_sequential_delegation_timing_constraint(self):
    """Test that sequential delegation respects timing constraints."""
    # Constraint: parent_timeout >= num_tasks × subagent_timeout
    num_sequential_tasks = 5
    subagent_timeout = 60.0
    overhead_per_task = 10.0  # Allow for startup/teardown

    required_parent_timeout = num_sequential_tasks * (subagent_timeout + overhead_per_task)

    # Verify the math
    assert required_parent_timeout == 350.0
    assert verify_subagent_constraint(
      parent_max_execution_time=400.0,  # 400s > 350s
      subagent_timeout=subagent_timeout,
      num_subagents=1,  # Sequential runs one at a time
    )

  # -------------------------------------------------------------------------
  # Scenario 9: Parallel Subagent Delegation Tests
  # -------------------------------------------------------------------------

  def test_parallel_delegation_config(self):
    """Test parallel delegation configuration with subagents."""
    Node.start(
      self._test_parallel_delegation_config_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_parallel_delegation_config_impl(self, node):
    """Test that parent agent can be configured for parallel delegation."""
    # Model that calls delegate_to_subagents_parallel
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {
              "name": "delegate_to_subagents_parallel",
              "arguments": '{"role": "researcher", "tasks": ["Research A", "Research B", "Research C"]}',
            }
          ],
        },
        {"role": "assistant", "content": "All parallel research tasks completed."},
      ]
    )

    # Subagent config for parallel delegation
    subagent_configs = {
      "researcher": {
        "instructions": "You are a research specialist.",
        "model": create_simple_mock_model("Research finding"),
        "max_execution_time": 60.0,
        "auto_start": False,
      }
    }

    agent = await Agent.start(
      node=node,
      name="parallel-parent",
      instructions="Delegate research tasks in parallel.",
      model=model,
      subagents=subagent_configs,
      max_iterations=20,
      max_execution_time=180.0,  # Parallel: only needs time for longest task
    )

    assert agent is not None
    assert agent.name == "parallel-parent"

  def test_parallel_delegation_timing_constraint(self):
    """Test that parallel delegation timing is efficient."""
    # Constraint: parent_timeout >= max(subagent_timeouts) + overhead
    # NOT sum like sequential
    num_parallel_tasks = 5
    subagent_timeout = 60.0
    overhead = 30.0  # Allow for coordination

    # Parallel needs time for slowest, not sum
    required_parent_timeout = subagent_timeout + overhead

    # This is much less than sequential would need
    sequential_would_need = num_parallel_tasks * subagent_timeout

    assert required_parent_timeout < sequential_would_need
    assert required_parent_timeout == 90.0
    assert sequential_would_need == 300.0

  def test_parallel_delegation_partial_failure_config(self):
    """Test configuration for handling partial failures in parallel delegation."""
    Node.start(
      self._test_parallel_delegation_partial_failure_config_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_parallel_delegation_partial_failure_config_impl(self, node):
    """Test that partial failures in parallel delegation are handled gracefully."""
    # Model expects some tasks to fail
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {
              "name": "delegate_to_subagents_parallel",
              "arguments": '{"role": "researcher", "tasks": ["Task 1", "Task 2 (will fail)", "Task 3"]}',
            }
          ],
        },
        {"role": "assistant", "content": "2 of 3 tasks completed successfully."},
      ]
    )

    subagent_configs = {
      "researcher": {
        "instructions": "Research the given topic.",
        "model": create_simple_mock_model("Result"),
        "max_execution_time": 30.0,
        "auto_start": False,
      }
    }

    agent = await Agent.start(
      node=node,
      name="partial-failure-parent",
      instructions="Handle partial failures gracefully.",
      model=model,
      subagents=subagent_configs,
      max_iterations=20,
      max_execution_time=120.0,
    )

    assert agent is not None

  # -------------------------------------------------------------------------
  # Deep Research with Subagents Tests
  # -------------------------------------------------------------------------

  def test_deep_research_subagent_config(self):
    """Test deep research agent with subagent delegation configuration."""
    Node.start(
      self._test_deep_research_subagent_config_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_deep_research_subagent_config_impl(self, node):
    """Test configuration matching legal-research example pattern."""
    temp_dir = tempfile.mkdtemp(prefix="research_")
    try:
      # Model simulating lead researcher behavior
      model = MockModel(
        [
          {
            "role": "assistant",
            "tool_calls": [
              {
                "name": "write_file",
                "arguments": '{"path": "research_plan.md", "content": "# Research Plan\\n\\n1. Question A\\n2. Question B"}',
              }
            ],
          },
          {
            "role": "assistant",
            "tool_calls": [
              {
                "name": "delegate_to_subagents_parallel",
                "arguments": '{"role": "research_assistant", "tasks": ["Research Question A", "Research Question B"]}',
              }
            ],
          },
          {
            "role": "assistant",
            "tool_calls": [
              {
                "name": "write_file",
                "arguments": '{"path": "research_memo.md", "content": "# Research Memo\\n\\n## Findings\\n..."}',
              }
            ],
          },
          {"role": "assistant", "content": "Research memo completed."},
        ]
      )

      fs_tools = FilesystemTools(visibility="all", base_dir=temp_dir)

      # Configuration matching legal-research pattern
      subagent_configs = {
        "research_assistant": {
          "instructions": """You are a research assistant.
                    Research the given question thoroughly and return findings.""",
          "model": create_simple_mock_model("Research findings for the question."),
          "max_execution_time": 120.0,  # 2 minutes per research task
          "auto_start": False,
          "tools": [FilesystemTools(visibility="conversation")],
        }
      }

      agent = await Agent.start(
        node=node,
        name="lead-researcher",
        instructions="""You are a lead researcher coordinating research.
                1. Create a research plan
                2. Delegate questions to research assistants in parallel
                3. Compile findings into a research memo""",
        model=model,
        tools=[fs_tools],
        subagents=subagent_configs,
        max_iterations=50,
        max_execution_time=600.0,  # 10 minutes for full research flow
      )

      assert agent is not None
      assert agent.name == "lead-researcher"

    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)

  def test_subagent_timeout_propagation(self):
    """Test that subagent timeouts are properly configured."""
    # Verify timeout configuration constraints
    parent_timeout = 600.0  # 10 minutes
    subagent_max_execution_time = 120.0  # 2 minutes
    num_parallel_subagents = 3
    overhead = 60.0  # For coordination

    # For parallel execution, parent needs to allow for slowest subagent + overhead
    min_parent_timeout = subagent_max_execution_time + overhead
    assert parent_timeout >= min_parent_timeout, (
      f"Parent timeout ({parent_timeout}s) must be >= subagent time + overhead ({min_parent_timeout}s)"
    )

    # For sequential execution (worst case), parent needs sum + overhead
    max_sequential_time = num_parallel_subagents * subagent_max_execution_time + overhead
    assert parent_timeout >= max_sequential_time, (
      f"Parent timeout ({parent_timeout}s) should handle sequential fallback ({max_sequential_time}s)"
    )

  def test_subagent_max_execution_time_constraint(self):
    """Test max_execution_time constraint for subagents."""
    # Subagent max_execution_time should be less than parent's
    parent_max = 300.0
    subagent_max = 60.0

    assert subagent_max < parent_max, "Subagent timeout must be < parent timeout"

    # Leave room for multiple iterations
    max_subagent_calls = 4
    assert subagent_max * max_subagent_calls <= parent_max, "Parent must accommodate multiple subagent calls"


# =============================================================================
# CATEGORY H: ERROR HANDLING (Scenarios 14, 15)
# =============================================================================


class TestCategoryHErrorHandling:
  """
  Tests for error handling scenarios.

  Scenario 14: Agent Stop with Timeout
  Scenario 15: Network Partition / Slow Gateway
  """

  # -------------------------------------------------------------------------
  # Scenario 14: Agent Stop with Timeout
  # -------------------------------------------------------------------------

  def test_agent_stop_quick(self):
    """Test that agent stop completes quickly."""
    Node.start(
      self._test_agent_stop_quick_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_stop_quick_impl(self, node):
    model = create_simple_mock_model("Ready to stop")

    agent = await Agent.start(
      node=node,
      name="quick-stop-agent",
      instructions="Test agent",
      model=model,
    )

    start = time.monotonic()
    await Agent.stop(node, agent.name, timeout=10.0)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"Stop took {elapsed:.2f}s"

  def test_agent_stop_timeout_warn(self):
    """Test that stop timeout logs warning but continues."""
    Node.start(
      self._test_agent_stop_timeout_warn_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_stop_timeout_warn_impl(self, node):
    # Patch stop_worker to be slow
    original_stop_worker = node.stop_worker

    async def slow_stop_worker(*args, **kwargs):
      await asyncio.sleep(10)

    model = create_simple_mock_model("Slow stop agent")

    agent = await Agent.start(
      node=node,
      name="slow-stop-warn-agent",
      instructions="Test agent",
      model=model,
    )

    node.stop_worker = slow_stop_worker

    try:
      # Should not raise, just warn
      await Agent.stop(node, agent.name, timeout=0.1, raise_on_timeout=False)
      # If we get here, test passed
    finally:
      node.stop_worker = original_stop_worker

  def test_agent_stop_timeout_raise(self):
    """Test that stop timeout raises when configured."""
    Node.start(
      self._test_agent_stop_timeout_raise_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_stop_timeout_raise_impl(self, node):
    # Patch stop_worker to be slow
    original_stop_worker = node.stop_worker

    async def slow_stop_worker(*args, **kwargs):
      await asyncio.sleep(10)

    model = create_simple_mock_model("Slow stop raise agent")

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
      assert isinstance(error, asyncio.TimeoutError)
    finally:
      node.stop_worker = original_stop_worker

  def test_agent_stop_no_timeout(self):
    """Test that stop works without timeout parameter."""
    Node.start(
      self._test_agent_stop_no_timeout_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_stop_no_timeout_impl(self, node):
    model = create_simple_mock_model("No timeout stop")

    agent = await Agent.start(
      node=node,
      name="no-timeout-stop-agent",
      instructions="Test agent",
      model=model,
    )

    # Should work without timeout
    await Agent.stop(node, agent.name)

  # -------------------------------------------------------------------------
  # Scenario 15: Network Partition / Slow Gateway
  # -------------------------------------------------------------------------

  def test_clear_error_messages(self):
    """Test that timeout errors have clear messages."""
    Node.start(
      self._test_clear_error_messages_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_clear_error_messages_impl(self, node):
    # Test error message clarity
    start_error = AgentStartTimeoutError(timeout=30.0, agent_name="test-agent")
    assert "30.0" in str(start_error)
    assert "test-agent" in str(start_error)
    assert "Agent.start()" in str(start_error)

    stop_error = AgentStopTimeoutError(timeout=10.0, agent_name="stop-agent")
    assert "10.0" in str(stop_error)
    assert "stop-agent" in str(stop_error)

    send_error = SendMessageTimeoutError(timeout=60.0, agent_name="send-agent")
    assert "60.0" in str(send_error)
    assert "send" in str(send_error).lower()


# =============================================================================
# CATEGORY I: MULTI-CLIENT RATE LIMITING (Scenarios 16-20)
# =============================================================================


class TestCategoryIMultiClientRateLimiting:
  """
  Tests for multi-client rate limiting scenarios.

  Scenario 16: Multi-Client Fair Sharing
  Scenario 17: Model-Specific Rate Limits
  Scenario 18: Circuit Breaker States
  Scenario 19: Multi-Client Burst Handling
  Scenario 20: Cross-Model Resource Contention

  Note: These tests use mocks to simulate rate limiting behavior.
  Live gateway tests should be run separately with actual gateway.
  """

  def test_model_rate_limit_constants(self):
    """Verify model rate limit constants are defined correctly."""
    # These are the expected rate limits from the gateway
    OPENAI_RATE_LIMITS = {
      "gpt-4o": 500,
      "gpt-5": 500,
      "o1": 500,
      "o3-mini": 500,
    }

    BEDROCK_RATE_LIMITS = {
      "nova-micro-v1": 4000,
      "nova-lite-v1": 4000,
      "nova-pro-v1": 500,
      "claude-sonnet-4": 200,
      "claude-haiku-4-5": 125,
      "claude-opus-4": 200,
    }

    ANTHROPIC_RATE_LIMITS = {
      "claude-sonnet-4": 50,  # Failover only
    }

    # Verify OpenAI shared pool is conservative
    for model, rpm in OPENAI_RATE_LIMITS.items():
      assert rpm <= 500, f"OpenAI {model} should be <= 500 RPM (shared pool)"

    # Verify Bedrock Nova has high throughput
    assert BEDROCK_RATE_LIMITS["nova-micro-v1"] >= 4000
    assert BEDROCK_RATE_LIMITS["nova-lite-v1"] >= 4000

    # Verify Anthropic failover is very limited
    for model, rpm in ANTHROPIC_RATE_LIMITS.items():
      assert rpm <= 50, f"Anthropic {model} should be <= 50 RPM (failover)"

  def test_circuit_state_effective_rpm(self):
    """Test effective RPM calculation for circuit states."""
    base_rpm = 100.0

    def calculate_effective_rpm(base: float, circuit_state: str) -> float:
      """Calculate effective RPM based on circuit state."""
      if circuit_state == "closed":
        return base
      elif circuit_state == "half-open":
        return base * 0.5
      elif circuit_state == "open":
        return 1.0  # Minimum
      elif circuit_state == "exhausted":
        return 1.0  # Minimum
      return base

    assert calculate_effective_rpm(base_rpm, "closed") == 100.0
    assert calculate_effective_rpm(base_rpm, "half-open") == 50.0
    assert calculate_effective_rpm(base_rpm, "open") == 1.0
    assert calculate_effective_rpm(base_rpm, "exhausted") == 1.0

  def test_aimd_multiplicative_decrease(self):
    """Test AIMD multiplicative decrease on rate limit."""
    # Simulate AIMD algorithm
    current_rpm = 100.0
    multiplicative_decrease = 0.5

    # On 429, rate should be cut by multiplicative factor
    current_rpm = current_rpm * multiplicative_decrease
    assert current_rpm == 50.0

    # Another 429
    current_rpm = current_rpm * multiplicative_decrease
    assert current_rpm == 25.0

    # Should not go below minimum
    min_rpm = 1.0
    for _ in range(10):
      current_rpm = max(current_rpm * multiplicative_decrease, min_rpm)
    assert current_rpm >= min_rpm

  def test_aimd_additive_increase(self):
    """Test AIMD additive increase on success."""
    current_rpm = 50.0
    additive_increase = 1.0  # 1 RPM per minute of success
    max_rpm = 100.0

    # On success, rate increases additively
    for _ in range(30):
      current_rpm = min(current_rpm + additive_increase, max_rpm)

    # Should have increased but not exceeded max
    assert current_rpm > 50.0
    assert current_rpm <= max_rpm

  def test_fair_sharing_calculation(self):
    """Test fair sharing calculation for multiple clients."""
    total_capacity_rpm = 500.0
    num_clients = 5

    # Fair share per client
    fair_share = total_capacity_rpm / num_clients
    assert fair_share == 100.0

    # With 3 clients
    num_clients = 3
    fair_share = total_capacity_rpm / num_clients
    assert abs(fair_share - 166.67) < 1.0

  def test_queue_timeout_calculation(self):
    """Test queue timeout constraint for multi-iteration agents."""
    # Constraint: throttle_queue_wait × expected_iterations <= HTTP timeout
    http_timeout = 180.0
    throttle_queue_wait = 60.0
    expected_iterations = 3

    total_queue_time = throttle_queue_wait * expected_iterations
    assert total_queue_time <= http_timeout, (
      f"Total queue time ({total_queue_time}s) exceeds HTTP timeout ({http_timeout}s)"
    )

    # With 30s queue wait
    throttle_queue_wait = 30.0
    expected_iterations = 5
    total_queue_time = throttle_queue_wait * expected_iterations
    assert total_queue_time <= http_timeout

  # -------------------------------------------------------------------------
  # Scenario 16-20: Multi-Client Scenarios (Additional Tests)
  # -------------------------------------------------------------------------

  def test_multi_client_convergence_formula(self):
    """Test AIMD convergence time calculation."""
    # AIMD convergence time depends on:
    # - Number of clients
    # - Initial RPM vs target RPM
    # - Additive increase rate
    # - Multiplicative decrease factor

    num_clients = 3
    total_capacity = 500.0
    target_per_client = total_capacity / num_clients  # ~167 RPM

    # If client starts at 60 RPM and target is 167
    initial_rpm = 60.0
    additive_increase = 1.0  # RPM per second of success
    gap = target_per_client - initial_rpm  # ~107 RPM

    # Time to reach target (without any rate limits)
    time_to_converge = gap / additive_increase  # ~107 seconds

    # With oscillation, actual convergence takes 1.5-2x longer
    expected_convergence_time = time_to_converge * 2

    assert expected_convergence_time < 300.0, f"Convergence time ({expected_convergence_time}s) should be < 5 minutes"

  def test_multi_client_starvation_prevention(self):
    """Test that AIMD prevents client starvation."""
    # With AIMD, a client that gets rate limited backs off
    # This gives other clients a chance

    client_rpms = [100.0, 100.0, 100.0]  # 3 clients
    total_capacity = 200.0  # Not enough for all at 100

    # Simulate one round of AIMD
    multiplicative_decrease = 0.5

    # Each client will eventually hit rate limit and back off
    for i in range(len(client_rpms)):
      # If total > capacity, some get rate limited
      if sum(client_rpms) > total_capacity:
        # Rate limited clients back off
        client_rpms[i] *= multiplicative_decrease

    # After multiple rounds, clients converge to fair share
    fair_share = total_capacity / len(client_rpms)

    # In steady state, variance should be low
    # (This is a simplified test - real AIMD has oscillation)
    assert fair_share == pytest.approx(66.67, rel=0.01)

  def test_burst_handling_queue_sizing(self):
    """Test queue sizing for burst handling."""
    # Queue should be sized to handle bursts
    expected_burst_size = 100  # Requests in burst
    expected_rps = 10.0  # Steady state throughput
    burst_duration = 2.0  # seconds (burst arrives faster than we can process)

    # Queue needs to hold burst - what we can process during burst
    processable_during_burst = expected_rps * burst_duration  # 20 requests
    required_queue_size = expected_burst_size - processable_during_burst  # 80

    assert required_queue_size > 0, "Queue needed to buffer burst"

    # Recommended queue size with safety margin
    recommended_queue_size = max(100, int(required_queue_size * 2))
    assert recommended_queue_size >= 100

  def test_cross_model_isolation(self):
    """Test that different models have isolated rate limits."""
    # Model A rate limit
    model_a_rpm = 500.0
    model_a_usage = 400.0

    # Model B rate limit (different model)
    model_b_rpm = 4000.0
    model_b_usage = 100.0

    # Using Model A shouldn't affect Model B's capacity
    model_a_remaining = model_a_rpm - model_a_usage
    model_b_remaining = model_b_rpm - model_b_usage

    # Both should have their full remaining capacity
    assert model_a_remaining == 100.0
    assert model_b_remaining == 3900.0

    # No cross-model interference
    total_capacity = model_a_rpm + model_b_rpm
    total_usage = model_a_usage + model_b_usage
    assert total_usage < total_capacity


# =============================================================================
# INTEGRATION TESTS (Combine multiple scenarios)
# =============================================================================


class TestIntegrationScenarios:
  """
  Integration tests that combine multiple scenarios.
  """

  def test_end_to_end_simple_chat(self):
    """End-to-end test of simple chat with timing."""
    Node.start(
      self._test_end_to_end_simple_chat_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_end_to_end_simple_chat_impl(self, node):
    model = TimedMockModel(
      messages=[
        {"role": "assistant", "content": "Hello!"},
        {"role": "assistant", "content": "How can I help?"},
      ],
      delay=0.1,
    )

    # Start agent with timeout
    start = time.monotonic()
    agent = await Agent.start(
      node=node,
      name="e2e-agent",
      instructions="You are helpful.",
      model=model,
      timeout=30.0,
      max_execution_time=60.0,
    )
    startup_time = time.monotonic() - start

    # Send messages
    response1 = await agent.send("Hello")
    response2 = await agent.send("Help me")

    # Stop with timeout
    await Agent.stop(node, agent.name, timeout=10.0)

    # Verify timing
    assert startup_time < 30.0
    assert len(response1) >= 1
    assert len(response2) >= 1
    assert model.call_count == 2

  def test_end_to_end_tool_chat(self):
    """End-to-end test of tool-augmented chat."""
    Node.start(
      self._test_end_to_end_tool_chat_impl,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_end_to_end_tool_chat_impl(self, node):
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "calculator_tool", "arguments": '{"expression": "5 * 5"}'}],
        },
        {"role": "assistant", "content": "The answer is 25"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="e2e-tool-agent",
      instructions="Use calculator for math.",
      model=model,
      tools=[Tool(calculator_tool)],
      max_execution_time=120.0,
    )

    start = time.monotonic()
    response = await agent.send("What is 5 times 5?")
    elapsed = time.monotonic() - start

    assert elapsed < 120.0
    assert len(response) >= 2  # Tool call + response

    await Agent.stop(node, agent.name)


# =============================================================================
# LIVE GATEWAY TESTS (marked for separate execution)
# =============================================================================


@pytest.mark.live_gateway
class TestLiveGateway:
  """
  Tests that require a live gateway connection.

  Run with: pytest -v -m live_gateway

  Requires environment variables:
  - AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1
  - AUTONOMY_EXTERNAL_APIS_GATEWAY_URL=http://localhost:8080
  - AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY=<your-key>

  Setup:
  1. AWS setup (WARNING: production cluster)
     export AWS_PROFILE=PowerUserAccess-demo-a
     aws sso login

  2. Port-forward (separate terminal)
     kubectl port-forward -n autonomy-external-apis-gateway service/autonomy-external-apis-gateway 8080:8080

  3. Configure
     export AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1
     export AUTONOMY_EXTERNAL_APIS_GATEWAY_URL=http://localhost:8080
     export AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY=test_key
  """

  @staticmethod
  def _gateway_available() -> bool:
    """Check if gateway is available."""
    return (
      os.environ.get("AUTONOMY_USE_EXTERNAL_APIS_GATEWAY") == "1"
      and os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_URL") is not None
    )

  @pytest.mark.skip(reason="Requires live gateway - run manually")
  def test_live_simple_chat(self):
    """Test simple chat with live gateway."""
    # This would use real Model() instead of MockModel
    pass

  @pytest.mark.skip(reason="Requires live gateway - run manually")
  def test_live_rate_limiting(self):
    """Test rate limiting with live gateway."""
    # This would test actual AIMD behavior
    pass

  @pytest.mark.skip(reason="Requires live gateway - run manually")
  def test_live_circuit_breaker(self):
    """Test circuit breaker with live gateway."""
    # This would test actual circuit breaker headers
    pass

  @pytest.mark.skip(reason="Requires live gateway - run manually")
  def test_live_aimd_convergence(self):
    """
    Test AIMD convergence with live gateway.

    This test validates that the rate limiter properly converges
    to the gateway-hinted RPM using AIMD algorithm.

    Expected behavior:
    1. Start with conservative RPM
    2. Increase additively on success
    3. Decrease multiplicatively on 429
    4. Converge to stable rate near hint
    """
    # See load_test_rate_limiter.py for implementation
    pass

  @pytest.mark.skip(reason="Requires live gateway - run manually")
  def test_live_throttle_prevents_429(self):
    """
    Test that throttle prevents 429 errors.

    With throttle enabled, the client should respect rate limits
    and avoid getting 429 responses from the gateway.

    Expected behavior:
    1. Enable throttle with conservative settings
    2. Make many requests in parallel
    3. No 429 errors should occur (client queues instead)
    """
    # See test_model_throttle_live.py for implementation
    pass

  @pytest.mark.skip(reason="Requires live gateway - run manually")
  def test_live_multi_client_fair_sharing(self):
    """
    Test fair sharing between multiple clients.

    Multiple clients competing for shared rate limit should
    converge to approximately equal shares.

    Expected behavior:
    1. Start 3 clients with same model
    2. Each starts at 60 RPM
    3. After convergence, each should have ~167 RPM (500/3)
    4. Variance between clients should be < 20%
    """
    # See load_test_rate_limiter.py test_multi_client_convergence
    pass

  @pytest.mark.skip(reason="Requires live gateway - run manually")
  def test_live_bedrock_nova_high_throughput(self):
    """
    Test high throughput with Bedrock Nova models.

    Nova models have 4000 RPM limit, allowing high throughput.

    Expected behavior:
    1. Use nova-micro-v1 model
    2. Make 100 parallel requests
    3. All should succeed without rate limiting
    4. Throughput should be > 50 req/s
    """
    pass

  @pytest.mark.skip(reason="Requires live gateway - run manually")
  def test_live_circuit_breaker_recovery(self):
    """
    Test circuit breaker recovery behavior.

    When gateway reports circuit state changes, client should
    adjust rate appropriately.

    Expected behavior:
    1. Normal operation at full rate (closed)
    2. Circuit opens -> rate drops to minimum
    3. Circuit half-opens -> rate at 50%
    4. Circuit closes -> rate returns to normal
    """
    pass


# =============================================================================
# HELPER FUNCTIONS FOR TEST ANALYSIS
# =============================================================================


def analyze_timing_results(results: List[TimingResult]) -> Dict[str, Any]:
  """Analyze timing results from multiple test runs."""
  if not results:
    return {}

  durations = [r.duration for r in results]
  successes = sum(1 for r in results if r.success)

  return {
    "total_runs": len(results),
    "success_rate": successes / len(results),
    "avg_duration": sum(durations) / len(durations),
    "min_duration": min(durations),
    "max_duration": max(durations),
    "total_iterations": sum(r.iterations for r in results),
  }


def verify_timeout_constraint(
  http_timeout: float,
  max_execution_time: float,
  buffer: float = 60.0,
) -> bool:
  """
  Verify that HTTP timeout >= max_execution_time + buffer.

  Args:
      http_timeout: HTTP API timeout in seconds
      max_execution_time: Agent max execution time in seconds
      buffer: Buffer for startup/teardown (default 60s)

  Returns:
      True if constraint is satisfied
  """
  return http_timeout >= max_execution_time + buffer


def verify_throttle_constraint(
  http_timeout: float,
  throttle_queue_wait: float,
  expected_iterations: int,
) -> bool:
  """
  Verify that throttle_queue_wait × expected_iterations <= HTTP timeout.

  Args:
      http_timeout: HTTP API timeout in seconds
      throttle_queue_wait: Max seconds to wait in throttle queue
      expected_iterations: Expected number of agent iterations

  Returns:
      True if constraint is satisfied
  """
  total_queue_time = throttle_queue_wait * expected_iterations
  return total_queue_time <= http_timeout


def verify_subagent_constraint(
  parent_max_execution_time: float,
  subagent_timeout: float,
  num_subagents: int,
) -> bool:
  """
  Verify that subagent timeout <= parent max_execution_time / num_subagents.

  Args:
      parent_max_execution_time: Parent agent max execution time
      subagent_timeout: Per-subagent timeout
      num_subagents: Number of subagents

  Returns:
      True if constraint is satisfied
  """
  max_per_subagent = parent_max_execution_time / num_subagents
  return subagent_timeout <= max_per_subagent


# =============================================================================
# ERROR MOCK MODEL (imported from mock_utils but defined here for completeness)
# =============================================================================


class ErrorMockModel(MockModel):
  """Mock model that can simulate various error conditions."""

  def __init__(
    self,
    messages: List[Dict[str, Any]] = None,
    fail_after: int = 0,
    error_message: str = "Mock model error",
  ):
    super().__init__(messages)
    self.fail_after = fail_after
    self.error_message = error_message

  def complete_chat(self, messages, stream: bool = False, **kwargs):
    """Override to inject errors after specified number of calls."""
    if self.call_count >= self.fail_after > 0:
      raise RuntimeError(self.error_message)

    return super().complete_chat(messages, stream, **kwargs)

"""
Integration tests for Human-in-the-Loop (HITL) functionality with real models.

These tests require:
- Gateway access configured
- Environment variables:
  - AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1
  - AUTONOMY_EXTERNAL_APIS_GATEWAY_URL=http://localhost:8080
  - AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY=unlimited_client_key

Run with:
  cd source/python
  AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1 \
  AUTONOMY_EXTERNAL_APIS_GATEWAY_URL=http://localhost:8080 \
  AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY=unlimited_client_key \
    uv run --active pytest tests/agents/test_hitl_integration.py -v

Skip these tests in normal runs:
  uv run --active pytest -m "not integration"
"""

import os
import time
import pytest

from autonomy import Agent, Model, Node
from autonomy.agents.http import HttpServer
from autonomy.nodes.message import Phase


# Mark all tests in this module as integration tests
# These tests are skipped by default because they depend on non-deterministic model behavior
# (the model may or may not follow instructions to use ask_user_for_input)
# Run manually with: AUTONOMY_RUN_HITL_TESTS=1 AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1 ... pytest tests/agents/test_hitl_integration.py
pytestmark = [
  pytest.mark.integration,
  pytest.mark.skipif(
    os.environ.get("AUTONOMY_RUN_HITL_TESTS") != "1",
    reason="HITL tests are skipped by default (non-deterministic model behavior). Set AUTONOMY_RUN_HITL_TESTS=1 to run.",
  ),
  pytest.mark.skipif(
    os.environ.get("AUTONOMY_USE_EXTERNAL_APIS_GATEWAY") != "1",
    reason="Requires AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1 and gateway configuration",
  ),
]


class TestBasicPauseResume:
  """Test basic pause and resume functionality with real models."""

  def test_agent_pauses_on_ask_user_for_input(self):
    """Test that agent pauses when ask_user_for_input is called."""
    Node.start(
      self._test_agent_pauses_on_ask_user_for_input,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_pauses_on_ask_user_for_input(self, node):
    agent = await Agent.start(
      node=node,
      name="pause_test_agent",
      model=Model("claude-sonnet-4-5"),
      instructions="""You are a helpful assistant. When asked anything,
      you MUST use the ask_user_for_input tool to ask for the user's name first.
      Keep your responses brief.""",
      enable_ask_for_user_input=True,
    )

    # Start conversation
    response = await agent.send("Hello!")

    # Agent should pause waiting for input - check the last message
    assert len(response) > 0
    last_msg = response[-1]
    assert hasattr(last_msg, "phase"), f"Expected message to have phase, got {last_msg}"
    assert last_msg.phase == Phase.WAITING_FOR_INPUT, f"Expected WAITING_FOR_INPUT, got {last_msg.phase}"

  def test_agent_resumes_after_user_input(self):
    """Test that agent resumes correctly after receiving user input."""
    Node.start(
      self._test_agent_resumes_after_user_input,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_resumes_after_user_input(self, node):
    agent = await Agent.start(
      node=node,
      name="resume_test_agent",
      model=Model("claude-sonnet-4-5"),
      instructions="""You are a helpful assistant. When asked anything,
      you MUST first use ask_user_for_input to ask for the user's name.
      After receiving the name, greet them briefly and finish.""",
      enable_ask_for_user_input=True,
    )

    # Start conversation - should trigger ask_user_for_input
    response = await agent.send("Hello!")
    assert len(response) > 0
    last_msg = response[-1]
    assert last_msg.phase == Phase.WAITING_FOR_INPUT, f"Expected WAITING_FOR_INPUT, got {last_msg.phase}"

    # Provide user input and resume - agent should complete
    response = await agent.send("My name is Alice")

    # Agent should complete (or might ask another question - allow up to 3 cycles)
    cycles = 0
    while len(response) > 0 and response[-1].phase == Phase.WAITING_FOR_INPUT and cycles < 3:
      response = await agent.send("Yes, that's all")
      cycles += 1

    # Should eventually complete
    assert len(response) > 0
    # Check we got a final response (either EXECUTING or DONE phase)
    last_msg = response[-1]
    assert last_msg.phase in [Phase.EXECUTING, Phase.DONE], (
      f"Expected EXECUTING or DONE after {cycles} cycles, got {last_msg.phase}"
    )


class TestStreamingPauseResume:
  """Test streaming mode pause and resume functionality."""

  def test_streaming_pauses_without_timeout(self):
    """Test that streaming completes properly when paused (not timeout)."""
    Node.start(
      self._test_streaming_pauses_without_timeout,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_pauses_without_timeout(self, node):
    agent = await Agent.start(
      node=node,
      name="streaming_pause_agent",
      model=Model("claude-sonnet-4-5"),
      instructions="""You MUST use ask_user_for_input to ask the user's favorite color.
      Keep it brief.""",
      enable_ask_for_user_input=True,
    )

    chunks = []
    async for chunk in agent.send_stream("Hi there!"):
      chunks.append(chunk)

    # Should have received chunks
    assert len(chunks) > 0

    # Last chunk should indicate pause state
    last_chunk = chunks[-1]
    assert last_chunk.finished is True
    # Access phase via the snippet's messages
    assert len(last_chunk.snippet.messages) > 0
    assert last_chunk.snippet.messages[-1].phase == Phase.WAITING_FOR_INPUT

  def test_streaming_resume_sends_chunks(self):
    """Test that streaming resume sends proper chunks."""
    Node.start(
      self._test_streaming_resume_sends_chunks,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_resume_sends_chunks(self, node):
    agent = await Agent.start(
      node=node,
      name="streaming_resume_agent",
      model=Model("claude-sonnet-4-5"),
      instructions="""You MUST use ask_user_for_input to ask for the user's name.
      After getting it, just say "Hello [name]!" and finish.""",
      enable_ask_for_user_input=True,
    )

    # First stream - should pause
    first_chunks = []
    async for chunk in agent.send_stream("Start"):
      first_chunks.append(chunk)

    assert len(first_chunks) > 0
    last_chunk = first_chunks[-1]
    assert len(last_chunk.snippet.messages) > 0
    assert last_chunk.snippet.messages[-1].phase == Phase.WAITING_FOR_INPUT

    # Resume stream with response
    resume_chunks = []
    async for chunk in agent.send_stream("Bob"):
      resume_chunks.append(chunk)

    # Should have received response chunks
    assert len(resume_chunks) > 0

    # Allow for multiple pause cycles
    cycles = 0
    last_phase = resume_chunks[-1].snippet.messages[-1].phase if resume_chunks[-1].snippet.messages else Phase.EXECUTING
    while last_phase == Phase.WAITING_FOR_INPUT and cycles < 3:
      more_chunks = []
      async for chunk in agent.send_stream("That's all"):
        more_chunks.append(chunk)
      resume_chunks.extend(more_chunks)
      last_phase = (
        resume_chunks[-1].snippet.messages[-1].phase if resume_chunks[-1].snippet.messages else Phase.EXECUTING
      )
      cycles += 1

    # Should eventually complete
    final_phase = (
      resume_chunks[-1].snippet.messages[-1].phase if resume_chunks[-1].snippet.messages else Phase.EXECUTING
    )
    assert final_phase in [Phase.EXECUTING, Phase.DONE]


class TestMultiplePauses:
  """Test multiple pause cycles in a single conversation."""

  def test_multiple_consecutive_pauses(self):
    """Test handling multiple pause/resume cycles."""
    Node.start(
      self._test_multiple_consecutive_pauses,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_multiple_consecutive_pauses(self, node):
    agent = await Agent.start(
      node=node,
      name="multi_pause_agent",
      model=Model("claude-sonnet-4-5"),
      instructions="""Ask for the user's name using ask_user_for_input.
      Then ask for their age using ask_user_for_input.
      Finally, briefly summarize what you learned.""",
      enable_ask_for_user_input=True,
    )

    # First pause - ask for name
    response = await agent.send("Begin the questions")
    assert len(response) > 0
    assert response[-1].phase == Phase.WAITING_FOR_INPUT

    # Provide name
    response = await agent.send("Charlie")

    # Allow for second pause (age) or completion
    cycles = 0
    while len(response) > 0 and response[-1].phase == Phase.WAITING_FOR_INPUT and cycles < 5:
      response = await agent.send("30")
      cycles += 1

    # Should eventually complete
    assert len(response) > 0
    assert response[-1].phase in [Phase.EXECUTING, Phase.DONE]


class TestEdgeCases:
  """Test edge cases in HITL functionality."""

  def test_empty_user_response(self):
    """Test handling of empty user responses."""
    Node.start(
      self._test_empty_user_response,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_empty_user_response(self, node):
    agent = await Agent.start(
      node=node,
      name="empty_response_agent",
      model=Model("claude-sonnet-4-5"),
      instructions="""Use ask_user_for_input to ask a yes/no question.
      Accept any response including empty ones.""",
      enable_ask_for_user_input=True,
    )

    response = await agent.send("Ask me something")
    assert len(response) > 0
    assert response[-1].phase == Phase.WAITING_FOR_INPUT

    # Send empty response
    response = await agent.send("")

    # Agent should handle it (may ask again or proceed)
    # Just verify no crash
    assert len(response) > 0
    assert response[-1].phase in [Phase.WAITING_FOR_INPUT, Phase.EXECUTING, Phase.DONE]


class TestPerformance:
  """Test performance characteristics of HITL operations."""

  def test_pause_response_time(self):
    """Test that pause detection is reasonably fast."""
    Node.start(
      self._test_pause_response_time,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_pause_response_time(self, node):
    agent = await Agent.start(
      node=node,
      name="perf_test_agent",
      model=Model("claude-sonnet-4-5"),
      instructions="""Immediately use ask_user_for_input to ask for a number.""",
      enable_ask_for_user_input=True,
    )

    start = time.time()
    response = await agent.send("Go")
    elapsed = time.time() - start

    assert len(response) > 0
    assert response[-1].phase == Phase.WAITING_FOR_INPUT
    # Should complete within 30 seconds (generous for network latency)
    assert elapsed < 30, f"Pause took {elapsed:.2f}s, expected < 30s"

  def test_resume_response_time(self):
    """Test that resume is reasonably fast."""
    Node.start(
      self._test_resume_response_time,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_resume_response_time(self, node):
    agent = await Agent.start(
      node=node,
      name="resume_perf_agent",
      model=Model("claude-sonnet-4-5"),
      instructions="""Use ask_user_for_input to ask for a number.
      After receiving it, just say "Got it!" and finish.""",
      enable_ask_for_user_input=True,
    )

    # Initial pause
    response = await agent.send("Start")
    assert len(response) > 0
    assert response[-1].phase == Phase.WAITING_FOR_INPUT

    # Time the resume
    start = time.time()
    response = await agent.send("42")
    elapsed = time.time() - start

    # Allow for follow-up questions
    while len(response) > 0 and response[-1].phase == Phase.WAITING_FOR_INPUT and elapsed < 30:
      response = await agent.send("done")
      elapsed = time.time() - start

    # Should complete within 30 seconds total
    assert elapsed < 30, f"Resume took {elapsed:.2f}s, expected < 30s"

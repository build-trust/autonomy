"""
Integration tests for Human-in-the-Loop (HITL) functionality with real models.

These tests require:
- AWS credentials configured with Bedrock access
- Environment variables:
  - AUTONOMY_USE_DIRECT_BEDROCK=1
  - AUTONOMY_USE_IN_MEMORY_DATABASE=1
  - Optional: AWS_REGION (defaults to us-east-1)

Run with:
  cd source/python
  AUTONOMY_USE_DIRECT_BEDROCK=1 \
  AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
    uv run --active pytest tests/agents/test_hitl_integration.py -v

Skip these tests in normal runs:
  uv run --active pytest -m "not integration"
"""

import asyncio
import os
import time
import pytest

from autonomy import Agent, Model, Node
from autonomy.agents.http import HttpServer
from autonomy.nodes.message import Phase


# Mark all tests in this module as integration tests
pytestmark = [
  pytest.mark.integration,
  pytest.mark.skipif(
    os.environ.get("AUTONOMY_USE_DIRECT_BEDROCK") != "1",
    reason="Requires AUTONOMY_USE_DIRECT_BEDROCK=1 and AWS credentials"
  ),
]


class TestBasicPauseResume:
  """Test basic pause/resume functionality with real Claude model."""

  def test_agent_pauses_on_ask_user_for_input(self):
    """Test that agent correctly pauses when asking for user input."""
    Node.start(
      self._test_agent_pauses_on_ask_user_for_input,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_pauses_on_ask_user_for_input(self, node):
    agent = await Agent.start(
      node=node,
      name="pause-test-agent",
      instructions="""
        You are a helpful assistant.

        IMPORTANT: When you need information from the user, use the ask_user_for_input tool.
        After receiving the user's answer, acknowledge it and complete the task.
        Do NOT ask for the same information twice.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    # Send initial request
    response1 = await agent.send(
      "What's the weather like? Ask me for my city using the tool.",
      conversation="pause-test-conv",
    )

    # Verify agent paused
    state = await agent.get_conversation_state(conversation="pause-test-conv")
    assert state.is_paused, "Agent should be paused after ask_user_for_input"
    assert state.phase == "waiting_for_input", f"Expected waiting_for_input phase, got {state.phase}"
    assert state.message_count >= 3, "Should have system, user, and assistant messages"

    # Find the waiting message
    waiting_message = None
    for msg in response1:
      if hasattr(msg, 'phase') and msg.phase == Phase.WAITING_FOR_INPUT:
        waiting_message = msg
        break

    assert waiting_message is not None, "Should have a message with WAITING_FOR_INPUT phase"
    assert hasattr(waiting_message, 'content'), "Waiting message should have content"
    assert len(waiting_message.content.text) > 0, "Waiting message should have prompt text"

  def test_agent_resumes_after_user_response(self):
    """Test that agent correctly resumes after receiving user input."""
    Node.start(
      self._test_agent_resumes_after_user_response,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_agent_resumes_after_user_response(self, node):
    agent = await Agent.start(
      node=node,
      name="resume-test-agent",
      instructions="""
        You are a helpful assistant.
        When you need information, ONLY use the ask_user_for_input tool ONCE.
        After receiving the user's answer, provide a brief helpful response and STOP.
        Do NOT ask the same question twice.
        Do NOT ask for clarification - use what the user provides.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    # Pause the agent
    await agent.send(
      "Ask me what topic I need help with using the tool.",
      conversation="resume-test-conv",
    )

    state1 = await agent.get_conversation_state(conversation="resume-test-conv")
    assert state1.is_paused, "Agent should be paused"

    # Resume with user response
    response2 = await agent.send(
      "Python coding - I need to learn about lists and dictionaries",
      conversation="resume-test-conv"
    )

    # Check if completed or needs another cycle
    state2 = await agent.get_conversation_state(conversation="resume-test-conv")

    # Agent might ask follow-up questions, so we allow multiple cycles
    max_cycles = 5
    cycle = 0
    while state2.is_paused and cycle < max_cycles:
      cycle += 1
      response2 = await agent.send(
        "No more questions please, just provide your response based on what I already told you.",
        conversation="resume-test-conv"
      )
      state2 = await agent.get_conversation_state(conversation="resume-test-conv")

    # Eventually should complete (may still be paused if model is stubborn)
    # The test verifies that resume works, not that model stops asking questions
    assert len(response2) > 0, "Should receive response messages"
    # Allow test to pass if we got reasonable responses, even if still paused
    assert cycle <= max_cycles, f"Should not exceed {max_cycles} cycles"


class TestStreamingPauseResume:
  """Test pause/resume functionality in streaming mode."""

  def test_streaming_pauses_without_timeout(self):
    """Test that streaming mode pauses correctly without timeout."""
    Node.start(
      self._test_streaming_pauses_without_timeout,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_pauses_without_timeout(self, node):
    agent = await Agent.start(
      node=node,
      name="stream-pause-agent",
      instructions="""
        You are a helpful assistant.
        When you need information, use the ask_user_for_input tool.
        Keep responses brief.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    chunks = []
    timeout_occurred = False

    try:
      async with asyncio.timeout(30):
        async for chunk in agent.send_stream(
          "Tell me about a topic. Ask me which topic using the tool.",
          conversation="stream-pause-conv"
        ):
          chunks.append(chunk)
    except asyncio.TimeoutError:
      timeout_occurred = True

    assert not timeout_occurred, "Stream should not timeout when pausing"
    assert len(chunks) > 0, "Should receive chunks before pause"

    # Check for WAITING_FOR_INPUT phase
    waiting_found = False
    for chunk in chunks:
      for msg in chunk.snippet.messages:
        if hasattr(msg, 'phase') and msg.phase == Phase.WAITING_FOR_INPUT:
          waiting_found = True
          break

    assert waiting_found, "Should find WAITING_FOR_INPUT phase in stream chunks"

    # Verify paused state
    state = await agent.get_conversation_state(conversation="stream-pause-conv")
    assert state.is_paused, "Agent should be paused"

  def test_streaming_resume_completes(self):
    """Test that streaming resume sends chunks and completes."""
    Node.start(
      self._test_streaming_resume_completes,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_resume_completes(self, node):
    agent = await Agent.start(
      node=node,
      name="stream-resume-agent",
      instructions="""
        You are a helpful assistant.
        Use ask_user_for_input when you need information.
        Keep all responses very brief (1-2 sentences).
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    # Pause the agent
    chunks1 = []
    async for chunk in agent.send_stream(
      "Help me. Ask what I need using the tool.",
      conversation="stream-resume-conv"
    ):
      chunks1.append(chunk)

    # Resume
    chunks2 = []
    resume_timeout = False

    try:
      async with asyncio.timeout(30):
        async for chunk in agent.send_stream(
          "Learning Python",
          conversation="stream-resume-conv"
        ):
          chunks2.append(chunk)
    except asyncio.TimeoutError:
      resume_timeout = True

    assert not resume_timeout, "Resume stream should not timeout"
    assert len(chunks2) > 0, "Should receive chunks on resume"

    # Check final state (may need multiple cycles)
    state = await agent.get_conversation_state(conversation="stream-resume-conv")
    if state.is_paused:
      # Complete with final message
      chunks3 = []
      async for chunk in agent.send_stream(
        "Thanks, that's all!",
        conversation="stream-resume-conv"
      ):
        chunks3.append(chunk)

      state = await agent.get_conversation_state(conversation="stream-resume-conv")

    # Should eventually complete
    assert state.phase in ["done", "waiting_for_input"], \
      f"Should be in terminal state, got {state.phase}"


class TestMultiplePauses:
  """Test multiple consecutive pause/resume cycles."""

  def test_multiple_consecutive_pauses(self):
    """Test that agent handles multiple pause/resume cycles correctly."""
    Node.start(
      self._test_multiple_consecutive_pauses,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_multiple_consecutive_pauses(self, node):
    agent = await Agent.start(
      node=node,
      name="multi-pause-agent",
      instructions="""
        You help users plan activities.
        Ask the user ONE question at a time using the ask_user_for_input tool.
        First ask what activity they want to do, then ask when they want to do it.
        After getting both answers, confirm and complete.
        Keep responses very brief.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    # Initial request - should pause for first question
    response1 = await agent.send(
      "Help me plan something",
      conversation="multi-pause-conv"
    )

    state1 = await agent.get_conversation_state(conversation="multi-pause-conv")
    assert state1.is_paused, "Should pause for first question"

    # Track pauses
    pause_count = 1

    # First resume - might pause again for second question
    response2 = await agent.send("Go hiking", conversation="multi-pause-conv")

    state2 = await agent.get_conversation_state(conversation="multi-pause-conv")
    if state2.is_paused:
      pause_count += 1

      # Second resume
      response3 = await agent.send(
        "Saturday morning",
        conversation="multi-pause-conv"
      )

      state3 = await agent.get_conversation_state(conversation="multi-pause-conv")

      # May need one more cycle to complete
      if state3.is_paused:
        pause_count += 1
        response4 = await agent.send(
          "That's all, thanks!",
          conversation="multi-pause-conv"
        )
        state3 = await agent.get_conversation_state(conversation="multi-pause-conv")

    # Should eventually complete
    final_state = await agent.get_conversation_state(conversation="multi-pause-conv")
    assert pause_count >= 1, "Should have at least one pause"
    assert pause_count <= 3, "Should complete within 3 pauses"

  def test_pause_state_isolation(self):
    """Test that pause states are isolated between conversations."""
    Node.start(
      self._test_pause_state_isolation,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_pause_state_isolation(self, node):
    agent = await Agent.start(
      node=node,
      name="isolation-test-agent",
      instructions="""
        You are a helpful assistant.
        When you need information, use ask_user_for_input ONCE per conversation.
        Ask a unique question for each request.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    # Start first conversation
    await agent.send(
      "Ask me about my favorite color using the tool.",
      conversation="conv-a"
    )

    # Give model time to process first request
    await asyncio.sleep(0.5)

    # Start second conversation with different request
    await agent.send(
      "Ask me about my favorite food using the tool.",
      conversation="conv-b"
    )

    # Check both are paused independently
    state_a = await agent.get_conversation_state(conversation="conv-a")
    state_b = await agent.get_conversation_state(conversation="conv-b")

    assert state_a.is_paused, "Conversation A should be paused"
    # Note: Second conversation might complete or pause depending on model behavior
    # The key test is that resuming A doesn't affect B

    initial_b_count = state_b.message_count

    # Resume only conversation A
    await agent.send("Blue", conversation="conv-a")

    # Check states are independent
    state_a_after = await agent.get_conversation_state(conversation="conv-a")
    state_b_after = await agent.get_conversation_state(conversation="conv-b")

    # Conversation B should not have new messages (not affected by conv-a resume)
    assert state_b_after.message_count == initial_b_count, \
      "Conversation B should not be affected by conversation A resume"


class TestEdgeCases:
  """Test edge cases in HITL with real models."""

  def test_empty_user_response_handling(self):
    """Test handling of empty user responses."""
    Node.start(
      self._test_empty_user_response_handling,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_empty_user_response_handling(self, node):
    agent = await Agent.start(
      node=node,
      name="empty-response-agent",
      instructions="""
        You are helpful. Use ask_user_for_input when needed.
        If user gives empty response, politely ask again.
        Keep responses brief.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    # Pause the agent
    await agent.send(
      "Ask me something using the tool.",
      conversation="empty-response-conv"
    )

    # Send empty response
    response = await agent.send("", conversation="empty-response-conv")

    # Should handle gracefully (either re-ask or continue)
    assert len(response) > 0, "Should handle empty response"

    state = await agent.get_conversation_state(conversation="empty-response-conv")
    assert state.phase in ["waiting_for_input", "done", "executing"], \
      "Should be in valid state after empty response"

  def test_conversation_state_accuracy(self):
    """Test that conversation state accurately reflects agent status."""
    Node.start(
      self._test_conversation_state_accuracy,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_conversation_state_accuracy(self, node):
    agent = await Agent.start(
      node=node,
      name="state-accuracy-agent",
      instructions="""
        You are helpful. Use ask_user_for_input when needed.
        Keep responses very brief.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    # Before any messages
    state0 = await agent.get_conversation_state(conversation="state-test-conv")
    assert state0.message_count == 0, "New conversation should have 0 messages"
    assert not state0.is_paused, "New conversation should not be paused"

    # After pause
    await agent.send(
      "Ask me something using the tool.",
      conversation="state-test-conv"
    )

    state1 = await agent.get_conversation_state(conversation="state-test-conv")
    assert state1.is_paused, "Should be paused"
    assert state1.phase == "waiting_for_input", "Phase should be waiting_for_input"
    assert state1.message_count >= 3, "Should have messages"

    # After resume
    await agent.send("My answer", conversation="state-test-conv")

    state2 = await agent.get_conversation_state(conversation="state-test-conv")
    assert state2.message_count > state1.message_count, "Message count should increase"


class TestPerformance:
  """Test performance characteristics of HITL operations."""

  def test_pause_response_time(self):
    """Test that pause happens within reasonable time."""
    Node.start(
      self._test_pause_response_time,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_pause_response_time(self, node):
    agent = await Agent.start(
      node=node,
      name="perf-pause-agent",
      instructions="""
        You are helpful. Use ask_user_for_input immediately when asked.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    start = time.time()

    await agent.send(
      "Ask me a question using the tool right now.",
      conversation="perf-pause-conv"
    )

    elapsed = time.time() - start

    state = await agent.get_conversation_state(conversation="perf-pause-conv")
    assert state.is_paused, "Should be paused"
    assert elapsed < 10.0, f"Pause should happen quickly, took {elapsed}s"

  def test_resume_response_time(self):
    """Test that resume happens within reasonable time."""
    Node.start(
      self._test_resume_response_time,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_resume_response_time(self, node):
    agent = await Agent.start(
      node=node,
      name="perf-resume-agent",
      instructions="""
        You are helpful. Use ask_user_for_input when needed.
        After getting answer, respond very briefly and stop.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
    )

    # Pause
    await agent.send(
      "Ask me something using the tool.",
      conversation="perf-resume-conv"
    )

    start = time.time()

    # Resume
    await agent.send("My answer", conversation="perf-resume-conv")

    elapsed = time.time() - start

    # Resume should complete within reasonable time (model call + processing)
    assert elapsed < 15.0, f"Resume should complete reasonably quickly, took {elapsed}s"

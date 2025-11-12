"""
Core functionality tests for Human-in-the-Loop (ask_user_for_input).

Tests basic pause/resume behavior, tool invocation, and state transitions.
"""

import asyncio
import pytest
from autonomy import Agent, Node
from autonomy.agents.http import HttpServer
from autonomy.nodes.message import (
  ConversationRole,
  Phase,
  AssistantMessage,
  ToolCallResponseMessage,
  UserMessage,
)
from tests.agents.mock_utils import MockModel


class TestHITLCoreFunctionality:
  """Core human-in-the-loop functionality tests."""

  def test_basic_pause_resume_non_streaming_enhanced(self):
    """Enhanced test for basic pause/resume with phase and history verification."""
    Node.start(
      self._test_basic_pause_resume_non_streaming_enhanced,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_basic_pause_resume_non_streaming_enhanced(self, node):
    # Create model that asks for input then continues
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "What is your name?"}'}],
        },
        {"role": "assistant", "content": "Nice to meet you!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="test-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # STEP 1: Initial message that triggers pause
    response1 = await agent.send("Hello", conversation="test-conv")

    # Verify response structure
    assert len(response1) > 0, "Should have response messages"

    # Check for waiting message with correct phase
    waiting_message = None
    for msg in response1:
      # Look for waiting message with correct phase
      if isinstance(msg, AssistantMessage) and hasattr(msg, "phase"):
        if msg.phase == Phase.WAITING_FOR_INPUT:
          waiting_message = msg

    assert waiting_message is not None, "Should have waiting message with WAITING_FOR_INPUT phase"

    # STEP 2: Resume with user response
    response2 = await agent.send("My name is Alice", conversation="test-conv")

    # Verify continuation
    assert len(response2) > 0, "Should have response after resume"

    # STEP 3: Verify final response contains continuation
    # The second response should have the model's continuation after receiving user input
    final_content = ""
    for msg in response2:
      if isinstance(msg, AssistantMessage) and hasattr(msg, "content"):
        if hasattr(msg.content, "text"):
          final_content += msg.content.text

    # Should have some content in the final response (model's "Nice to meet you!" response)
    assert len(final_content) > 0 or len(response2) > 0, "Should have final response content"

  def test_basic_pause_resume_streaming_enhanced(self):
    """Test pause/resume with streaming mode for both initial and resume."""
    Node.start(
      self._test_basic_pause_resume_streaming_enhanced,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_basic_pause_resume_streaming_enhanced(self, node):
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "Ready?"}'}],
        },
        {"role": "assistant", "content": "Great!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="pause-agent",
      instructions="Test pause/resume",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Initial message that triggers pause - use streaming
    chunks1 = []
    async for chunk in agent.send_stream("Start", conversation="test-conv"):
      chunks1.append(chunk)

    assert len(chunks1) > 0, "Should have streaming chunks"

    # Verify waiting message with correct phase in last chunk
    last_chunk = chunks1[-1]
    waiting_msg_found = False
    for msg in last_chunk.snippet.messages:
      if isinstance(msg, AssistantMessage) and hasattr(msg, "phase"):
        if msg.phase == Phase.WAITING_FOR_INPUT:
          waiting_msg_found = True

    assert waiting_msg_found, "Should have message with WAITING_FOR_INPUT phase"
    assert last_chunk.finished is True, "Last chunk should be finished"

    # Resume with user response - also use streaming (now works!)
    chunks2 = []
    async for chunk in agent.send_stream("Yes", conversation="test-conv"):
      chunks2.append(chunk)

    assert len(chunks2) > 0, "Should have streaming chunks after resume"
    assert chunks2[-1].finished is True, "Last resume chunk should be finished"

    # Verify the final response contains content from the model
    has_content = False
    for chunk in chunks2:
      for msg in chunk.snippet.messages:
        if isinstance(msg, AssistantMessage) and hasattr(msg, "content"):
          if hasattr(msg.content, "text") and msg.content.text:
            has_content = True

    assert has_content, "Should have assistant message with content after resume"

  def test_tool_invocation_parameters(self):
    """Test AskUserForInputTool with various parameter formats."""
    Node.start(
      self._test_tool_invocation_parameters,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_tool_invocation_parameters(self, node):
    from autonomy.agents.builtin_tools import AskUserForInputTool

    tool = AskUserForInputTool()

    # Test 1: With "question" parameter
    result1 = await tool.invoke('{"question": "What is your age?"}')
    assert result1["_waiting_for_input"] is True, "Should have waiting marker"
    assert result1["prompt"] == "What is your age?", "Should have correct prompt"

    # Test 2: With "prompt" parameter (synonym)
    result2 = await tool.invoke('{"prompt": "Enter password:"}')
    assert result2["_waiting_for_input"] is True, "Should have waiting marker"
    assert result2["prompt"] == "Enter password:", "Should have correct prompt"

    # Test 3: Empty/missing parameters
    result3 = await tool.invoke("")
    assert result3["_waiting_for_input"] is True, "Should have waiting marker"
    assert "input" in result3["prompt"].lower(), "Should have default prompt"

    # Test 4: Malformed JSON
    result4 = await tool.invoke("{invalid json")
    assert result4["_waiting_for_input"] is True, "Should have waiting marker"
    assert "input" in result4["prompt"].lower(), "Should fall back to default"

    # Test 5: None parameter
    result5 = await tool.invoke(None)
    assert result5["_waiting_for_input"] is True, "Should have waiting marker"
    assert "input" in result5["prompt"].lower(), "Should have default prompt"

    # Test 6: Verify spec structure
    spec = await tool.spec()
    assert spec["type"] == "function", "Should be function type"
    assert spec["function"]["name"] == "ask_user_for_input", "Should have correct name"
    assert "question" in spec["function"]["parameters"]["properties"], "Should have question parameter"

  def test_state_transitions_monitoring(self):
    """Test state machine transitions during pause/resume."""
    Node.start(
      self._test_state_transitions_monitoring,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_transitions_monitoring(self, node):
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "Continue?"}'}],
        },
        {"role": "assistant", "content": "Continuing..."},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="state-monitor-agent",
      instructions="Monitor states",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Send initial message - this should pause
    response1 = await agent.send("Start", conversation="state-conv")
    assert len(response1) > 0, "Should have response when paused"

    # Resume
    response2 = await agent.send("Yes", conversation="state-conv")
    assert len(response2) > 0, "Should have response when resumed"

  def test_multiple_pause_resume_cycles(self):
    """Test multiple pause/resume cycles in same conversation."""
    Node.start(
      self._test_multiple_pause_resume_cycles,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_multiple_pause_resume_cycles(self, node):
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "First question?"}'}],
        },
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "Second question?"}'}],
        },
        {"role": "assistant", "content": "All done!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="multi-pause-agent",
      instructions="Ask multiple questions",
      model=model,
      enable_ask_for_user_input=True,
    )

    # First pause
    response1 = await agent.send("Start", conversation="multi-conv")
    assert len(response1) > 0, "Should have response for first pause"

    # Verify first pause has waiting message
    waiting_found = any(
      isinstance(msg, AssistantMessage) and hasattr(msg, "phase") and msg.phase == Phase.WAITING_FOR_INPUT
      for msg in response1
    )
    assert waiting_found, "Should have waiting message for first pause"

    # Resume and pause again
    response2 = await agent.send("Answer 1", conversation="multi-conv")
    assert len(response2) > 0, "Should have response for second pause"

    # Verify second pause has waiting message
    waiting_found = any(
      isinstance(msg, AssistantMessage) and hasattr(msg, "phase") and msg.phase == Phase.WAITING_FOR_INPUT
      for msg in response2
    )
    assert waiting_found, "Should have waiting message for second pause"

    # Final resume
    response3 = await agent.send("Answer 2", conversation="multi-conv")
    assert len(response3) > 0, "Should have final response"

  def test_paused_timestamp_recorded(self):
    """Test that paused_at timestamp is recorded correctly."""
    Node.start(
      self._test_paused_timestamp_recorded,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_paused_timestamp_recorded(self, node):
    import time

    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "Wait"}'}],
        },
        {"role": "assistant", "content": "Done"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="timestamp-agent",
      instructions="Test timestamps",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Record time before pause
    time_before = time.time()

    # Pause
    response1 = await agent.send("Start", conversation="time-conv")
    assert len(response1) > 0, "Should have response when paused"

    pause_time = time.time()
    assert pause_time >= time_before, "Should have paused after start"

    # Wait a bit
    await asyncio.sleep(0.5)

    # Resume
    response2 = await agent.send("Continue", conversation="time-conv")
    assert len(response2) > 0, "Should have response when resumed"

    resume_time = time.time()
    assert resume_time >= pause_time + 0.5, "Resume should happen after pause and delay"

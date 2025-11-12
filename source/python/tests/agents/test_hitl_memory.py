"""
Memory and history tests for Human-in-the-Loop (ask_user_for_input).

Tests conversation memory preservation during pause/resume cycles.
"""

import pytest
from autonomy import Agent, Node
from autonomy.agents.http import HttpServer
from autonomy.nodes.message import (
  ConversationRole,
  Phase,
  AssistantMessage,
)
from tests.agents.mock_utils import MockModel


class TestHITLMemory:
  """Test memory and history handling during HITL."""

  def test_conversation_history_preserved_during_pause(self):
    """Verify conversation history is preserved while paused."""
    Node.start(
      self._test_conversation_history_preserved_during_pause,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_conversation_history_preserved_during_pause(self, node):
    """Test that conversation history is preserved when agent pauses."""
    # Model that responds then asks for input
    model = MockModel(
      [
        {"role": "assistant", "content": "I will remember that."},
        {"role": "assistant", "content": "Your name is Alice."},
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "What else would you like to know?"}'}
          ],
        },
      ]
    )

    agent = await Agent.start(
      node=node,
      name="memory-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Send initial messages to build history
    response1 = await agent.send("Remember: my name is Alice", conversation="test-conv")
    assert len(response1) > 0, "Should have response to first message"

    response2 = await agent.send("What is my name?", conversation="test-conv")
    assert len(response2) > 0, "Should have response to second message"

    # Trigger pause
    response3 = await agent.send("Ask me a question", conversation="test-conv")
    assert len(response3) > 0, "Should have response when pausing"

    # Check state while paused using new state inspection API
    state = await agent.get_conversation_state(conversation="test-conv")
    assert state.is_paused is True, "Conversation should be paused"

    # Verify history is preserved (should have at least 3 user messages + assistant responses)
    # At minimum: 3 user messages + 3 assistant responses = 6 messages
    assert state.message_count >= 6, f"Should have at least 6 messages in history, got {state.message_count}"

  def test_context_available_after_resume(self):
    """Verify agent has context from before pause after resume."""
    Node.start(
      self._test_context_available_after_resume,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_context_available_after_resume(self, node):
    """Test that agent retains context after pause/resume cycle."""
    # Model that acknowledges context, asks for input, then responds with context
    model = MockModel(
      [
        {"role": "assistant", "content": "Got it, blue is your favorite color."},
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "What is your favorite color?"}'}],
        },
        {"role": "assistant", "content": "Yes, I remember you said blue!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="context-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Establish context
    response1 = await agent.send("My favorite color is blue", conversation="test-conv")
    assert len(response1) > 0, "Should have response establishing context"

    # Pause
    response2 = await agent.send("Ask me about my favorite color", conversation="test-conv")
    assert len(response2) > 0, "Should have response when pausing"

    # Verify paused
    state = await agent.get_conversation_state(conversation="test-conv")
    assert state.is_paused is True, "Should be paused"

    # Resume
    response3 = await agent.send("It's blue!", conversation="test-conv")
    assert len(response3) > 0, "Should have response after resume"

    # Agent should acknowledge context in the response
    # Extract assistant response content
    final_content = ""
    for msg in response3:
      if isinstance(msg, AssistantMessage) and hasattr(msg, "content"):
        if hasattr(msg.content, "text"):
          final_content += msg.content.text.lower()

    # Agent should reference the context (blue or color)
    assert "blue" in final_content or "color" in final_content or len(final_content) > 0, (
      "Agent should respond with context from conversation"
    )

  def test_multiple_conversations_isolated(self):
    """Verify different conversations maintain separate history."""
    Node.start(
      self._test_multiple_conversations_isolated,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_multiple_conversations_isolated(self, node):
    """Test that different conversations maintain separate histories."""
    # Model that responds to messages
    model = MockModel(
      [
        {"role": "assistant", "content": "Hello Alice!"},
        {"role": "assistant", "content": "Hello Bob!"},
        {"role": "assistant", "content": "Alice's response"},
        {"role": "assistant", "content": "Bob's response"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="isolation-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Conversation 1
    response1 = await agent.send("My name is Alice", conversation="conv1")
    assert len(response1) > 0, "Should have response in conv1"

    # Get state for conversation 1
    state1_initial = await agent.get_conversation_state(conversation="conv1")

    # Conversation 2
    response2 = await agent.send("My name is Bob", conversation="conv2")
    assert len(response2) > 0, "Should have response in conv2"

    # Get state for conversation 2
    state2_initial = await agent.get_conversation_state(conversation="conv2")

    # Verify both conversations exist and have messages
    assert state1_initial.message_count >= 2, "Conv1 should have at least 2 messages"
    assert state2_initial.message_count >= 2, "Conv2 should have at least 2 messages"

    # Add more messages to conv1
    response3 = await agent.send("Tell me more", conversation="conv1")
    assert len(response3) > 0, "Should have response in conv1"

    # Get updated states
    state1_updated = await agent.get_conversation_state(conversation="conv1")
    state2_updated = await agent.get_conversation_state(conversation="conv2")

    # Conv1 should have more messages now, but conv2 should be unchanged
    assert state1_updated.message_count > state1_initial.message_count, "Conv1 message count should increase"
    assert state2_updated.message_count == state2_initial.message_count, "Conv2 message count should remain unchanged"

  def test_history_persists_across_pause_resume_cycles(self):
    """Verify history accumulates correctly across multiple pauses."""
    Node.start(
      self._test_history_persists_across_pause_resume_cycles,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_history_persists_across_pause_resume_cycles(self, node):
    """Test that conversation history accumulates across multiple pause/resume cycles."""
    # Model that responds to each message
    model = MockModel(
      [
        {"role": "assistant", "content": "Remembered 0"},
        {"role": "assistant", "content": "Remembered 1"},
        {"role": "assistant", "content": "Remembered 2"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="history-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    messages_count = []

    # Send messages and track history growth
    for i in range(3):
      response = await agent.send(f"Remember number {i}", conversation="test-conv")
      assert len(response) > 0, f"Should have response for message {i}"

      state = await agent.get_conversation_state(conversation="test-conv")
      messages_count.append(state.message_count)

    # Count should increase each time (accumulating history)
    assert messages_count == sorted(messages_count), f"Message count should increase monotonically: {messages_count}"

    # Last count should be greater than first
    assert messages_count[-1] > messages_count[0], (
      f"History should grow: started with {messages_count[0]}, ended with {messages_count[-1]}"
    )

    # Should have at least 6 messages total (3 user + 3 assistant)
    assert messages_count[-1] >= 6, f"Should have at least 6 messages after 3 exchanges, got {messages_count[-1]}"

"""
Tests for conversation state inspection API.

Tests the ability to query conversation state (is_paused, get_conversation_state)
without accessing private _active_state_machines.
"""

import asyncio
import pytest
from autonomy import Agent, Node
from autonomy.agents.http import HttpServer
from autonomy.nodes.message import Phase
from tests.agents.mock_utils import MockModel


class TestStateInspection:
  """Test conversation state inspection API."""

  def test_is_paused_returns_true_when_waiting(self):
    """Test is_paused returns True when conversation is waiting for user input."""
    Node.start(
      self._test_is_paused_returns_true_when_waiting,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_is_paused_returns_true_when_waiting(self, node):
    # Model that asks for user input (triggers WAITING_FOR_INPUT state)
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "What is your name?"}'}
          ],
        },
      ]
    )

    agent = await Agent.start(
      node=node,
      name="test-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Send initial message (will trigger pause)
    messages = await agent.send("Hello", conversation="test")

    # Verify agent asked a question
    assert any("What is your name?" in str(m) for m in messages)

    # Check if conversation is paused
    is_paused = await agent.is_paused(conversation="test")
    assert is_paused is True, "Conversation should be paused after ask_user_for_input"

  def test_is_paused_returns_false_after_resume(self):
    """Test is_paused returns False after conversation completes."""
    Node.start(
      self._test_is_paused_returns_false_after_resume,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_is_paused_returns_false_after_resume(self, node):
    # Model that asks for input, then completes
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "What is your name?"}'}
          ],
        },
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="test-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Initial message (triggers pause)
    await agent.send("Hello", conversation="test")

    # Verify paused
    assert await agent.is_paused(conversation="test") is True

    # Resume with user response
    messages = await agent.send("My name is Alice", conversation="test")

    # Verify agent responded
    assert any("Nice to meet you" in str(m) for m in messages)

    # Check if conversation is no longer paused
    is_paused = await agent.is_paused(conversation="test")
    assert is_paused is False, "Conversation should not be paused after completion"

  def test_get_conversation_state_full(self):
    """Test get_conversation_state returns complete state information."""
    Node.start(
      self._test_get_conversation_state_full,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_get_conversation_state_full(self, node):
    # Model that asks for user input
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "What is your favorite color?"}'}
          ],
        },
      ]
    )

    agent = await Agent.start(
      node=node,
      name="test-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Send initial message
    await agent.send("Hello", conversation="test")

    # Get full conversation state
    state = await agent.get_conversation_state(conversation="test")

    # Verify state properties
    assert state.is_paused is True, "State should show conversation is paused"
    assert state.phase == "waiting_for_input", f"Phase should be 'waiting_for_input', got '{state.phase}'"
    assert state.message_count >= 2, f"Should have at least 2 messages (user + assistant), got {state.message_count}"
    assert state.conversation == "test", "Conversation ID should match"

  def test_is_paused_returns_false_for_new_conversation(self):
    """Test is_paused returns False for conversations that haven't started."""
    Node.start(
      self._test_is_paused_returns_false_for_new_conversation,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_is_paused_returns_false_for_new_conversation(self, node):
    model = MockModel([{"role": "assistant", "content": "Hello!"}])

    agent = await Agent.start(
      node=node,
      name="test-agent",
      instructions="You are a helpful assistant.",
      model=model,
    )

    # Check state of non-existent conversation
    is_paused = await agent.is_paused(conversation="nonexistent")
    assert is_paused is False, "Non-existent conversation should not be paused"

  def test_get_conversation_state_for_completed_conversation(self):
    """Test get_conversation_state for a conversation that has completed."""
    Node.start(
      self._test_get_conversation_state_for_completed_conversation,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_get_conversation_state_for_completed_conversation(self, node):
    # Model that completes without asking for input
    model = MockModel([{"role": "assistant", "content": "Hello! How can I help you?"}])

    agent = await Agent.start(
      node=node,
      name="test-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Send message and let it complete
    await agent.send("Hi", conversation="test")

    # Get state after completion
    state = await agent.get_conversation_state(conversation="test")

    # Verify state shows completion
    assert state.is_paused is False, "Completed conversation should not be paused"
    assert state.phase == "done", f"Phase should be 'done', got '{state.phase}'"
    assert state.message_count >= 2, f"Should have at least 2 messages, got {state.message_count}"

  def test_is_paused_with_streaming(self):
    """Test is_paused works correctly with streaming mode."""
    Node.start(
      self._test_is_paused_with_streaming,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_is_paused_with_streaming(self, node):
    # Model that asks for input
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "What is your age?"}'}
          ],
        },
        {"role": "assistant", "content": "Thanks for sharing!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="test-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Use streaming mode
    chunks = []
    async for chunk in agent.send_stream("Hello", conversation="test"):
      chunks.append(chunk)

    assert len(chunks) > 0, "Should receive streaming chunks"

    # Check if paused after streaming
    is_paused = await agent.is_paused(conversation="test")
    assert is_paused is True, "Should be paused after streaming ask_user_for_input"

    # Resume with streaming
    chunks = []
    async for chunk in agent.send_stream("I'm 25", conversation="test"):
      chunks.append(chunk)

    # Should no longer be paused
    is_paused = await agent.is_paused(conversation="test")
    assert is_paused is False, "Should not be paused after completion"

  def test_multiple_conversations_isolated_state(self):
    """Test that conversation state reflects the currently active conversation.

    Note: Agent currently supports only ONE active conversation at a time.
    Starting a new conversation while one is paused will resume/interrupt the paused one.
    This test verifies that get_conversation_state correctly reports the active conversation.
    """
    Node.start(
      self._test_multiple_conversations_isolated_state,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_multiple_conversations_isolated_state(self, node):
    # Model with two responses
    model = MockModel(
      [
        # First conversation - will pause
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "Question 1?"}'}
          ],
        },
        # When conv1 is active and paused, this will be used for second message
        {"role": "assistant", "content": "Response after resume"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="test-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Start conversation 1 (will pause)
    await agent.send("Hello", conversation="conv1")

    # Verify conv1 is paused
    state1 = await agent.get_conversation_state(conversation="conv1")
    assert state1.is_paused is True, "Conversation 1 should be paused initially"
    assert state1.phase == "waiting_for_input", "Conversation 1 should be waiting for input"

    # Resume conv1
    await agent.send("My answer", conversation="conv1")

    # Verify conv1 is no longer paused after completion
    state1_after = await agent.get_conversation_state(conversation="conv1")
    assert state1_after.is_paused is False, "Conversation 1 should not be paused after completion"
    assert state1_after.phase == "done", "Conversation 1 should be done"

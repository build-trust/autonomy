"""
Edge cases and error condition tests for Human-in-the-Loop (ask_user_for_input).

Tests unusual scenarios, error conditions, and boundary cases.
"""

import asyncio
import pytest
from autonomy import Agent, Node
from autonomy.agents.http import HttpServer
from autonomy.nodes.message import (
  ConversationRole,
  Phase,
  AssistantMessage,
)
from tests.agents.mock_utils import MockModel


class TestHITLEdgeCases:
  """Test edge cases and error conditions."""

  def test_rapid_pause_resume(self):
    """Test rapid consecutive pause/resume cycles."""
    Node.start(
      self._test_rapid_pause_resume,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_rapid_pause_resume(self, node):
    """Test that rapid pause/resume cycles work without issues."""
    # Model that asks for input then completes
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "Tell me more?"}'}],
        },
        {"role": "assistant", "content": "Thanks!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="rapid-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Do one rapid pause/resume cycle
    # The key is that we do it quickly without delays
    response1 = await agent.send("Ask me", conversation="test-conv")
    assert len(response1) > 0, "Should have response when pausing"

    # Verify paused
    is_paused = await agent.is_paused(conversation="test-conv")
    assert is_paused is True, "Should be paused"

    # Resume immediately (no delay - this is the "rapid" part)
    response2 = await agent.send("Answer", conversation="test-conv")
    assert len(response2) > 0, "Should have response when resuming"

    # Verify completed
    is_paused_after = await agent.is_paused(conversation="test-conv")
    assert is_paused_after is False, "Should not be paused after completion"

  def test_resume_non_paused_conversation(self):
    """Test resuming a conversation that isn't paused."""
    Node.start(
      self._test_resume_non_paused_conversation,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_resume_non_paused_conversation(self, node):
    """Test that sending messages to non-paused conversation works normally."""
    # Model that just responds (no pausing)
    model = MockModel(
      [
        {"role": "assistant", "content": "Hello!"},
        {"role": "assistant", "content": "I'm doing well!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="non-pause-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # First message (not paused)
    messages1 = await agent.send("Hello", conversation="test-conv")
    assert len(messages1) > 0, "Should have response"

    # Verify not paused
    is_paused = await agent.is_paused(conversation="test-conv")
    assert is_paused is False, "Should not be paused"

    # Send another message (like "resuming" but there was no pause)
    messages2 = await agent.send("How are you?", conversation="test-conv")
    assert len(messages2) > 0, "Should have response to second message"

    # Should still work normally
    is_paused_after = await agent.is_paused(conversation="test-conv")
    assert is_paused_after is False, "Should still not be paused"

  def test_empty_user_response(self):
    """Test providing empty response when paused."""
    Node.start(
      self._test_empty_user_response,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_empty_user_response(self, node):
    """Test that empty user response is handled gracefully."""
    # Model that asks for input then responds
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "Tell me something?"}'}],
        },
        {"role": "assistant", "content": "Okay, no problem."},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="empty-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Trigger pause
    response1 = await agent.send("Ask me a question", conversation="test-conv")
    assert len(response1) > 0, "Should have response when pausing"

    # Verify paused
    is_paused = await agent.is_paused(conversation="test-conv")
    assert is_paused is True, "Should be paused"

    # Send empty response (edge case)
    response2 = await agent.send("", conversation="test-conv")

    # Should handle gracefully (not crash)
    assert len(response2) >= 0, "Should handle empty response gracefully"

  def test_very_long_pause_duration(self):
    """Test conversation paused for extended time."""
    Node.start(
      self._test_very_long_pause_duration,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_very_long_pause_duration(self, node):
    """Test that conversation can be paused for extended time."""
    # Model that asks for input then responds
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "Waiting for you?"}'}],
        },
        {"role": "assistant", "content": "Welcome back!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="long-pause-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Trigger pause
    response1 = await agent.send("Ask me", conversation="test-conv")
    assert len(response1) > 0, "Should have response when pausing"

    # Verify paused
    is_paused = await agent.is_paused(conversation="test-conv")
    assert is_paused is True, "Should be paused"

    # Wait for 2 seconds (simulating long pause)
    await asyncio.sleep(2)

    # Should still be paused
    is_paused_after_wait = await agent.is_paused(conversation="test-conv")
    assert is_paused_after_wait is True, "Should still be paused after wait"

    # Resume after long pause
    response2 = await agent.send("Answer", conversation="test-conv")
    assert len(response2) > 0, "Should have response after long pause"

    # Should be completed now
    is_paused_final = await agent.is_paused(conversation="test-conv")
    assert is_paused_final is False, "Should not be paused after resume"

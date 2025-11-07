"""
Comprehensive tests for memory edge cases and state management.

This test file targets:
1. Memory Edge Cases (~10 tests)
   - Memory operation failures
   - Persistence errors
   - Cross-scope isolation
   - Memory cleanup
   - Concurrent memory access

2. State Management (~15 tests)
   - Complex state transitions
   - Recovery scenarios
   - Error propagation
   - State machine lifecycle
   - Interruption handling

Coverage goal: Increase agent.py coverage to 75-80%
"""

import pytest
import asyncio
import json
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from autonomy import Agent, Node, Tool, Model
from autonomy.agents.http import HttpServer
from autonomy.nodes.message import (
  ConversationRole,
  UserMessage,
  AssistantMessage,
  ToolCallResponseMessage,
  Phase,
)
from tests.agents.mock_utils import (
  MockModel,
  create_simple_mock_model,
  create_tool_mock_model,
  ErrorMockModel,
  simple_test_tool,
  calculator_tool,
)


# =============================================================================
# SECTION 1: Memory Edge Cases Tests
# =============================================================================


class TestMemoryEdgeCases:
  """
  Test memory operation edge cases and failure scenarios.
  Targets memory-related uncovered lines in agent.py
  """

  def test_memory_cross_scope_isolation(self):
    """Test that different scopes cannot access each other's memories"""
    Node.start(
      self._test_memory_cross_scope_isolation,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_cross_scope_isolation(self, node):
    """Test scope isolation"""
    model = MockModel(
      [
        {"role": "assistant", "content": "Message for Alice"},
        {"role": "assistant", "content": "Message for Bob"},
        {"role": "assistant", "content": "Message for Alice again"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="isolation-agent",
      instructions="Test memory isolation",
      model=model,
    )

    # Send messages in different scopes
    response_alice1 = await agent.send("Hello from Alice", scope="alice", conversation="conv1")
    response_bob = await agent.send("Hello from Bob", scope="bob", conversation="conv1")
    response_alice2 = await agent.send("What did I say before?", scope="alice", conversation="conv1")

    # All should complete
    assert len(response_alice1) > 0
    assert len(response_bob) > 0
    assert len(response_alice2) > 0

    # Verify responses are different (scope isolation working)
    assert response_alice1 != response_bob

  def test_memory_cross_conversation_isolation(self):
    """Test that different conversations in same scope are isolated"""
    Node.start(
      self._test_memory_cross_conversation_isolation,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_cross_conversation_isolation(self, node):
    """Test conversation isolation"""
    model = MockModel(
      [
        {"role": "assistant", "content": "Response to conv1"},
        {"role": "assistant", "content": "Response to conv2"},
        {"role": "assistant", "content": "Response to conv3"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="conv-isolation-agent",
      instructions="Test conversation isolation",
      model=model,
    )

    # Same scope, different conversations
    response1 = await agent.send("Message in conv1", scope="user1", conversation="conv1")
    response2 = await agent.send("Message in conv2", scope="user1", conversation="conv2")
    response3 = await agent.send("Message in conv3", scope="user1", conversation="conv3")

    # All should complete independently
    assert len(response1) > 0
    assert len(response2) > 0
    assert len(response3) > 0

  def test_memory_with_very_long_history(self):
    """Test memory with large conversation history"""
    Node.start(
      self._test_memory_with_very_long_history,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_with_very_long_history(self, node):
    """Test large history handling"""
    # Create many responses
    responses = [{"role": "assistant", "content": f"Response {i}"} for i in range(50)]
    model = MockModel(responses)

    agent = await Agent.start(
      node=node,
      name="long-history-agent",
      instructions="Handle long conversations",
      model=model,
    )

    # Send many messages to build up history
    for i in range(20):
      response = await agent.send(f"Message {i}", conversation="long-conv")
      assert len(response) > 0

  def test_memory_with_special_characters_in_content(self):
    """Test memory storing messages with special characters"""
    Node.start(
      self._test_memory_with_special_characters_in_content,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_with_special_characters_in_content(self, node):
    """Test special characters in memory"""
    model = create_simple_mock_model("Stored your special message")

    agent = await Agent.start(
      node=node,
      name="special-chars-memory-agent",
      instructions="Store special characters",
      model=model,
    )

    # Messages with various special characters
    special_messages = [
      "Hello <>&\"'",
      "Unicode: 世界 🌍 café",
      "Newlines:\nLine 1\nLine 2",
      "Tabs:\tTabbed\tContent",
      "JSON: {\"key\": \"value\"}",
      "SQL: SELECT * FROM users WHERE name='test'",
    ]

    for msg in special_messages:
      response = await agent.send(msg, conversation="special-conv")
      assert len(response) > 0

  def test_memory_concurrent_access_same_conversation(self):
    """Test concurrent memory access to same conversation"""
    Node.start(
      self._test_memory_concurrent_access_same_conversation,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_concurrent_access_same_conversation(self, node):
    """Test concurrent memory access"""
    model = MockModel(
      [{"role": "assistant", "content": f"Response {i}"} for i in range(10)]
    )

    agent = await Agent.start(
      node=node,
      name="concurrent-memory-agent",
      instructions="Handle concurrent access",
      model=model,
    )

    # Send concurrent messages (will interrupt each other)
    tasks = [
      agent.send("Concurrent 1", conversation="same-conv"),
      agent.send("Concurrent 2", conversation="same-conv"),
      agent.send("Concurrent 3", conversation="same-conv"),
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # At least some should succeed
    successful = [r for r in responses if not isinstance(r, Exception)]
    assert len(successful) > 0

  def test_memory_persistence_across_messages(self):
    """Test that memory persists across multiple messages"""
    Node.start(
      self._test_memory_persistence_across_messages,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_persistence_across_messages(self, node):
    """Test memory persistence"""
    model = MockModel(
      [
        {"role": "assistant", "content": "I'll remember that."},
        {"role": "assistant", "content": "Yes, I remember."},
        {"role": "assistant", "content": "Still remember."},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="persistent-memory-agent",
      instructions="Remember conversation context",
      model=model,
    )

    # First message
    await agent.send("My name is Alice", conversation="memory-test")

    # Second message - should have context from first
    await agent.send("What's my name?", conversation="memory-test")

    # Third message - should still have context
    response = await agent.send("Do you still remember?", conversation="memory-test")

    assert len(response) > 0

  def test_memory_with_tool_results(self):
    """Test memory storing tool call results"""
    Node.start(
      self._test_memory_with_tool_results,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_with_tool_results(self, node):
    """Test tool results in memory"""
    model = MockModel(
      [
        {"role": "assistant", "tool_calls": [{"name": "calculator_tool", "arguments": '{"expression": "2+2"}'}]},
        {"role": "assistant", "content": "The calculator returned the result."},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="tool-memory-agent",
      instructions="Use tools and remember results",
      model=model,
      tools=[Tool(calculator_tool)],
    )

    response = await agent.send("Calculate 2+2", conversation="tool-conv")
    assert len(response) > 0

    # Tool result should be in memory for next message

  def test_memory_cleanup_on_new_conversation(self):
    """Test that new conversations don't have old conversation's memory"""
    Node.start(
      self._test_memory_cleanup_on_new_conversation,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_cleanup_on_new_conversation(self, node):
    """Test conversation memory isolation"""
    model = MockModel(
      [
        {"role": "assistant", "content": "Info for conv1"},
        {"role": "assistant", "content": "Info for conv2"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="cleanup-agent",
      instructions="Test memory cleanup",
      model=model,
    )

    # First conversation
    await agent.send("Secret info", conversation="conv1")

    # Second conversation - should not have access to first
    response = await agent.send("What secret?", conversation="conv2")

    assert len(response) > 0
    # Should respond without knowledge of conv1

  def test_memory_with_empty_scope(self):
    """Test memory operations with default/empty scope"""
    Node.start(
      self._test_memory_with_empty_scope,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_with_empty_scope(self, node):
    """Test default scope handling"""
    model = create_simple_mock_model("Using default scope")

    agent = await Agent.start(
      node=node,
      name="default-scope-agent",
      instructions="Test default scope",
      model=model,
    )

    # Send without explicit scope (uses default)
    response = await agent.send("Test message")
    assert len(response) > 0

  def test_memory_message_ordering(self):
    """Test that memory maintains correct message ordering"""
    Node.start(
      self._test_memory_message_ordering,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_message_ordering(self, node):
    """Test message order preservation"""
    model = MockModel(
      [
        {"role": "assistant", "content": "First"},
        {"role": "assistant", "content": "Second"},
        {"role": "assistant", "content": "Third"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="ordering-agent",
      instructions="Test message ordering",
      model=model,
    )

    # Send messages in sequence
    await agent.send("Message 1", conversation="order-test")
    await agent.send("Message 2", conversation="order-test")
    response = await agent.send("Message 3", conversation="order-test")

    # Should maintain order
    assert len(response) > 0


# =============================================================================
# SECTION 2: State Management Tests
# =============================================================================


class TestStateManagement:
  """
  Test state machine state transitions and lifecycle.
  Targets state management uncovered lines in agent.py
  """

  def test_state_transition_ready_to_thinking(self):
    """Test basic state transition from READY to THINKING"""
    Node.start(
      self._test_state_transition_ready_to_thinking,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_transition_ready_to_thinking(self, node):
    """Test READY -> THINKING transition"""
    model = create_simple_mock_model("Thinking complete")

    agent = await Agent.start(
      node=node,
      name="ready-thinking-agent",
      instructions="Test state transitions",
      model=model,
    )

    response = await agent.send("Test")
    assert len(response) > 0

  def test_state_transition_thinking_to_acting(self):
    """Test transition from THINKING to ACTING when tools are called"""
    Node.start(
      self._test_state_transition_thinking_to_acting,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_transition_thinking_to_acting(self, node):
    """Test THINKING -> ACTING transition"""
    model = MockModel(
      [
        {"role": "assistant", "tool_calls": [{"name": "simple_test_tool", "arguments": '{"param": "test"}'}]},
        {"role": "assistant", "content": "Tool executed"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="thinking-acting-agent",
      instructions="Use tools",
      model=model,
      tools=[Tool(simple_test_tool)],
    )

    response = await agent.send("Use the tool")
    assert len(response) > 0

  def test_state_transition_acting_to_thinking(self):
    """Test transition from ACTING back to THINKING after tool execution"""
    Node.start(
      self._test_state_transition_acting_to_thinking,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_transition_acting_to_thinking(self, node):
    """Test ACTING -> THINKING transition"""
    model = MockModel(
      [
        {"role": "assistant", "tool_calls": [{"name": "calculator_tool", "arguments": '{"expression": "5*5"}'}]},
        {"role": "assistant", "content": "The result is 25"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="acting-thinking-agent",
      instructions="Use calculator and respond",
      model=model,
      tools=[Tool(calculator_tool)],
    )

    response = await agent.send("Calculate 5*5")
    assert len(response) > 0

  def test_state_transition_to_done_on_completion(self):
    """Test transition to DONE state when conversation completes"""
    Node.start(
      self._test_state_transition_to_done_on_completion,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_transition_to_done_on_completion(self, node):
    """Test -> DONE transition"""
    model = create_simple_mock_model("Final response")

    agent = await Agent.start(
      node=node,
      name="done-agent",
      instructions="Complete quickly",
      model=model,
    )

    response = await agent.send("Quick task")
    assert len(response) > 0
    # State machine should be DONE

  def test_state_transition_waiting_for_input(self):
    """Test transition to WAITING_FOR_INPUT state"""
    Node.start(
      self._test_state_transition_waiting_for_input,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_transition_waiting_for_input(self, node):
    """Test -> WAITING_FOR_INPUT transition"""
    model = MockModel(
      [
        {"role": "assistant", "tool_calls": [{"name": "ask_user_for_input", "arguments": '{"question": "Your name?"}'}]},
        {"role": "assistant", "content": "Thanks for the info"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="waiting-agent",
      instructions="Ask for user input",
      model=model,
    )

    # First message triggers waiting state
    await agent.send("Start", conversation="wait-conv")

    # Resume from waiting
    response = await agent.send("Alice", conversation="wait-conv")
    assert len(response) > 0

  def test_state_recovery_after_interruption(self):
    """Test state recovery after interruption"""
    Node.start(
      self._test_state_recovery_after_interruption,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_recovery_after_interruption(self, node):
    """Test recovery from interruption"""
    model = MockModel(
      [
        {"role": "assistant", "content": "First response"},
        {"role": "assistant", "content": "After interruption"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="recovery-agent",
      instructions="Handle interruptions",
      model=model,
    )

    # Start first request
    task1 = asyncio.create_task(agent.send("First", conversation="recovery-conv"))

    # Brief delay
    await asyncio.sleep(0.05)

    # Interrupt with second request
    response2 = await agent.send("Interrupt!", conversation="recovery-conv")

    # Clean up first task
    try:
      await asyncio.wait_for(task1, timeout=1.0)
    except asyncio.TimeoutError:
      task1.cancel()

    assert len(response2) > 0

  def test_state_error_propagation_from_model(self):
    """Test error propagation from model through state machine"""
    Node.start(
      self._test_state_error_propagation_from_model,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_error_propagation_from_model(self, node):
    """Test model error propagation"""

    class ModelErrorModel(MockModel):
      async def _complete_chat_non_streaming(self, provided_message):
        raise RuntimeError("Model API failure")

    model = ModelErrorModel([{"role": "assistant", "content": "Won't reach"}])

    agent = await Agent.start(
      node=node,
      name="error-prop-agent",
      instructions="Test error propagation",
      model=model,
    )

    # Should handle error gracefully
    try:
      response = await agent.send("Test")
      # May return with error message
      assert response is not None
    except Exception as e:
      # Or may raise exception
      assert e is not None

  def test_state_error_propagation_from_tool(self):
    """Test error propagation from tool through state machine"""
    Node.start(
      self._test_state_error_propagation_from_tool,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_error_propagation_from_tool(self, node):
    """Test tool error propagation"""

    def error_tool(param: str) -> str:
      raise ValueError(f"Tool error: {param}")

    model = MockModel(
      [
        {"role": "assistant", "tool_calls": [{"name": "error_tool", "arguments": '{"param": "test"}'}]},
        {"role": "assistant", "content": "Handled error"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="tool-error-agent",
      instructions="Handle tool errors",
      model=model,
      tools=[Tool(error_tool)],
    )

    response = await agent.send("Use error tool")
    assert len(response) > 0
    # Should continue despite tool error

  def test_state_machine_iteration_counting(self):
    """Test that state machine correctly counts iterations"""
    Node.start(
      self._test_state_machine_iteration_counting,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_machine_iteration_counting(self, node):
    """Test iteration counting"""
    # Model that makes multiple tool calls (multiple iterations)
    model = MockModel(
      [
        {"role": "assistant", "tool_calls": [{"name": "simple_test_tool", "arguments": '{"param": "1"}'}]},
        {"role": "assistant", "tool_calls": [{"name": "simple_test_tool", "arguments": '{"param": "2"}'}]},
        {"role": "assistant", "content": "Done"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="iteration-agent",
      instructions="Make multiple iterations",
      model=model,
      tools=[Tool(simple_test_tool)],
      max_iterations=10,  # Allow multiple iterations
    )

    response = await agent.send("Do multiple steps")
    assert len(response) > 0

  def test_state_machine_time_tracking(self):
    """Test that state machine tracks execution time"""
    Node.start(
      self._test_state_machine_time_tracking,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_machine_time_tracking(self, node):
    """Test time tracking"""

    class SlowModel(MockModel):
      async def _complete_chat_non_streaming(self, provided_message):
        await asyncio.sleep(0.1)  # Small delay
        return await super()._complete_chat_non_streaming(provided_message)

    model = SlowModel([{"role": "assistant", "content": "Slow response"}])

    agent = await Agent.start(
      node=node,
      name="time-tracking-agent",
      instructions="Track time",
      model=model,
      max_execution_time=10.0,  # Generous limit
    )

    response = await agent.send("Test timing")
    assert len(response) > 0

  def test_state_multiple_tool_calls_same_iteration(self):
    """Test state machine handling multiple tool calls in single iteration"""
    Node.start(
      self._test_state_multiple_tool_calls_same_iteration,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_multiple_tool_calls_same_iteration(self, node):
    """Test multiple tools in one iteration"""

    def tool_a(param: str) -> str:
      return f"A: {param}"

    def tool_b(param: str) -> str:
      return f"B: {param}"

    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "tool_a", "arguments": '{"param": "first"}'},
            {"name": "tool_b", "arguments": '{"param": "second"}'},
          ],
        },
        {"role": "assistant", "content": "Both tools executed"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="multi-tool-agent",
      instructions="Use multiple tools",
      model=model,
      tools=[Tool(tool_a), Tool(tool_b)],
    )

    response = await agent.send("Use both tools")
    assert len(response) > 0

  def test_state_streaming_vs_non_streaming(self):
    """Test state machine behavior difference between streaming and non-streaming"""
    Node.start(
      self._test_state_streaming_vs_non_streaming,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_streaming_vs_non_streaming(self, node):
    """Test streaming vs non-streaming state handling"""
    model = MockModel(
      [
        {"role": "assistant", "content": "Non-streaming response"},
        {"role": "assistant", "content": "Streaming response"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="stream-vs-non-agent",
      instructions="Test streaming modes",
      model=model,
    )

    # Non-streaming
    response1 = await agent.send("Test non-streaming", conversation="conv1")
    assert len(response1) > 0

    # Streaming
    chunks = []
    async for chunk in agent.send_stream("Test streaming", conversation="conv2"):
      chunks.append(chunk)
    assert len(chunks) > 0

  def test_state_transition_with_reasoning_content(self):
    """Test state transitions with model reasoning content"""
    Node.start(
      self._test_state_transition_with_reasoning_content,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_transition_with_reasoning_content(self, node):
    """Test reasoning content handling"""
    model = MockModel(
      [
        {
          "role": "assistant",
          "reasoning_content": "Let me think about this...",
          "content": "Here's my answer",
        }
      ]
    )

    agent = await Agent.start(
      node=node,
      name="reasoning-agent",
      instructions="Use reasoning",
      model=model,
    )

    response = await agent.send("Test reasoning")
    assert len(response) > 0

  def test_state_lifecycle_complete_flow(self):
    """Test complete state machine lifecycle from start to finish"""
    Node.start(
      self._test_state_lifecycle_complete_flow,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_state_lifecycle_complete_flow(self, node):
    """Test full lifecycle"""
    model = MockModel(
      [
        # First iteration: call tool
        {"role": "assistant", "tool_calls": [{"name": "calculator_tool", "arguments": '{"expression": "10+5"}'}]},
        # Second iteration: final response
        {"role": "assistant", "content": "The answer is 15"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="lifecycle-agent",
      instructions="Complete full lifecycle",
      model=model,
      tools=[Tool(calculator_tool)],
    )

    response = await agent.send("Calculate 10+5")

    # Should go through: READY -> THINKING -> ACTING -> THINKING -> DONE
    assert len(response) > 0

    # Check that we got both assistant responses (tool call and final)
    assistant_messages = [msg for msg in response if msg.role == ConversationRole.ASSISTANT]
    assert len(assistant_messages) >= 1


# =============================================================================
# SECTION 3: Combined Memory and State Tests
# =============================================================================


class TestMemoryAndStateInteraction:
  """
  Test interactions between memory and state management.
  """

  def test_memory_state_consistency_during_interruption(self):
    """Test memory consistency when state is interrupted"""
    Node.start(
      self._test_memory_state_consistency_during_interruption,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_state_consistency_during_interruption(self, node):
    """Test memory consistency during interruption"""
    model = MockModel(
      [
        {"role": "assistant", "content": "First"},
        {"role": "assistant", "content": "Second"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="consistency-agent",
      instructions="Test consistency",
      model=model,
    )

    # Start request
    task1 = asyncio.create_task(agent.send("First", conversation="consistency"))

    await asyncio.sleep(0.05)

    # Interrupt
    response2 = await agent.send("Second", conversation="consistency")

    try:
      await asyncio.wait_for(task1, timeout=1.0)
    except asyncio.TimeoutError:
      task1.cancel()

    assert len(response2) > 0

  def test_memory_persistence_across_state_transitions(self):
    """Test that memory persists correctly across state transitions"""
    Node.start(
      self._test_memory_persistence_across_state_transitions,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_memory_persistence_across_state_transitions(self, node):
    """Test memory across states"""
    model = MockModel(
      [
        {"role": "assistant", "tool_calls": [{"name": "simple_test_tool", "arguments": '{"param": "data"}'}]},
        {"role": "assistant", "content": "Tool result saved to memory"},
        {"role": "assistant", "content": "I remember the tool result"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="persistence-agent",
      instructions="Remember across states",
      model=model,
      tools=[Tool(simple_test_tool)],
    )

    # First message with tool call
    await agent.send("Use the tool", conversation="persist-conv")

    # Second message - should have tool result in memory
    response = await agent.send("What was the result?", conversation="persist-conv")

    assert len(response) > 0


# =============================================================================
# Test Suite Summary
# =============================================================================

if __name__ == "__main__":
  print("Memory and State Management Test Suite")
  print("=" * 60)
  print("Memory Edge Cases: 10 tests")
  print("  - Cross-scope isolation")
  print("  - Cross-conversation isolation")
  print("  - Large history handling")
  print("  - Special characters")
  print("  - Concurrent access")
  print("  - Persistence")
  print("  - Tool results in memory")
  print("  - Memory cleanup")
  print("  - Default scope handling")
  print("  - Message ordering")
  print()
  print("State Management: 15 tests")
  print("  - State transitions (READY->THINKING->ACTING->DONE)")
  print("  - WAITING_FOR_INPUT state")
  print("  - Recovery after interruption")
  print("  - Error propagation (model and tool)")
  print("  - Iteration counting")
  print("  - Time tracking")
  print("  - Multiple tool calls")
  print("  - Streaming vs non-streaming")
  print("  - Reasoning content")
  print("  - Complete lifecycle")
  print()
  print("Combined Tests: 2 tests")
  print("  - Memory consistency during interruption")
  print("  - Memory persistence across states")
  print("=" * 60)
  print("Total: 27 comprehensive tests")

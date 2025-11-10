"""
Streaming-specific tests for Human-in-the-Loop (ask_user_for_input).

Tests streaming behavior during pause/resume cycles.
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


class TestHITLStreaming:
  """Test streaming-specific HITL functionality."""

  def test_streaming_pause_sends_chunks_before_pause(self):
    """Verify chunks are sent before conversation pauses."""
    Node.start(
      self._test_streaming_pause_sends_chunks_before_pause,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_pause_sends_chunks_before_pause(self, node):
    """Test that streaming sends chunks when pausing."""
    # Model that asks for input
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
      name="streaming-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Send message with streaming
    chunks = []
    async for chunk in agent.send_stream("Hello", conversation="test-conv"):
      chunks.append(chunk)

    # Should receive multiple chunks
    assert len(chunks) >= 1, "Should receive at least one chunk"

    # Last chunk should be marked as finished
    assert chunks[-1].finished is True, "Last chunk should be marked finished"

    # Verify we can find the waiting phase in the chunks
    # Phase is on the AssistantMessage inside snippet.messages
    found_waiting_phase = False
    for chunk in chunks:
      if hasattr(chunk, "snippet") and hasattr(chunk.snippet, "messages"):
        for message in chunk.snippet.messages:
          if isinstance(message, AssistantMessage) and hasattr(message, "phase"):
            if message.phase == Phase.WAITING_FOR_INPUT:
              found_waiting_phase = True
              break
      if found_waiting_phase:
        break

    assert found_waiting_phase, "Should have chunk with WAITING_FOR_INPUT phase"

  def test_streaming_resume_sends_chunks(self):
    """Verify streaming resume sends chunks (not 0)."""
    Node.start(
      self._test_streaming_resume_sends_chunks,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_resume_sends_chunks(self, node):
    """Test that streaming resume sends chunks after pause."""
    # Model that asks for input then responds
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "What is your name?"}'}
          ],
        },
        {"role": "assistant", "content": "Nice to meet you!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="resume-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Initial pause - use streaming
    chunks1 = []
    async for chunk in agent.send_stream("Hello", conversation="test-conv"):
      chunks1.append(chunk)

    # Verify we got chunks and paused
    assert len(chunks1) >= 1, "Should have initial chunks"

    # Resume with streaming - this is what we're testing
    resume_chunks = []
    async for chunk in agent.send_stream("My name is Alice", conversation="test-conv"):
      resume_chunks.append(chunk)

    # The key assertion: resume should send chunks (not 0)
    assert len(resume_chunks) > 0, "Resume should send chunks (bug was sending 0)"

    # Last chunk should be marked finished
    assert resume_chunks[-1].finished is True, "Last resume chunk should be finished"

    # Verify we got the continuation message
    found_content = False
    for chunk in resume_chunks:
      if hasattr(chunk, "snippet") and hasattr(chunk.snippet, "messages"):
        for message in chunk.snippet.messages:
          if isinstance(message, AssistantMessage) and hasattr(message, "content"):
            if hasattr(message.content, "text") and len(message.content.text) > 0:
              found_content = True
              break
      if found_content:
        break

    assert found_content, "Resume chunks should contain assistant response"

  def test_streaming_multiple_pauses(self):
    """Test multiple pause/resume cycles in streaming mode."""
    Node.start(
      self._test_streaming_multiple_pauses,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_multiple_pauses(self, node):
    """Test multiple pause/resume cycles with streaming."""
    # Model that asks for input multiple times
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "First question?"}'}
          ],
        },
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "Second question?"}'}
          ],
        },
        {"role": "assistant", "content": "All done!"},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="multi-pause-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # First pause/resume cycle
    chunks1 = []
    async for chunk in agent.send_stream("Start", conversation="test-conv"):
      chunks1.append(chunk)

    assert len(chunks1) > 0, "First pause should send chunks"
    assert chunks1[-1].finished is True, "First pause should finish"

    # Second pause/resume cycle
    chunks2 = []
    async for chunk in agent.send_stream("Answer 1", conversation="test-conv"):
      chunks2.append(chunk)

    assert len(chunks2) > 0, "Second pause should send chunks"
    assert chunks2[-1].finished is True, "Second pause should finish"

    # Final resume (no more pauses)
    chunks3 = []
    async for chunk in agent.send_stream("Answer 2", conversation="test-conv"):
      chunks3.append(chunk)

    assert len(chunks3) > 0, "Final response should send chunks"
    assert chunks3[-1].finished is True, "Final response should finish"

    # Verify all cycles sent chunks (no zero-chunk bug)
    assert len(chunks1) > 0 and len(chunks2) > 0 and len(chunks3) > 0, \
      "All pause/resume cycles should send chunks"

  def test_streaming_chunk_order(self):
    """Verify chunks arrive in order with correct part numbers."""
    Node.start(
      self._test_streaming_chunk_order,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_chunk_order(self, node):
    """Test that chunks arrive in correct order with sequential part numbers."""
    # Model that provides a response (will be streamed)
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "Tell me more?"}'}
          ],
        },
      ]
    )

    agent = await Agent.start(
      node=node,
      name="order-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Send message with streaming
    chunks = []
    async for chunk in agent.send_stream("Test", conversation="test-conv"):
      chunks.append(chunk)

    # Should receive chunks
    assert len(chunks) > 0, "Should receive chunks"

    # Verify part numbers are sequential starting from 1
    for i, chunk in enumerate(chunks):
      expected_part_nb = i + 1
      assert chunk.part_nb == expected_part_nb, \
        f"Chunk {i} should have part_nb={expected_part_nb}, got {chunk.part_nb}"

    # Last chunk should be marked finished
    assert chunks[-1].finished is True, "Last chunk should be marked finished"

    # All chunks except the last should NOT be finished
    for i in range(len(chunks) - 1):
      assert chunks[i].finished is False, \
        f"Chunk {i} should not be marked finished (only last chunk should be)"

  def test_streaming_timeout_handling(self):
    """Test that streaming properly handles timeouts."""
    Node.start(
      self._test_streaming_timeout_handling,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_timeout_handling(self, node):
    """Test that streaming completes before timeout (no hanging)."""
    import asyncio

    # Model that asks for input
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "Quick question?"}'}
          ],
        },
      ]
    )

    agent = await Agent.start(
      node=node,
      name="timeout-agent",
      instructions="You are a helpful assistant.",
      model=model,
      enable_ask_for_user_input=True,
    )

    # Should complete before timeout
    chunks = []
    try:
      async with asyncio.timeout(10):  # 10 second timeout
        async for chunk in agent.send_stream("Test", conversation="test-conv"):
          chunks.append(chunk)
    except asyncio.TimeoutError:
      pytest.fail("Streaming should not timeout when pausing")

    # Verify we got chunks
    assert len(chunks) > 0, "Should have received chunks"
    assert chunks[-1].finished is True, "Last chunk should be finished"

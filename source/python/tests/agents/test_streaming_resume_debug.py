"""
Debug test for streaming resume issue.

This minimal test isolates the streaming resume bug where the agent
processes the resume successfully but sends zero chunks to the client.
"""

import asyncio
import pytest
from autonomy import Agent, Node
from autonomy.agents.http import HttpServer
from autonomy.nodes.message import Phase, AssistantMessage
from tests.agents.mock_utils import MockModel


class TestStreamingResumeDebug:
  """Minimal tests to debug streaming resume."""

  def test_streaming_resume_minimal(self):
    """Minimal test to debug streaming resume - should receive chunks."""
    Node.start(
      self._test_streaming_resume_minimal,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_resume_minimal(self, node):
    """Minimal test to debug streaming resume - should receive chunks."""

    # Set up agent with mock model
    # First response: asks for input (pause)
    # Second response: continues after receiving input
    model = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "What is your favorite color?"}'}
          ],
        },
        {"role": "assistant", "content": "Great choice! Blue is a nice color."},
      ]
    )

    agent = await Agent.start(
      node=node,
      name="debug-agent",
      instructions="You are a test assistant. When asked to collect info, use ask_user_for_input.",
      model=model,
    )

    print("\n" + "="*60)
    print("PHASE 1: Initial request (should pause)")
    print("="*60)

    # Initial request - should work and pause
    chunks1 = []
    async for chunk in agent.send_stream("Ask me a question", conversation="test"):
      chunks1.append(chunk)
      print(f"INITIAL CHUNK {len(chunks1)}: part_nb={chunk.part_nb}, finished={chunk.finished}")
      if hasattr(chunk, 'snippet') and hasattr(chunk.snippet, 'phase'):
        print(f"  Phase: {chunk.snippet.phase}")

    print(f"\nINITIAL TOTAL: Received {len(chunks1)} chunks")
    assert len(chunks1) > 0, "Initial request should receive chunks"

    # Verify it paused - check for waiting_for_input phase in messages
    last_chunk = chunks1[-1]
    assert last_chunk.finished is True, "Last chunk should be finished"

    # Check if any message has waiting_for_input phase
    waiting_found = any(
      isinstance(msg, AssistantMessage) and hasattr(msg, "phase") and msg.phase == Phase.WAITING_FOR_INPUT
      for msg in last_chunk.snippet.messages
    )
    assert waiting_found, "Should be waiting for input"

    print("\n" + "="*60)
    print("PHASE 2: Resume request (CURRENTLY BROKEN)")
    print("="*60)

    # Resume - currently broken (receives 0 chunks, times out)
    chunks2 = []

    try:
      # Add timeout to catch the issue
      async with asyncio.timeout(15):
        async for chunk in agent.send_stream("My answer is blue", conversation="test"):
          chunks2.append(chunk)
          print(f"RESUME CHUNK {len(chunks2)}: part_nb={chunk.part_nb}, finished={chunk.finished}")
          if hasattr(chunk, 'snippet'):
            print(f"  Messages: {len(chunk.snippet.messages)}")
    except asyncio.TimeoutError:
      print(f"\n⚠️  TIMEOUT! Received {len(chunks2)} chunks before timeout")

    print(f"\nRESUME TOTAL: Received {len(chunks2)} chunks")

    # This is the bug - we expect chunks but get 0
    if len(chunks2) > 0:
      assert chunks2[-1].finished is True, "Last resume chunk should be finished"
      print("✅ Streaming resume works!")
    else:
      print("❌ BUG: Streaming resume sends 0 chunks")
      # Still assert to fail the test
      assert len(chunks2) > 0, f"Resume should receive chunks, got {len(chunks2)}"

  def test_streaming_vs_nonstreaming_resume(self):
    """Compare streaming vs non-streaming resume to understand difference."""
    Node.start(
      self._test_streaming_vs_nonstreaming_resume,
      wait_until_interrupted=False,
      http_server=HttpServer(listen_address="127.0.0.1:0"),
    )

  async def _test_streaming_vs_nonstreaming_resume(self, node):
    """Compare streaming vs non-streaming resume to understand difference."""

    print("\n" + "="*60)
    print("TEST A: Streaming initial, Non-streaming resume (WORKAROUND)")
    print("="*60)

    model1 = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "Tell me something?"}'}
          ],
        },
        {"role": "assistant", "content": "Thanks for the info!"},
      ]
    )

    agent1 = await Agent.start(
      node=node,
      name="debug-agent-1",
      instructions="You are a test assistant.",
      model=model1,
    )

    # Initial with streaming
    chunks = []
    async for chunk in agent1.send_stream("Ask me something", conversation="conv1"):
      chunks.append(chunk)

    print(f"Initial: {len(chunks)} chunks")
    assert len(chunks) > 0

    # Resume with NON-streaming (this works)
    messages = await agent1.send("My response", conversation="conv1")
    print(f"Resume (non-streaming): {len(messages)} messages")
    assert len(messages) > 0
    print("✅ Non-streaming resume works!")

    print("\n" + "="*60)
    print("TEST B: Streaming initial, Streaming resume (BROKEN)")
    print("="*60)

    model2 = MockModel(
      [
        {
          "role": "assistant",
          "tool_calls": [
            {"name": "ask_user_for_input", "arguments": '{"question": "Tell me more?"}'}
          ],
        },
        {"role": "assistant", "content": "Got it, thanks!"},
      ]
    )

    agent2 = await Agent.start(
      node=node,
      name="debug-agent-2",
      instructions="You are a test assistant.",
      model=model2,
    )

    # Initial with streaming
    chunks1 = []
    async for chunk in agent2.send_stream("Ask me something else", conversation="conv2"):
      chunks1.append(chunk)

    print(f"Initial: {len(chunks1)} chunks")
    assert len(chunks1) > 0

    # Resume with streaming (this is broken)
    chunks2 = []
    try:
      async with asyncio.timeout(15):
        async for chunk in agent2.send_stream("Another response", conversation="conv2"):
          chunks2.append(chunk)
    except asyncio.TimeoutError:
      print(f"⚠️  TIMEOUT on streaming resume!")

    print(f"Resume (streaming): {len(chunks2)} chunks")

    if len(chunks2) == 0:
      print("❌ BUG CONFIRMED: Streaming resume sends 0 chunks")
    else:
      print("✅ Streaming resume works!")

    # This assertion will fail, showing the bug
    assert len(chunks2) > 0, "Streaming resume should work like non-streaming resume"

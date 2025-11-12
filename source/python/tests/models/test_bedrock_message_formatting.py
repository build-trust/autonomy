"""
Tests for Bedrock client message formatting, particularly tool result consolidation.

This module tests the fix for the issue where consecutive tool messages need to be
consolidated into a single user message with multiple tool_result blocks for the
Anthropic/Claude API format.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from autonomy.models.clients.bedrock_client import BedrockClient


class TestBedrockMessageFormatting:
  """Test Bedrock client message formatting and consolidation."""

  def setup_method(self):
    """Set up test fixtures."""
    # Mock boto3 client
    self.mock_bedrock = MagicMock()
    self.mock_session = MagicMock()
    self.mock_session.client.return_value = self.mock_bedrock

  def test_single_tool_result_conversion(self):
    """Test that a single tool result is converted to user message format."""
    with patch("boto3.Session", return_value=self.mock_session):
      client = BedrockClient("anthropic.claude-3-5-sonnet-20241022-v2:0")

      messages = [
        {"role": "user", "content": "What's the weather?"},
        {
          "role": "assistant",
          "content": "Let me check.",
          "tool_calls": [
            {
              "id": "tool_123",
              "type": "function",
              "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
          ],
        },
        {"role": "tool", "tool_call_id": "tool_123", "name": "get_weather", "content": "Sunny, 22°C"},
      ]

      payload = client._prepare_bedrock_payload(messages)

      # Check that the tool result was converted to user message with tool_result block
      formatted = payload["messages"]
      assert len(formatted) == 3

      # First message should be user
      assert formatted[0]["role"] == "user"
      assert formatted[0]["content"][0]["type"] == "text"

      # Second message should be assistant with tool_use
      assert formatted[1]["role"] == "assistant"
      assert formatted[1]["content"][1]["type"] == "tool_use"
      assert formatted[1]["content"][1]["id"] == "tool_123"

      # Third message should be user with tool_result
      assert formatted[2]["role"] == "user"
      assert formatted[2]["content"][0]["type"] == "tool_result"
      assert formatted[2]["content"][0]["tool_use_id"] == "tool_123"

  def test_consecutive_tool_results_consolidation(self):
    """Test that consecutive tool results are consolidated into a single user message."""
    with patch("boto3.Session", return_value=self.mock_session):
      client = BedrockClient("anthropic.claude-3-5-sonnet-20241022-v2:0")

      messages = [
        {"role": "user", "content": "Compare weather in Tokyo and London"},
        {
          "role": "assistant",
          "content": "I'll check both cities.",
          "tool_calls": [
            {
              "id": "tool_123",
              "type": "function",
              "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            },
            {
              "id": "tool_456",
              "type": "function",
              "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            },
          ],
        },
        {"role": "tool", "tool_call_id": "tool_123", "name": "get_weather", "content": "Sunny, 22°C"},
        {"role": "tool", "tool_call_id": "tool_456", "name": "get_weather", "content": "Rainy, 15°C"},
      ]

      payload = client._prepare_bedrock_payload(messages)

      # Check that both tool results were consolidated into a single user message
      formatted = payload["messages"]
      assert len(formatted) == 3

      # Third message should be user with TWO tool_result blocks
      assert formatted[2]["role"] == "user"
      assert len(formatted[2]["content"]) == 2
      assert formatted[2]["content"][0]["type"] == "tool_result"
      assert formatted[2]["content"][0]["tool_use_id"] == "tool_123"
      assert formatted[2]["content"][1]["type"] == "tool_result"
      assert formatted[2]["content"][1]["tool_use_id"] == "tool_456"

  def test_tool_results_after_user_message(self):
    """Test that tool results after a user message are merged into that user message."""
    with patch("boto3.Session", return_value=self.mock_session):
      client = BedrockClient("anthropic.claude-3-5-sonnet-20241022-v2:0")

      messages = [
        {"role": "user", "content": "First question"},
        {
          "role": "assistant",
          "content": "Let me check.",
          "tool_calls": [
            {
              "id": "tool_123",
              "type": "function",
              "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
          ],
        },
        {"role": "tool", "tool_call_id": "tool_123", "name": "get_weather", "content": "Sunny, 22°C"},
        {"role": "assistant", "content": "The weather is sunny."},
        {"role": "user", "content": "Second question"},
        # This tool result should be merged with the user message above
        {"role": "tool", "tool_call_id": "tool_456", "name": "get_time", "content": "15:30 JST"},
      ]

      payload = client._prepare_bedrock_payload(messages)
      formatted = payload["messages"]

      # Find the last user message - it should contain both text and tool_result
      last_user_msg = None
      for msg in reversed(formatted):
        if msg["role"] == "user":
          last_user_msg = msg
          break

      assert last_user_msg is not None
      assert len(last_user_msg["content"]) == 2
      # First should be text from "Second question"
      assert last_user_msg["content"][0]["type"] == "text"
      assert "Second question" in last_user_msg["content"][0]["text"]
      # Second should be the tool result
      assert last_user_msg["content"][1]["type"] == "tool_result"
      assert last_user_msg["content"][1]["tool_use_id"] == "tool_456"

  def test_multi_turn_conversation_with_tools(self):
    """
    Test a multi-turn conversation to ensure tool results don't reference
    tool_use_ids from previous turns.

    This test specifically addresses the bug where tool results in later turns
    would reference tool_use_ids that are no longer in the immediately preceding
    assistant message.
    """
    with patch("boto3.Session", return_value=self.mock_session):
      client = BedrockClient("anthropic.claude-3-5-sonnet-20241022-v2:0")

      messages = [
        # Turn 1
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
          "role": "assistant",
          "content": "Checking weather...",
          "tool_calls": [
            {
              "id": "tool_turn1",
              "type": "function",
              "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
          ],
        },
        {"role": "tool", "tool_call_id": "tool_turn1", "name": "get_weather", "content": "Sunny, 22°C"},
        {"role": "assistant", "content": "It's sunny in Tokyo."},
        # Turn 2
        {"role": "user", "content": "What time is it there?"},
        {
          "role": "assistant",
          "content": "Checking time...",
          "tool_calls": [
            {
              "id": "tool_turn2",
              "type": "function",
              "function": {"name": "get_time", "arguments": '{"city": "Tokyo"}'},
            }
          ],
        },
        {"role": "tool", "tool_call_id": "tool_turn2", "name": "get_time", "content": "15:30 JST"},
      ]

      payload = client._prepare_bedrock_payload(messages)
      formatted = payload["messages"]

      # Verify structure: should alternate user/assistant properly
      # and tool results should be in user messages
      roles = [msg["role"] for msg in formatted]

      # Find all tool_result blocks and their tool_use_ids
      tool_results = []
      for msg in formatted:
        if msg["role"] == "user" and isinstance(msg["content"], list):
          for content_block in msg["content"]:
            if isinstance(content_block, dict) and content_block.get("type") == "tool_result":
              tool_results.append(content_block["tool_use_id"])

      # Find all tool_use blocks and their ids
      tool_uses = []
      for msg in formatted:
        if msg["role"] == "assistant" and isinstance(msg["content"], list):
          for content_block in msg["content"]:
            if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
              tool_uses.append(content_block["id"])

      # Each tool_result should reference a tool_use that exists
      for tool_result_id in tool_results:
        assert tool_result_id in tool_uses, f"Tool result {tool_result_id} has no corresponding tool use"

      # Specifically check that turn 2's tool result references turn 2's tool use
      assert "tool_turn2" in tool_results
      assert "tool_turn2" in tool_uses

  def test_user_message_content_types(self):
    """Test that different user message content types are handled correctly."""
    with patch("boto3.Session", return_value=self.mock_session):
      client = BedrockClient("anthropic.claude-3-5-sonnet-20241022-v2:0")

      # Test string content
      messages = [{"role": "user", "content": "Hello"}]
      payload = client._prepare_bedrock_payload(messages)
      assert payload["messages"][0]["content"][0]["type"] == "text"
      assert payload["messages"][0]["content"][0]["text"] == "Hello"

      # Test dict content with type=text
      messages = [{"role": "user", "content": {"type": "text", "text": "Hello"}}]
      payload = client._prepare_bedrock_payload(messages)
      assert payload["messages"][0]["content"][0]["type"] == "text"
      assert payload["messages"][0]["content"][0]["text"] == "Hello"

      # Test list content
      messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
      payload = client._prepare_bedrock_payload(messages)
      assert payload["messages"][0]["content"][0]["type"] == "text"
      assert payload["messages"][0]["content"][0]["text"] == "Hello"

  def test_assistant_message_with_multiple_tool_calls(self):
    """Test that assistant messages with multiple tool calls are formatted correctly."""
    with patch("boto3.Session", return_value=self.mock_session):
      client = BedrockClient("anthropic.claude-3-5-sonnet-20241022-v2:0")

      messages = [
        {"role": "user", "content": "Compare Tokyo and London"},
        {
          "role": "assistant",
          "content": "I'll check both.",
          "tool_calls": [
            {
              "id": "tool_1",
              "type": "function",
              "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            },
            {
              "id": "tool_2",
              "type": "function",
              "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            },
          ],
        },
      ]

      payload = client._prepare_bedrock_payload(messages)
      formatted = payload["messages"]

      # Assistant message should have text + 2 tool_use blocks
      assistant_msg = formatted[1]
      assert assistant_msg["role"] == "assistant"
      assert len(assistant_msg["content"]) == 3  # text + 2 tool_use blocks
      assert assistant_msg["content"][0]["type"] == "text"
      assert assistant_msg["content"][1]["type"] == "tool_use"
      assert assistant_msg["content"][1]["id"] == "tool_1"
      assert assistant_msg["content"][2]["type"] == "tool_use"
      assert assistant_msg["content"][2]["id"] == "tool_2"

  def test_system_message_extraction(self):
    """Test that system messages are extracted to the 'system' field."""
    with patch("boto3.Session", return_value=self.mock_session):
      client = BedrockClient("anthropic.claude-3-5-sonnet-20241022-v2:0")

      messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
      ]

      payload = client._prepare_bedrock_payload(messages)

      # System message should be in 'system' field, not in messages array
      assert "system" in payload
      assert payload["system"] == "You are a helpful assistant."
      assert len(payload["messages"]) == 1
      assert payload["messages"][0]["role"] == "user"

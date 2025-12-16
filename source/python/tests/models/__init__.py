"""
Test suite for the autonomy.models module.

This package contains comprehensive tests for:
- Model class and its functionality
- LiteLLMClient implementation
- GatewayClient implementation (OpenAI SDK)
- AnthropicGatewayClient implementation (Anthropic SDK)
- DefaultModelClient fallback implementation
- Model constants and configuration
- Message normalization and utilities
- Provider integrations and configurations

Test Structure:
- test_model.py: Tests for the main Model class
- test_litellm_client.py: Tests for LiteLLM client implementation
- test_gateway_client.py: Tests for gateway client implementations
- test_default_client.py: Tests for the default/fallback client
- test_models_constants.py: Tests for model constants and mappings

Usage:
    Run all model tests:
    pytest autonomy/source/python/tests/models/

    Run specific test file:
    pytest autonomy/source/python/tests/models/test_model.py

    Run with coverage:
    pytest autonomy/source/python/tests/models/ --cov=autonomy.models

Test Categories:
- Unit tests for individual components
- Integration tests for component interactions
- Edge case and error handling tests
- Configuration and constant validation tests
- Provider-specific functionality tests

Dependencies:
- pytest: Main testing framework
- pytest-asyncio: For async/await test support
- unittest.mock: For mocking external dependencies
- autonomy.models: The module being tested
- autonomy.nodes.message: For message type testing

Note: Some tests may require environment variables or AWS credentials
to be set for full provider testing. Mock objects are used extensively
to avoid external dependencies in most tests.
"""

# Test utilities and common fixtures can be added here if needed
import pytest
from unittest.mock import MagicMock


# Common test fixtures that might be used across multiple test files
@pytest.fixture
def mock_boto3_client():
  """Mock boto3 client for AWS tests."""
  mock_client = MagicMock()
  mock_client.get_caller_identity.return_value = {"Account": "123456789012"}
  return mock_client


@pytest.fixture
def sample_messages():
  """Sample messages for testing."""
  return [
    {"role": "user", "content": "Hello, how can you help me today?"},
    {"role": "assistant", "content": "I'm here to help! What do you need assistance with?"},
    {"role": "user", "content": "Can you explain machine learning?"},
  ]


@pytest.fixture
def sample_conversation_messages():
  """Sample ConversationMessage objects for testing."""
  from autonomy.nodes.message import UserMessage, AssistantMessage, SystemMessage

  return [
    SystemMessage(content="You are a helpful assistant."),
    UserMessage(content="Hello, how can you help me today?"),
    AssistantMessage(content="I'm here to help! What do you need assistance with?"),
  ]


@pytest.fixture
def sample_tools():
  """Sample tool definitions for testing."""
  return [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}},
          "required": ["location"],
        },
      },
    },
    {
      "type": "function",
      "function": {
        "name": "calculate",
        "description": "Perform basic arithmetic calculations",
        "parameters": {
          "type": "object",
          "properties": {"expression": {"type": "string", "description": "Mathematical expression to evaluate"}},
          "required": ["expression"],
        },
      },
    },
  ]


# Common test utilities
def assert_message_structure(message_dict):
  """Assert that a message dictionary has the required structure."""
  assert isinstance(message_dict, dict)
  assert "role" in message_dict
  assert "content" in message_dict
  assert isinstance(message_dict["role"], str)
  assert message_dict["role"] in ["system", "user", "assistant", "tool"]


def assert_tool_call_structure(tool_call):
  """Assert that a tool call has the required structure."""
  assert isinstance(tool_call, dict)
  assert "id" in tool_call
  assert "type" in tool_call
  assert "function" in tool_call
  assert tool_call["type"] == "function"

  function = tool_call["function"]
  assert isinstance(function, dict)
  assert "name" in function
  assert "arguments" in function


def assert_response_structure(response, streaming=False):
  """Assert that a response has the expected structure."""
  assert hasattr(response, "choices")
  assert len(response.choices) >= 1

  choice = response.choices[0]

  if streaming:
    assert hasattr(choice, "delta")
    assert hasattr(choice.delta, "content")
  else:
    assert hasattr(choice, "message")
    assert hasattr(choice.message, "role")
    assert hasattr(choice.message, "content")
    assert choice.message.role in ["assistant"]


def mock_litellm_response(content="Test response", tool_calls=None, finish_reason="stop"):
  """Create a mock LiteLLM response."""
  mock_response = MagicMock()
  mock_response.choices = [MagicMock()]
  mock_response.choices[0].message.content = content
  mock_response.choices[0].message.role = "assistant"
  mock_response.choices[0].message.tool_calls = tool_calls
  mock_response.choices[0].finish_reason = finish_reason
  return mock_response


def mock_gateway_response(content="Test response", tool_calls=None, finish_reason="stop"):
  """Create a mock gateway response (OpenAI format)."""
  mock_response = MagicMock()
  mock_response.choices = [MagicMock()]
  mock_response.choices[0].message.content = content
  mock_response.choices[0].message.role = "assistant"
  mock_response.choices[0].message.tool_calls = tool_calls
  mock_response.choices[0].finish_reason = finish_reason
  return mock_response


# Export commonly used test utilities
__all__ = [
  "assert_message_structure",
  "assert_tool_call_structure",
  "assert_response_structure",
  "mock_litellm_response",
  "mock_gateway_response",
]

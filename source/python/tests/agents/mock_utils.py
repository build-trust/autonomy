"""
Shared utilities for testing Autonomy agents.

This module provides consistent mock implementations that work with the
agent system's async interface requirements.
"""

import asyncio
from copy import deepcopy
from typing import List, Dict, Any, Optional
from unittest.mock import Mock


class MockModel:
  """
  Mock model implementation that properly supports the agent system's async interface.

  This mock model correctly implements both streaming and non-streaming responses
  with proper async/await support.

  Usage:
      # Simple response
      model = MockModel([{"role": "assistant", "content": "Hello!"}])

      # Tool call response
      model = MockModel([{
          "role": "assistant",
          "tool_calls": [{"name": "calculator", "arguments": '{"x": 5, "y": 3}'}]
      }])

      # Multiple responses for conversation
      model = MockModel([
          {"role": "assistant", "content": "First response"},
          {"role": "assistant", "content": "Second response"}
      ])
  """

  def __init__(self, messages: List[Dict[str, Any]] = None):
    """
    Initialize the mock model with predefined messages.

    Args:
        messages: List of message dictionaries to return in sequence
    """
    self.provided_messages = messages or []
    self.tool_call_counter = 0
    self.original_name = "MockModel"
    self.name = "MockModel"
    self.call_count = 0

  def complete_chat(self, messages, stream: bool = False, **kwargs):
    """
    Main completion method that routes to streaming or non-streaming handlers.

    This method is not async itself but returns async generators/coroutines
    that can be properly awaited by the agent system.
    """
    self.call_count += 1

    if not self.provided_messages:
      return self._default_response(stream)

    provided_message = self.provided_messages.pop(0)
    if stream:
      return self._complete_chat_streaming(provided_message)
    else:
      return self._complete_chat_non_streaming(provided_message)

  async def _complete_chat_streaming(self, provided_message: Dict[str, Any]):
    """
    Async generator for streaming responses.

    Yields mock streaming chunks that simulate real LLM streaming behavior.
    """
    delta = Mock()
    delta.role = provided_message.get("role", "assistant")
    delta.content = ""

    # Stream reasoning content first (thinking)
    reasoning_content = provided_message.get("reasoning_content", "")
    if reasoning_content:
      for character in reasoning_content:
        delta.reasoning_content = str(character)
        delta.content = None
        delta.tool_calls = []
        yield Mock(choices=[Mock(delta=deepcopy(delta), finish_reason=None)])

    # Stream regular content
    content = provided_message.get("content", "")
    if content:
      for character in content:
        delta.content = str(character)
        delta.reasoning_content = None
        delta.tool_calls = []
        yield Mock(choices=[Mock(delta=deepcopy(delta), finish_reason=None)])

    # Stream tool calls
    tool_calls = provided_message.get("tool_calls", [])
    if tool_calls:
      delta.content = ""
      for provided_tool_call in tool_calls:
        function = Mock()
        function.name = provided_tool_call["name"]
        function.arguments = ""

        tool_call = Mock()
        tool_call.type = "function"
        tool_call.id = f"tool_call_{self.tool_call_counter}"
        tool_call.index = str(self.tool_call_counter)
        self.tool_call_counter += 1
        tool_call.function = function
        delta.tool_calls = [tool_call]

        yield Mock(choices=[Mock(delta=deepcopy(delta), finish_reason=None)])

        # Reset for argument streaming
        function.name = None
        tool_call.id = None
        tool_call.type = None

        # Stream tool arguments
        arguments = provided_tool_call.get("arguments", "")
        for character in arguments:
          function.arguments = str(character)
          yield Mock(choices=[Mock(delta=deepcopy(delta), finish_reason=None)])

    # Final termination chunk
    delta.content = None
    delta.reasoning_content = None
    delta.tool_calls = []
    yield Mock(choices=[Mock(delta=deepcopy(delta), finish_reason="stop")])

  async def _complete_chat_non_streaming(self, provided_message: Dict[str, Any]):
    """
    Async method for non-streaming responses.

    Returns a complete mock response that simulates a full LLM completion.
    """
    message = Mock()
    message.content = provided_message.get("content", "")
    message.reasoning_content = provided_message.get("reasoning_content", "")
    message.role = provided_message.get("role", "assistant")
    message.tool_calls = []

    # Add tool calls if present
    for tool_call_data in provided_message.get("tool_calls", []):
      function = Mock()
      function.name = tool_call_data["name"]
      function.arguments = tool_call_data["arguments"]

      tool_call = Mock()
      tool_call.id = f"tool_call_{self.tool_call_counter}"
      tool_call.function = function
      tool_call.type = "function"

      message.tool_calls.append(tool_call)
      self.tool_call_counter += 1

    choice = Mock()
    choice.message = message

    return Mock(choices=[choice])

  def _default_response(self, stream: bool = False):
    """
    Default response when no messages are provided.
    """
    default_message = {"role": "assistant", "content": "Default MockModel response"}
    if stream:
      return self._complete_chat_streaming(default_message)
    else:
      return self._complete_chat_non_streaming(default_message)

  async def embeddings(self, text, **kwargs):
    """
    Mock embeddings method for testing memory/knowledge functionality.

    Returns dummy embeddings vectors for any input text.
    """
    if isinstance(text, str):
      text = [text]

    # Return simple dummy embeddings
    return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in text]

  def reset(self):
    """Reset the model state for reuse."""
    self.call_count = 0
    self.tool_call_counter = 0

  def add_messages(self, messages: List[Dict[str, Any]]):
    """Add more messages to the queue."""
    self.provided_messages.extend(messages)


class SlowMockModel(MockModel):
  """
  Mock model that simulates slow responses for performance testing.
  """

  def __init__(self, messages: List[Dict[str, Any]] = None, delay: float = 0.5):
    super().__init__(messages)
    self.delay = delay

  async def _complete_chat_non_streaming(self, provided_message: Dict[str, Any]):
    """Add delay to simulate slow model response."""
    await asyncio.sleep(self.delay)
    return await super()._complete_chat_non_streaming(provided_message)

  async def _complete_chat_streaming(self, provided_message: Dict[str, Any]):
    """Add delays between streaming chunks."""
    async for chunk in super()._complete_chat_streaming(provided_message):
      await asyncio.sleep(self.delay / 10)  # Shorter delays between chunks
      yield chunk


class ErrorMockModel(MockModel):
  """
  Mock model that can simulate various error conditions.
  """

  def __init__(
    self, messages: List[Dict[str, Any]] = None, fail_after: int = 0, error_message: str = "Mock model error"
  ):
    super().__init__(messages)
    self.fail_after = fail_after
    self.error_message = error_message

  def complete_chat(self, messages, stream: bool = False, **kwargs):
    """Override to inject errors after specified number of calls."""
    if self.call_count >= self.fail_after > 0:
      raise RuntimeError(self.error_message)

    return super().complete_chat(messages, stream, **kwargs)


class ToolCallMockModel(MockModel):
  """
  Specialized mock model for testing tool integration.

  Automatically generates tool call responses based on available tools.
  """

  def __init__(self, tool_names: List[str] = None, final_response: str = "Task completed"):
    self.tool_names = tool_names or []
    self.final_response = final_response
    super().__init__(self._generate_tool_messages())

  def _generate_tool_messages(self):
    """Generate tool call messages for all provided tools."""
    messages = []

    # Add tool call message for each tool
    for tool_name in self.tool_names:
      messages.append(
        {
          "role": "assistant",
          "tool_calls": [{"name": tool_name, "arguments": f'{{"param": "test_value_for_{tool_name}"}}'}],
        }
      )

    # Add final response
    messages.append({"role": "assistant", "content": self.final_response})

    return messages


def create_simple_mock_model(response: str = "Test response") -> MockModel:
  """Convenience function to create a simple mock model with one response."""
  return MockModel([{"role": "assistant", "content": response}])


def create_tool_mock_model(
  tool_name: str, arguments: str = "{}", final_response: str = "Tool executed successfully"
) -> MockModel:
  """Convenience function to create a mock model that calls a tool."""
  return MockModel(
    [
      {"role": "assistant", "tool_calls": [{"name": tool_name, "arguments": arguments}]},
      {"role": "assistant", "content": final_response},
    ]
  )


def create_conversation_mock_model(responses: List[str]) -> MockModel:
  """Convenience function to create a mock model with multiple responses."""
  messages = [{"role": "assistant", "content": response} for response in responses]
  return MockModel(messages)


# Mock tools for testing
def simple_test_tool(param: str) -> str:
  """Simple test tool that echoes the parameter."""
  return f"Tool executed with param: {param}"


def calculator_tool(expression: str) -> str:
  """Safe calculator tool for testing."""
  try:
    # Basic safety check
    allowed_chars = set("0123456789+-*/().")
    if not all(c in allowed_chars or c.isspace() for c in expression):
      return "Error: Invalid characters in expression"

    result = eval(expression)
    return str(result)
  except Exception as e:
    return f"Error: {str(e)}"


def async_test_tool(param: str) -> str:
  """Async test tool for testing async tool integration."""
  return f"Async tool result: {param}"


def error_test_tool(param: str) -> str:
  """Test tool that always raises an error."""
  raise ValueError(f"Test error with param: {param}")


def complex_test_tool(name: str, age: int, metadata: Optional[dict] = None) -> dict:
  """Test tool with complex parameters and return type."""
  return {"name": name, "age": age, "metadata": metadata or {}, "processed": True}


def add_numbers(a: int, b: int) -> int:
  """Add two numbers together."""
  return a + b


def multiply_numbers(x: float, y: float) -> float:
  """Multiply two numbers together."""
  return x * y


def subtract_numbers(a: int, b: int) -> int:
  """Subtract second number from first number."""
  return a - b


def divide_numbers(x: float, y: float) -> float:
  """Divide first number by second number."""
  if y == 0:
    raise ValueError("Cannot divide by zero")
  return x / y

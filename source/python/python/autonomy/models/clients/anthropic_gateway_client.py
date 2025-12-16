"""
Anthropic SDK gateway client for native Claude access.

This client uses the official Anthropic Python SDK to connect to the
Autonomy External APIs Gateway's /v1/messages endpoint. This provides
native Anthropic API experience for Claude models.

Use this client when you want:
- Native Anthropic SDK types and responses
- Direct access to Claude-specific features
- Streaming with Anthropic event types
"""

import json
from typing import List, Optional
from copy import deepcopy

import tiktoken
from anthropic import AsyncAnthropic

from ...logs import get_logger, InfoContext, DebugContext
from ...nodes.message import ConversationMessage
from ...transcripts import log_raw_request, log_raw_response
from .gateway_config import get_gateway_url, get_gateway_api_key, get_client_metadata_headers


def normalize_messages_for_anthropic(
  messages: List[dict] | List[ConversationMessage],
  is_thinking: bool = False,
) -> tuple[Optional[str], List[dict]]:
  """
  Normalize messages to the format expected by the Anthropic API.

  Anthropic API expects:
  - System message as a separate parameter (not in messages list)
  - Messages alternating between user and assistant roles
  - No system role in the messages list

  :param messages: List of messages (either dicts or ConversationMessage objects)
  :param is_thinking: Whether the model is in thinking mode
  :return: Tuple of (system_message, normalized_messages)
  """
  system_message = None
  normalized = []

  for message in messages:
    if isinstance(message, dict):
      msg = deepcopy(message)
    else:
      # Convert ConversationMessage to dict
      content = getattr(message, "content", None)
      # Handle TextContent objects - extract the text string
      if hasattr(content, "text"):
        content = content.text
      msg = {
        "role": message.role.value if hasattr(message.role, "value") else str(message.role),
        "content": content,
      }

      # Handle tool use
      if hasattr(message, "tool_calls") and message.tool_calls:
        # Convert OpenAI tool_calls format to Anthropic tool_use format
        content = []
        if msg.get("content"):
          content.append({"type": "text", "text": msg["content"]})
        for tc in message.tool_calls:
          # Handle both dict and object formats for tool calls
          if isinstance(tc, dict):
            tc_id = tc.get("id")
            func = tc.get("function", {})
            tc_name = func.get("name") if isinstance(func, dict) else getattr(func, "name", None)
            tc_args = func.get("arguments", {}) if isinstance(func, dict) else getattr(func, "arguments", {})
          else:
            tc_id = getattr(tc, "id", None)
            func = getattr(tc, "function", None)
            tc_name = getattr(func, "name", None) if func else None
            tc_args = getattr(func, "arguments", {}) if func else {}

          # Parse arguments if they're a JSON string
          if isinstance(tc_args, str):
            try:
              tc_args = json.loads(tc_args)
            except json.JSONDecodeError:
              # If it's not valid JSON, keep as string (will be wrapped in content)
              tc_args = {"input": tc_args}

          content.append(
            {
              "type": "tool_use",
              "id": tc_id,
              "name": tc_name,
              "input": tc_args,
            }
          )
        msg["content"] = content

      # Handle tool results
      if hasattr(message, "tool_call_id") and message.tool_call_id:
        msg["content"] = [
          {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id,
            "content": msg.get("content", ""),
          }
        ]

    # Extract system message
    if msg.get("role") == "system":
      system_message = msg.get("content", "")
      continue

    # Convert tool role to user role with tool_result content
    if msg.get("role") == "tool":
      msg["role"] = "user"
      tool_call_id = msg.pop("tool_call_id", None)
      if tool_call_id:
        msg["content"] = [
          {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": msg.get("content", ""),
          }
        ]

    normalized.append(msg)

  return system_message, normalized


class AnthropicGatewayClient(InfoContext, DebugContext):
  """
  Model client that uses the Anthropic SDK to connect to the Gateway's /v1/messages.

  This provides native Anthropic API experience for Claude models while still
  routing through the gateway for:
  - Load balancing across multiple Anthropic/Bedrock accounts
  - API key management
  - Usage tracking
  - Model aliasing
  """

  def __init__(self, name: str, max_input_tokens: Optional[int] = None, **kwargs):
    """
    Initialize the Anthropic gateway client.

    :param name: Model name (e.g., 'claude-3-5-sonnet-20241022')
    :param max_input_tokens: Maximum input tokens for the model
    :param kwargs: Additional parameters
    """
    self.logger = get_logger("model")
    self.original_name = name
    self.name = name
    self.max_input_tokens = max_input_tokens
    self.kwargs = kwargs

    # Get gateway configuration
    gateway_url = get_gateway_url()
    api_key = get_gateway_api_key()

    # Get client metadata headers for AWS usage tracking (Phase 8a)
    self.client_metadata = get_client_metadata_headers()

    # Initialize Anthropic client pointing to gateway
    # Note: Anthropic SDK adds /v1/messages to base_url, so we don't add /v1 here
    self.client = AsyncAnthropic(
      api_key=api_key,
      base_url=gateway_url,
      timeout=120.0,
      default_headers=self.client_metadata if self.client_metadata else None,
    )

    # Initialize tokenizer for token counting
    try:
      self._encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
      self._encoding = None
      self.logger.warning("Failed to load tiktoken encoding, token counting may be inaccurate")

    # Default max_tokens for Anthropic (required parameter)
    self._default_max_tokens = kwargs.get("max_tokens", 4096)

  def count_tokens(
    self, messages: List[dict] | List[ConversationMessage], is_thinking: bool = False, tools=None
  ) -> int:
    """
    Count tokens in the given messages.

    :param messages: Messages to count tokens for
    :param is_thinking: Whether in thinking mode
    :param tools: Optional tools to include in count
    :return: Estimated token count
    """
    if self._encoding is None:
      # Fallback to rough character-based estimation
      total_chars = 0
      for msg in messages:
        if isinstance(msg, dict):
          content = msg.get("content", "")
        else:
          content = getattr(msg, "content", "") or ""
        if isinstance(content, str):
          total_chars += len(content)
      return total_chars // 4

    # Use tiktoken for counting
    total_tokens = 0

    for msg in messages:
      if isinstance(msg, dict):
        content = msg.get("content", "")
      else:
        content = getattr(msg, "content", "") or ""

      # Count message overhead
      total_tokens += 4

      # Count content tokens
      if isinstance(content, str):
        total_tokens += len(self._encoding.encode(content))
      elif isinstance(content, list):
        for part in content:
          if isinstance(part, dict):
            if part.get("type") == "text":
              total_tokens += len(self._encoding.encode(part.get("text", "")))
            elif part.get("type") == "tool_use":
              import json

              total_tokens += len(self._encoding.encode(json.dumps(part.get("input", {}))))

    # Add tokens for tools if provided
    if tools:
      import json

      tools_str = json.dumps(tools)
      total_tokens += len(self._encoding.encode(tools_str))

    return total_tokens

  def support_tools(self) -> bool:
    """Check if the model supports tools/function calling."""
    return True

  def support_forced_assistant_answer(self) -> bool:
    """Check if the model supports forced assistant answers."""
    return True

  def prepare_llm_call(self, messages: List[dict] | List[ConversationMessage], is_thinking: bool, **kwargs):
    """
    Prepare messages and kwargs for the Anthropic API call.

    :param messages: Messages to prepare
    :param is_thinking: Whether in thinking mode
    :param kwargs: Additional parameters
    :return: Tuple of (system_message, normalized_messages, kwargs)
    """
    system, normalized = normalize_messages_for_anthropic(messages, is_thinking)

    # Merge default kwargs with provided kwargs
    merged_kwargs = {**self.kwargs, **kwargs}

    # Ensure max_tokens is set (required by Anthropic)
    if "max_tokens" not in merged_kwargs:
      merged_kwargs["max_tokens"] = self._default_max_tokens

    # Convert OpenAI-style tools to Anthropic format if needed
    if "tools" in merged_kwargs and merged_kwargs["tools"]:
      anthropic_tools = []
      for tool in merged_kwargs["tools"]:
        if tool.get("type") == "function":
          func = tool.get("function", {})
          anthropic_tools.append(
            {
              "name": func.get("name"),
              "description": func.get("description", ""),
              "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
          )
        else:
          # Already in Anthropic format
          anthropic_tools.append(tool)
      merged_kwargs["tools"] = anthropic_tools

    # Remove empty tools list
    if "tools" in merged_kwargs and len(merged_kwargs.get("tools", [])) == 0:
      del merged_kwargs["tools"]

    return system, normalized, merged_kwargs

  def complete_chat(
    self,
    messages: List[dict] | List[ConversationMessage],
    stream: bool = False,
    is_thinking: bool = False,
    agent_name: Optional[str] = None,
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
    **kwargs,
  ):
    """
    Send a messages request through the gateway using Anthropic SDK.

    :param messages: Messages to send
    :param stream: Whether to stream the response
    :param is_thinking: Whether in thinking mode
    :param agent_name: Optional agent name for logging
    :param scope: Optional scope for logging
    :param conversation: Optional conversation ID for logging
    :param kwargs: Additional parameters
    :return: Response (async generator for streaming, awaitable for non-streaming)
    """
    self.logger.info(f"Processing {len(messages)} messages with model '{self.original_name}'")
    self.logger.debug(f"Sending messages to gateway (Anthropic): {messages} (stream={stream})")

    system, messages, kwargs = self.prepare_llm_call(messages, is_thinking, **kwargs)

    if stream:
      return self._complete_chat_stream(
        system,
        messages,
        is_thinking,
        agent_name=agent_name,
        scope=scope,
        conversation=conversation,
        **kwargs,
      )
    else:

      async def _complete():
        return await self._complete_chat(
          system,
          messages,
          agent_name=agent_name,
          scope=scope,
          conversation=conversation,
          **kwargs,
        )

      return _complete()

  async def _complete_chat_stream(
    self,
    system: Optional[str],
    messages: List[dict],
    is_thinking: bool,
    agent_name: Optional[str] = None,
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
    **kwargs,
  ):
    """
    Stream messages completion from the gateway.

    :param system: System message
    :param messages: Prepared messages
    :param is_thinking: Whether in thinking mode
    :param agent_name: Optional agent name for logging
    :param scope: Optional scope for logging
    :param conversation: Optional conversation ID for logging
    :param kwargs: Additional parameters
    :yields: Response chunks (converted to OpenAI-like format for compatibility)
    """
    # Log raw request
    request_payload = {"model": self.name, "messages": messages, "stream": True, **kwargs}
    if system:
      request_payload["system"] = system

    log_raw_request(
      payload=request_payload,
      model_name=self.name,
      agent_name=agent_name,
      scope=scope,
      conversation=conversation,
    )

    # Build request kwargs
    request_kwargs = {"model": self.name, "messages": messages, **kwargs}
    if system:
      request_kwargs["system"] = system

    # Make streaming request
    async with self.client.messages.stream(**request_kwargs) as stream:
      accumulated_content = ""
      accumulated_tool_calls = []

      async for event in stream:
        # Convert Anthropic events to OpenAI-like format for compatibility
        chunk = self._convert_anthropic_event_to_openai(event, is_thinking)
        if chunk:
          # Track content
          if chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content

          # Track tool calls
          if hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
            for tc in chunk.choices[0].delta.tool_calls:
              accumulated_tool_calls.append(tc)

          yield chunk

      # Log accumulated response
      response_dict = {
        "choices": [
          {
            "message": {
              "role": "assistant",
              "content": accumulated_content if accumulated_content else None,
            },
            "finish_reason": "stop",
          }
        ]
      }
      if accumulated_tool_calls:
        response_dict["choices"][0]["message"]["tool_calls"] = accumulated_tool_calls

      log_raw_response(
        response=response_dict,
        model_name=self.name,
        agent_name=agent_name,
        scope=scope,
        conversation=conversation,
      )

  def _convert_anthropic_event_to_openai(self, event, is_thinking: bool):
    """
    Convert an Anthropic streaming event to OpenAI-like format.

    This provides compatibility with code expecting OpenAI response format.
    """

    # Create a mock response object that mimics OpenAI format
    class MockDelta:
      def __init__(self):
        self.content = None
        self.reasoning_content = None
        self.tool_calls = None
        self.role = None

    class MockChoice:
      def __init__(self):
        self.delta = MockDelta()
        self.finish_reason = None
        self.index = 0

    class MockChunk:
      def __init__(self):
        self.choices = [MockChoice()]
        self.id = None
        self.model = None

    chunk = MockChunk()

    # Handle different event types
    event_type = getattr(event, "type", None) if hasattr(event, "type") else type(event).__name__

    if event_type == "content_block_delta" or "ContentBlockDelta" in str(type(event)):
      delta = getattr(event, "delta", event)
      delta_type = getattr(delta, "type", None)

      if delta_type == "text_delta" or hasattr(delta, "text"):
        text = getattr(delta, "text", "")
        if is_thinking:
          chunk.choices[0].delta.reasoning_content = text
        else:
          chunk.choices[0].delta.content = text
        return chunk

      elif delta_type == "input_json_delta" or hasattr(delta, "partial_json"):
        # Tool input streaming - would need more context to handle properly
        pass

    elif event_type == "message_start" or "MessageStart" in str(type(event)):
      message = getattr(event, "message", None)
      if message:
        chunk.id = getattr(message, "id", None)
        chunk.model = getattr(message, "model", None)
        chunk.choices[0].delta.role = "assistant"
      return chunk

    elif event_type == "message_delta" or "MessageDelta" in str(type(event)):
      delta = getattr(event, "delta", event)
      stop_reason = getattr(delta, "stop_reason", None)
      if stop_reason:
        # Map Anthropic stop reasons to OpenAI
        reason_map = {
          "end_turn": "stop",
          "stop_sequence": "stop",
          "max_tokens": "length",
          "tool_use": "tool_calls",
        }
        chunk.choices[0].finish_reason = reason_map.get(stop_reason, stop_reason)
        return chunk

    elif event_type == "message_stop" or "MessageStop" in str(type(event)):
      chunk.choices[0].finish_reason = "stop"
      return chunk

    return None

  async def _complete_chat(
    self,
    system: Optional[str],
    messages: List[dict],
    agent_name: Optional[str] = None,
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
    **kwargs,
  ):
    """
    Non-streaming messages completion from the gateway.

    :param system: System message
    :param messages: Prepared messages
    :param agent_name: Optional agent name for logging
    :param scope: Optional scope for logging
    :param conversation: Optional conversation ID for logging
    :param kwargs: Additional parameters
    :return: Completion response (converted to OpenAI-like format)
    """
    # Log raw request
    request_payload = {"model": self.name, "messages": messages, "stream": False, **kwargs}
    if system:
      request_payload["system"] = system

    log_raw_request(
      payload=request_payload,
      model_name=self.name,
      agent_name=agent_name,
      scope=scope,
      conversation=conversation,
    )

    # Build request kwargs
    request_kwargs = {"model": self.name, "messages": messages, **kwargs}
    if system:
      request_kwargs["system"] = system

    # Make request
    response = await self.client.messages.create(**request_kwargs)

    # Log response
    log_raw_response(
      response=response.model_dump() if hasattr(response, "model_dump") else response,
      model_name=self.name,
      agent_name=agent_name,
      scope=scope,
      conversation=conversation,
    )

    # Convert Anthropic response to OpenAI-like format for compatibility
    return self._convert_anthropic_response_to_openai(response)

  def _convert_anthropic_response_to_openai(self, response):
    """
    Convert an Anthropic response to OpenAI-like format.

    This provides compatibility with code expecting OpenAI response format.
    """

    class MockMessage:
      def __init__(self):
        self.role = "assistant"
        self.content = None
        self.reasoning_content = None
        self.tool_calls = None

    class MockChoice:
      def __init__(self):
        self.message = MockMessage()
        self.finish_reason = None
        self.index = 0

    class MockUsage:
      def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    class MockResponse:
      def __init__(self):
        self.id = None
        self.model = None
        self.choices = [MockChoice()]
        self.usage = MockUsage()

    result = MockResponse()
    result.id = response.id
    result.model = response.model

    # Extract content
    content_parts = []
    tool_calls = []

    for block in response.content:
      if block.type == "text":
        content_parts.append(block.text)
      elif block.type == "tool_use":
        # Serialize arguments as proper JSON, not Python str() representation
        if isinstance(block.input, str):
          arguments = block.input
        else:
          arguments = json.dumps(block.input)
        tool_calls.append(
          {
            "id": block.id,
            "type": "function",
            "function": {
              "name": block.name,
              "arguments": arguments,
            },
          }
        )

    result.choices[0].message.content = "\n".join(content_parts) if content_parts else None
    if tool_calls:
      result.choices[0].message.tool_calls = tool_calls

    # Map stop reason
    reason_map = {
      "end_turn": "stop",
      "stop_sequence": "stop",
      "max_tokens": "length",
      "tool_use": "tool_calls",
    }
    result.choices[0].finish_reason = reason_map.get(response.stop_reason, response.stop_reason)

    # Set usage
    if response.usage:
      result.usage.prompt_tokens = response.usage.input_tokens
      result.usage.completion_tokens = response.usage.output_tokens
      result.usage.total_tokens = response.usage.input_tokens + response.usage.output_tokens

    return result

  async def embeddings(self, text: List[str], **kwargs) -> List[List[float]]:
    """
    Generate embeddings.

    Note: Anthropic doesn't have a native embeddings API, so this would need
    to use the gateway's OpenAI-compatible embeddings endpoint instead.
    """
    raise NotImplementedError("Anthropic doesn't have native embeddings. Use GatewayClient for embeddings.")

  async def text_to_speech(self, text: str, voice: str = "alloy", response_format: str = "mp3", **kwargs) -> bytes:
    """
    Text-to-speech is not supported by Anthropic.
    """
    raise NotImplementedError("Anthropic doesn't support text-to-speech. Use GatewayClient for TTS.")

  async def speech_to_text(self, audio_file, language: Optional[str] = None, **kwargs) -> str:
    """
    Speech-to-text is not supported by Anthropic.
    """
    raise NotImplementedError("Anthropic doesn't support speech-to-text. Use GatewayClient for STT.")

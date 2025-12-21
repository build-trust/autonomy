"""
Anthropic SDK gateway client for native Claude access.

This client uses the official Anthropic Python SDK to connect to the
Autonomy External APIs Gateway's /v1/messages endpoint. This provides
native Anthropic API experience for Claude models.

Use this client when you want:
- Native Anthropic SDK types and responses
- Direct access to Claude-specific features
- Streaming with Anthropic event types

Throttling:
When throttle=True is passed (via Model constructor), requests are routed through a
per-model queue with adaptive rate limiting (AIMD algorithm). This helps
prevent rate limit errors when making many concurrent requests.

Timeout Configuration:
Configurable timeouts for different scenarios:
- request_timeout: Default timeout for LLM calls (default: 120s)
- connect_timeout: Connection establishment timeout (default: 10s)
- stream_timeout: Timeout for streaming responses (default: 300s)
"""

import json
from typing import List, Optional
from copy import deepcopy

import tiktoken
from anthropic import AsyncAnthropic

from ...logs import get_logger, InfoContext, DebugContext
from ...nodes.message import ConversationMessage
from ...transcripts import log_raw_request, log_raw_response
import httpx

from .gateway_config import (
  get_gateway_url,
  get_gateway_api_key,
  get_client_metadata_headers,
)
from .rate_limiter import RateLimiterConfig
from .request_queue import QueueConfig, QueueManager, RetryConfig


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

    :param name: Model name (e.g., 'claude-sonnet-4')
    :param max_input_tokens: Maximum input tokens for the model
    :param kwargs: Additional parameters including:
      - request_timeout: Default timeout for LLM calls (default: 120.0s)
      - connect_timeout: Connection establishment timeout (default: 10.0s)
      - stream_timeout: Timeout for streaming responses (default: 300.0s)
      - throttle: Enable throttling with rate limiting and queuing (default: False)
      - throttle_requests_per_minute: Initial RPM for rate limiter (default: 60.0)
      - throttle_max_requests_in_progress: Max concurrent requests (default: 10)
      - throttle_max_requests_waiting_in_queue: Max queue depth (default: 1000)
      - throttle_max_seconds_to_wait_in_queue: Queue timeout (default: 300.0s)
      - throttle_max_retry_attempts: Max retries on rate limit (default: 3)
      - throttle_initial_seconds_between_retry_attempts: Initial backoff (default: 5.0s)
      - throttle_max_seconds_between_retry_attempts: Max backoff (default: 60.0s)
    """
    self.logger = get_logger("model")
    self.original_name = name
    self.name = name
    self.max_input_tokens = max_input_tokens

    # Extract timeout configuration
    self._request_timeout = kwargs.pop("request_timeout", 120.0)
    self._connect_timeout = kwargs.pop("connect_timeout", 10.0)
    self._stream_timeout = kwargs.pop("stream_timeout", 300.0)

    # Extract throttle configuration
    self._throttle = kwargs.pop("throttle", False)
    self._throttle_requests_per_minute = kwargs.pop("throttle_requests_per_minute", 60.0)
    self._throttle_max_requests_in_progress = kwargs.pop("throttle_max_requests_in_progress", 10)
    self._throttle_max_requests_waiting_in_queue = kwargs.pop("throttle_max_requests_waiting_in_queue", 1000)
    self._throttle_max_seconds_to_wait_in_queue = kwargs.pop("throttle_max_seconds_to_wait_in_queue", 300.0)
    self._throttle_max_retry_attempts = kwargs.pop("throttle_max_retry_attempts", 3)
    self._throttle_initial_seconds_between_retry_attempts = kwargs.pop(
      "throttle_initial_seconds_between_retry_attempts", 5.0
    )
    self._throttle_max_seconds_between_retry_attempts = kwargs.pop(
      "throttle_max_seconds_between_retry_attempts", 60.0
    )

    self._queue_manager: Optional[QueueManager] = None

    self.kwargs = kwargs

    # Get gateway configuration
    gateway_url = get_gateway_url()
    api_key = get_gateway_api_key()

    # Get client metadata headers for AWS usage tracking (Phase 8a)
    self.client_metadata = get_client_metadata_headers()

    # Configure httpx timeout with separate connect/read timeouts
    timeout = httpx.Timeout(
      connect=self._connect_timeout,
      read=self._request_timeout,
      write=30.0,
      pool=10.0,
    )

    # Initialize Anthropic client pointing to gateway
    # Note: Anthropic SDK adds /v1/messages to base_url, so we don't add /v1 here
    self.client = AsyncAnthropic(
      api_key=api_key,
      base_url=gateway_url,
      timeout=timeout,
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

    if self._throttle:
      self.logger.info(
        f"Throttling enabled for {name}: "
        f"rpm={self._throttle_requests_per_minute}, "
        f"max_in_progress={self._throttle_max_requests_in_progress}, "
        f"max_queue={self._throttle_max_requests_waiting_in_queue}"
      )

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

    # Define the actual request function
    # Use with_raw_response to access headers for rate limit hints
    async def make_request():
      raw_response = await self.client.messages.with_raw_response.create(**request_kwargs)

      # Extract headers for rate limiting hints
      headers = raw_response.headers
      hint_rpm = self._extract_header_float(headers, "x-ratelimit-hint-rpm")
      circuit_state = headers.get("x-gateway-circuit-state")

      # Parse the actual response
      response = raw_response.parse()

      # Attach headers info to response for queue to use
      response._gateway_hint_rpm = hint_rpm
      response._gateway_circuit_state = circuit_state

      return response

    # Route through queue if throttling is enabled
    if self._throttle:
      response = await self._execute_with_queue(make_request)
    else:
      response = await make_request()

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

  async def _execute_with_queue(self, request_fn):
    """
    Execute a request through the queue with rate limiting and retries.

    :param request_fn: Async function that makes the actual request
    :return: Response from request_fn
    """
    # Lazy initialize queue manager
    if self._queue_manager is None:
      self._queue_manager = await QueueManager.get_instance()

      # Configure rate limiter for this model
      rate_config = RateLimiterConfig(
        initial_rpm=self._throttle_requests_per_minute,
        max_concurrent=self._throttle_max_requests_in_progress,
      )

      # Configure retry behavior
      retry_config = RetryConfig(
        max_retry_attempts=self._throttle_max_retry_attempts,
        initial_seconds_between_retry_attempts=self._throttle_initial_seconds_between_retry_attempts,
        max_seconds_between_retry_attempts=self._throttle_max_seconds_between_retry_attempts,
      )

      # Configure queue
      queue_config = QueueConfig(
        max_queue_depth=self._throttle_max_requests_waiting_in_queue,
        queue_timeout=self._throttle_max_seconds_to_wait_in_queue,
        rate_limiter_config=rate_config,
        retry_config=retry_config,
      )
      self._queue_manager.configure_model(self.name, queue_config)

    # Get queue for this model
    queue = await self._queue_manager.get_queue(self.name)

    # Submit request through queue
    # The queue's _execute_request will extract hints from the response
    # (attached as _gateway_hint_rpm and _gateway_circuit_state by make_request)
    return await queue.submit(request_fn)

  def _extract_header_float(self, headers, header_name: str) -> Optional[float]:
    """
    Extract a float value from a header.

    :param headers: Response headers
    :param header_name: Name of the header (case-insensitive)
    :return: Float value or None
    """
    value = headers.get(header_name)
    if value:
      try:
        return float(value)
      except (ValueError, TypeError):
        pass
    return None

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

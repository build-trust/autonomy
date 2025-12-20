"""
Gateway client using the OpenAI SDK.

This client connects to the Autonomy External APIs Gateway using the official
OpenAI Python SDK. The gateway handles provider routing, model aliasing, and
format translation internally.

This is the recommended client for:
- OpenAI models (gpt-4o, gpt-4o-mini, etc.)
- Non-Claude models routed through the gateway
- Embeddings, text-to-speech, and speech-to-text

For Claude models, use AnthropicGatewayClient instead (automatic when
AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1 is set).

Request Queuing:
When AUTONOMY_USE_REQUEST_QUEUE=1 is set, requests are routed through a
per-model queue with adaptive rate limiting (AIMD algorithm). This helps
prevent rate limit errors when making many concurrent requests.
"""

from typing import List, Optional
from copy import deepcopy

import tiktoken
from openai import AsyncOpenAI

from ...logs import get_logger, InfoContext, DebugContext
from ...nodes.message import ConversationMessage
from ...transcripts import log_raw_request, log_raw_response
from .gateway_config import (
  get_gateway_url,
  get_gateway_api_key,
  get_client_metadata_headers,
  use_request_queue,
  get_queue_initial_rpm,
  get_queue_max_concurrent,
)
from .rate_limiter import RateLimiterConfig
from .request_queue import QueueConfig, QueueManager


def normalize_messages(
  messages: List[dict] | List[ConversationMessage],
  is_thinking: bool = False,
  support_tools: bool = True,
  support_forced_assistant_answer: bool = True,
) -> List[dict]:
  """
  Normalize messages to the format expected by the OpenAI API.

  :param messages: List of messages (either dicts or ConversationMessage objects)
  :param is_thinking: Whether the model is in thinking mode
  :param support_tools: Whether the model supports tools
  :param support_forced_assistant_answer: Whether to support forced assistant answers
  :return: List of normalized message dicts
  """
  normalized = []

  for message in messages:
    if isinstance(message, dict):
      msg = deepcopy(message)
    else:
      # Convert ConversationMessage to dict
      # Extract content - handle TextContent objects
      content = getattr(message, "content", None)
      if hasattr(content, "text"):
        content = content.text

      msg = {
        "role": message.role.value if hasattr(message.role, "value") else str(message.role),
        "content": content,
      }

      # Handle tool calls - convert ToolCall objects to dicts for JSON serialization
      if hasattr(message, "tool_calls") and message.tool_calls:
        msg["tool_calls"] = [
          {
            "id": tc.id,
            "type": tc.type if hasattr(tc, "type") else "function",
            "function": {
              "name": tc.function.name
              if hasattr(tc.function, "name") and tc.function.name
              else tc.function.get("name", "unknown")
              if hasattr(tc.function, "get")
              else "unknown",
              "arguments": tc.function.arguments
              if hasattr(tc.function, "arguments")
              else tc.function.get("arguments", ""),
            },
          }
          for tc in message.tool_calls
        ]

      # Handle tool call ID for tool responses
      if hasattr(message, "tool_call_id") and message.tool_call_id:
        msg["tool_call_id"] = message.tool_call_id

    # Filter out tool-related fields if tools aren't supported
    if not support_tools:
      msg.pop("tool_calls", None)
      msg.pop("tool_call_id", None)
      # Convert tool role messages to user messages
      if msg.get("role") == "tool":
        msg["role"] = "user"
        msg["content"] = f"Tool response: {msg.get('content', '')}"

    normalized.append(msg)

  return normalized


class GatewayClient(InfoContext, DebugContext):
  """
  Model client that uses the OpenAI SDK to connect to the Autonomy External APIs Gateway.

  The gateway handles:
  - Provider routing (OpenAI, Azure, Anthropic, Bedrock)
  - Model aliasing and load balancing
  - Format translation for non-OpenAI providers
  - Model nuance normalization (e.g., <think> tags)
  """

  def __init__(self, name: str, max_input_tokens: Optional[int] = None, **kwargs):
    """
    Initialize the gateway client.

    :param name: Model name or alias
    :param max_input_tokens: Maximum input tokens for the model
    :param kwargs: Additional parameters including:
      - use_queue: Override queue setting (default from AUTONOMY_USE_REQUEST_QUEUE)
      - queue_initial_rpm: Initial RPM for rate limiter (default from AUTONOMY_QUEUE_INITIAL_RPM)
      - queue_max_concurrent: Max concurrent requests (default from AUTONOMY_QUEUE_MAX_CONCURRENT)
    """
    self.logger = get_logger("model")
    self.original_name = name
    self.name = name
    self.max_input_tokens = max_input_tokens

    # Extract queue-specific kwargs
    self._use_queue = kwargs.pop("use_queue", use_request_queue())
    self._queue_initial_rpm = kwargs.pop("queue_initial_rpm", get_queue_initial_rpm())
    self._queue_max_concurrent = kwargs.pop("queue_max_concurrent", get_queue_max_concurrent())
    self._queue_manager: Optional[QueueManager] = None

    self.kwargs = kwargs

    # Get gateway configuration
    gateway_url = get_gateway_url()
    api_key = get_gateway_api_key()

    # Get client metadata headers for AWS usage tracking (Phase 8a)
    self.client_metadata = get_client_metadata_headers()

    # Initialize OpenAI client pointing to gateway
    self.client = AsyncOpenAI(
      api_key=api_key,
      base_url=f"{gateway_url}/v1",
      timeout=120.0,  # 2 minute timeout for long completions
      default_headers=self.client_metadata if self.client_metadata else None,
    )

    # Initialize tokenizer for token counting
    # Use cl100k_base which is used by GPT-4 and Claude models
    try:
      self._encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
      self._encoding = None
      self.logger.warning("Failed to load tiktoken encoding, token counting may be inaccurate")

    if self._use_queue:
      self.logger.info(
        f"Request queue enabled for {name}: "
        f"initial_rpm={self._queue_initial_rpm}, max_concurrent={self._queue_max_concurrent}"
      )

  def count_tokens(
    self, messages: List[dict] | List[ConversationMessage], is_thinking: bool = False, tools=None
  ) -> int:
    """
    Count tokens in the given messages.

    Uses tiktoken for accurate token counting.

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

    # Use tiktoken for accurate counting
    total_tokens = 0

    for msg in messages:
      if isinstance(msg, dict):
        content = msg.get("content", "")
        role = msg.get("role", "")
      else:
        content = getattr(msg, "content", "") or ""
        role = getattr(msg, "role", "user")
        if hasattr(role, "value"):
          role = role.value

      # Count role tokens (roughly 4 tokens for message formatting)
      total_tokens += 4

      # Count content tokens
      if isinstance(content, str):
        total_tokens += len(self._encoding.encode(content))
      elif isinstance(content, list):
        # Handle multi-part content
        for part in content:
          if isinstance(part, dict) and part.get("type") == "text":
            total_tokens += len(self._encoding.encode(part.get("text", "")))

    # Add tokens for tools if provided
    if tools:
      import json

      tools_str = json.dumps(tools)
      total_tokens += len(self._encoding.encode(tools_str))

    return total_tokens

  def support_tools(self) -> bool:
    """Check if the model supports tools/function calling."""
    # Most models through the gateway support tools
    # The gateway handles translation for providers that need it
    name_lower = self.name.lower()

    # DeepSeek models don't support tools well
    if "deepseek" in name_lower:
      return False

    return True

  def support_forced_assistant_answer(self) -> bool:
    """Check if the model supports forced assistant answers."""
    # Bedrock-routed models don't support this well
    # The gateway will handle these internally
    return True

  def prepare_llm_call(self, messages: List[dict] | List[ConversationMessage], is_thinking: bool, **kwargs):
    """
    Prepare messages and kwargs for the LLM call.

    :param messages: Messages to prepare
    :param is_thinking: Whether in thinking mode
    :param kwargs: Additional parameters
    :return: Tuple of (normalized_messages, kwargs)
    """
    messages = normalize_messages(messages, is_thinking, self.support_tools(), self.support_forced_assistant_answer())

    # Merge default kwargs with provided kwargs
    merged_kwargs = {**self.kwargs, **kwargs}

    # Remove empty tools list
    if "tools" in merged_kwargs and len(merged_kwargs.get("tools", [])) == 0:
      del merged_kwargs["tools"]

    return messages, merged_kwargs

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
    Send a chat completion request through the gateway.

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
    self.logger.debug(f"Sending messages to gateway: {messages} (stream={stream})")

    messages, kwargs = self.prepare_llm_call(messages, is_thinking, **kwargs)

    if stream:
      return self._complete_chat_stream(
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
          messages,
          agent_name=agent_name,
          scope=scope,
          conversation=conversation,
          **kwargs,
        )

      return _complete()

  async def _complete_chat_stream(
    self,
    messages: List[dict],
    is_thinking: bool,
    agent_name: Optional[str] = None,
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
    **kwargs,
  ):
    """
    Stream chat completion from the gateway.

    :param messages: Prepared messages
    :param is_thinking: Whether in thinking mode
    :param agent_name: Optional agent name for logging
    :param scope: Optional scope for logging
    :param conversation: Optional conversation ID for logging
    :param kwargs: Additional parameters
    :yields: Response chunks
    """
    # Log raw request
    log_raw_request(
      payload={"model": self.name, "messages": messages, "stream": True, **kwargs},
      model_name=self.name,
      agent_name=agent_name,
      scope=scope,
      conversation=conversation,
    )

    # Make streaming request
    stream = await self.client.chat.completions.create(
      model=self.name,
      messages=messages,
      stream=True,
      stream_options={"include_usage": True},
      **kwargs,
    )

    # Track accumulated content for logging
    accumulated_content = ""
    accumulated_tool_calls = []
    last_finish_reason = None

    async for chunk in stream:
      # Track content
      if chunk.choices and chunk.choices[0].delta.content:
        content = chunk.choices[0].delta.content
        accumulated_content += content

        # Handle thinking tags (gateway normalizes these, but handle just in case)
        if "<think>" in content:
          is_thinking = True
          chunk.choices[0].delta.content = content.replace("<think>", "")

        if "</think>" in content:
          is_thinking = False
          chunk.choices[0].delta.content = content.replace("</think>", "")

        # Set reasoning_content for thinking mode
        if is_thinking:
          chunk.choices[0].delta.reasoning_content = chunk.choices[0].delta.content
          chunk.choices[0].delta.content = None

      # Track tool calls
      if chunk.choices and hasattr(chunk.choices[0].delta, "tool_calls"):
        if chunk.choices[0].delta.tool_calls:
          for tc in chunk.choices[0].delta.tool_calls:
            tc_index = tc.index if hasattr(tc, "index") else 0
            while len(accumulated_tool_calls) <= tc_index:
              accumulated_tool_calls.append(None)

            if accumulated_tool_calls[tc_index] is None:
              accumulated_tool_calls[tc_index] = {
                "id": tc.id if hasattr(tc, "id") and tc.id else f"call_{tc_index}",
                "type": "function",
                "function": {
                  "name": tc.function.name if hasattr(tc.function, "name") and tc.function.name else "unknown",
                  "arguments": tc.function.arguments if hasattr(tc.function, "arguments") else "",
                },
              }
            else:
              # Update name if it arrives in a later chunk and was previously "unknown"
              if hasattr(tc.function, "name") and tc.function.name:
                if accumulated_tool_calls[tc_index]["function"]["name"] == "unknown":
                  accumulated_tool_calls[tc_index]["function"]["name"] = tc.function.name
              if hasattr(tc.function, "arguments") and tc.function.arguments:
                accumulated_tool_calls[tc_index]["function"]["arguments"] += tc.function.arguments

      # Track finish reason
      if chunk.choices and chunk.choices[0].finish_reason:
        last_finish_reason = chunk.choices[0].finish_reason

      yield chunk

    # Log accumulated response
    if last_finish_reason:
      response_dict = {
        "choices": [
          {
            "message": {
              "role": "assistant",
              "content": accumulated_content if accumulated_content else None,
            },
            "finish_reason": last_finish_reason,
          }
        ]
      }
      if accumulated_tool_calls:
        response_dict["choices"][0]["message"]["tool_calls"] = [tc for tc in accumulated_tool_calls if tc is not None]

      log_raw_response(
        response=response_dict,
        model_name=self.name,
        agent_name=agent_name,
        scope=scope,
        conversation=conversation,
      )

  async def _complete_chat(
    self,
    messages: List[dict],
    agent_name: Optional[str] = None,
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
    **kwargs,
  ):
    """
    Non-streaming chat completion from the gateway.

    :param messages: Prepared messages
    :param agent_name: Optional agent name for logging
    :param scope: Optional scope for logging
    :param conversation: Optional conversation ID for logging
    :param kwargs: Additional parameters
    :return: Completion response
    """
    # Log raw request
    log_raw_request(
      payload={"model": self.name, "messages": messages, "stream": False, **kwargs},
      model_name=self.name,
      agent_name=agent_name,
      scope=scope,
      conversation=conversation,
    )

    # Define the actual request function
    # Use with_raw_response to access headers for rate limit hints
    async def make_request():
      raw_response = await self.client.chat.completions.with_raw_response.create(
        model=self.name,
        messages=messages,
        **kwargs,
      )
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

    # Route through queue if enabled
    if self._use_queue:
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

    return response

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

  async def _execute_with_queue(self, request_fn):
    """
    Execute a request through the queue with rate limiting.

    :param request_fn: Async function that makes the actual request
    :return: Response from request_fn
    """
    # Lazy initialize queue manager
    if self._queue_manager is None:
      self._queue_manager = await QueueManager.get_instance()

      # Configure rate limiter for this model
      rate_config = RateLimiterConfig(
        initial_rpm=self._queue_initial_rpm,
        max_concurrent=self._queue_max_concurrent,
      )
      queue_config = QueueConfig(rate_limiter_config=rate_config)
      self._queue_manager.configure_model(self.name, queue_config)

    # Get queue for this model
    queue = await self._queue_manager.get_queue(self.name)

    # Submit request through queue
    # The queue's _execute_request will extract hints from the response
    # (attached as _gateway_hint_rpm and _gateway_circuit_state by make_request)
    return await queue.submit(request_fn)

  async def embeddings(self, text: List[str], **kwargs) -> List[List[float]]:
    """
    Generate embeddings through the gateway.

    :param text: List of texts to embed
    :param kwargs: Additional parameters
    :return: List of embedding vectors
    """
    self.logger.info(f"Generating embeddings for {len(text)} texts")

    response = await self.client.embeddings.create(
      model=self.name,
      input=text,
      **kwargs,
    )

    return [item.embedding for item in response.data]

  async def text_to_speech(self, text: str, voice: str = "alloy", response_format: str = "mp3", **kwargs) -> bytes:
    """
    Convert text to speech through the gateway.

    :param text: Text to convert
    :param voice: Voice to use
    :param response_format: Audio format
    :param kwargs: Additional parameters
    :return: Audio bytes
    """
    self.logger.info(f"Converting {len(text)} characters to speech with voice '{voice}'")

    # Use tts-1 model for text-to-speech
    tts_model = kwargs.pop("model", "tts-1")

    response = await self.client.audio.speech.create(
      model=tts_model,
      voice=voice,
      input=text,
      response_format=response_format,
      **kwargs,
    )

    return response.content

  async def speech_to_text(self, audio_file, language: Optional[str] = None, **kwargs) -> str:
    """
    Transcribe audio to text through the gateway.

    :param audio_file: Audio file or bytes to transcribe
    :param language: Optional language code
    :param kwargs: Additional parameters
    :return: Transcribed text
    """
    self.logger.info("Transcribing audio to text")

    # Use whisper-1 model for speech-to-text
    stt_model = kwargs.pop("model", "whisper-1")

    transcription = await self.client.audio.transcriptions.create(
      model=stt_model,
      file=audio_file,
      language=language,
      **kwargs,
    )

    return transcription.text

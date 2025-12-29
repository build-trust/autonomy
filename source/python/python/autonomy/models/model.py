from typing import List, Optional
import os
import httpx
from ..logs import get_logger, InfoContext, DebugContext
from ..nodes.message import ConversationMessage
from .clients.litellm_client import LiteLLMClient, PROVIDER_ALIASES, ALL_PROVIDER_ALLOWED_FULL_NAMES
from .clients.gateway_client import GatewayClient
from .clients.anthropic_gateway_client import AnthropicGatewayClient
from .clients.gateway_config import use_anthropic_sdk, get_gateway_url, get_gateway_api_key

logger = get_logger("model")

# Build comprehensive model list from all providers
MODEL_CLIENTS = {}

# Add all models from all providers
for provider_name, models in PROVIDER_ALIASES.items():
  for model_alias in models.keys():
    MODEL_CLIENTS[model_alias] = "litellm"

# Also add full model names for direct access
for full_name in ALL_PROVIDER_ALLOWED_FULL_NAMES:
  MODEL_CLIENTS[full_name] = "litellm"

# Models that should use Anthropic SDK when using gateway
# These are the supported Claude models via Bedrock (with Anthropic direct as failover)
ANTHROPIC_MODELS = {
  # Claude 4.5 (latest)
  "claude-sonnet-4-5",
  "claude-haiku-4-5",
  "claude-opus-4-5",
  # Claude 4 (stable)
  "claude-sonnet-4",
  "claude-opus-4",
  # Backward-compatible aliases (published names)
  "claude-sonnet-4-v1",
  "claude-opus-4-v1",
}


def is_anthropic_model(model_name: str) -> bool:
  """Check if a model should use the Anthropic SDK."""
  name_lower = model_name.lower()

  # Check explicit matches
  if model_name in ANTHROPIC_MODELS:
    return True

  # Check if it starts with claude
  if name_lower.startswith("claude"):
    return True

  # Check if it contains anthropic
  if "anthropic" in name_lower:
    return True

  return False


def _fetch_gateway_models() -> List[str]:
  """
  Fetch available models from the gateway and deduplicate by showing
  aliases instead of multiple full model IDs.

  When two models share the same alias (e.g., Bedrock and Anthropic direct
  both have Claude models), show only the alias once.

  :return: List of model IDs/aliases from the gateway (deduplicated)
  """
  try:
    gateway_url = get_gateway_url()
    api_key = get_gateway_api_key()

    response = httpx.get(
      f"{gateway_url}/v1/models",
      headers={"Authorization": f"Bearer {api_key}"},
      timeout=10.0,
    )
    response.raise_for_status()

    data = response.json()
    raw_models = [model["id"] for model in data.get("data", [])]

    # Deduplicate models by grouping those that share common base names
    # Pattern: models like "claude-sonnet-4-5-20250929" and "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    # should be represented by the alias "claude-sonnet-4-5"

    # Known alias patterns that map full model IDs to short aliases
    alias_patterns = {
      # Claude 4.5
      "claude-sonnet-4-5": ["claude-sonnet-4-5-20250929", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"],
      "claude-haiku-4-5": ["claude-haiku-4-5-20251001", "us.anthropic.claude-haiku-4-5-20251001-v1:0"],
      "claude-opus-4-5": ["claude-opus-4-5-20251101", "us.anthropic.claude-opus-4-5-20251101-v1:0"],
      # Claude 4
      "claude-sonnet-4": ["claude-sonnet-4-20250514", "us.anthropic.claude-sonnet-4-20250514-v1:0"],
      "claude-opus-4": ["claude-opus-4-20250514", "us.anthropic.claude-opus-4-20250514-v1:0"],
      # Nova models
      "nova-micro": ["us.amazon.nova-micro-v1:0"],
      "nova-lite": ["us.amazon.nova-lite-v1:0"],
      "nova-pro": ["us.amazon.nova-pro-v1:0"],
      "nova-premier": ["us.amazon.nova-premier-v1:0"],
      # Embeddings
      "titan-embed": ["amazon.titan-embed-text-v2:0"],
      "embed-english": ["cohere.embed-english-v3"],
      "embed-multilingual": ["cohere.embed-multilingual-v3"],
    }

    # Build reverse mapping: full model ID -> alias
    full_id_to_alias = {}
    for alias, full_ids in alias_patterns.items():
      for full_id in full_ids:
        full_id_to_alias[full_id] = alias

    # Process models: replace full IDs with aliases, deduplicate
    result_set = set()
    for model in raw_models:
      if model in full_id_to_alias:
        # Replace with alias
        result_set.add(full_id_to_alias[model])
      else:
        # Keep as-is (no alias mapping)
        result_set.add(model)

    return sorted(result_set)
  except Exception as e:
    logger.warning(f"Failed to fetch models from gateway: {e}")
    return []


class Model(InfoContext, DebugContext):
  """
  Model class that provides a unified interface to different LLM providers.

  **Recommended: Use Gateway Mode**

  Set ``AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1`` to use the Autonomy External APIs Gateway.
  The gateway provides:

  - Centralized provider routing (OpenAI, Anthropic, AWS Bedrock)
  - Automatic model aliasing and load balancing
  - AWS usage tracking via inference profiles (tagged by cluster/zone)
  - No AWS credentials needed in client containers

  Gateway environment variables:

  - ``AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1`` - Enable gateway mode (recommended)
  - ``AUTONOMY_EXTERNAL_APIS_GATEWAY_URL`` - Gateway URL (default: http://localhost:8000)
  - ``AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY`` - API key (default: test_key, requires GATEWAY_TEST_MODE=true)
  - ``CLUSTER`` - Cluster ID for AWS usage tracking
  - ``ZONE`` - Zone ID for AWS usage tracking

  **Timeout Configuration**

  Configure HTTP client timeouts for different scenarios:

  - ``request_timeout`` - Default timeout for LLM calls (default: 120s)
  - ``connect_timeout`` - Connection establishment timeout (default: 10s)
  - ``stream_timeout`` - Timeout for streaming responses (default: 300s)

  **Throttling Configuration**

  Enable client-side rate limiting and request queuing:

  - ``throttle`` - Enable throttling (default: False)
  - ``throttle_requests_per_minute`` - Initial RPM for rate limiter (default: 60)
  - ``throttle_max_requests_in_progress`` - Max concurrent requests (default: 10)
  - ``throttle_max_requests_waiting_in_queue`` - Max queue depth (default: 1000)
  - ``throttle_max_seconds_to_wait_in_queue`` - Queue timeout (default: 300s)
  - ``throttle_max_retry_attempts`` - Max retries on rate limit (default: 3)
  - ``throttle_initial_seconds_between_retry_attempts`` - Initial backoff (default: 5s)
  - ``throttle_max_seconds_between_retry_attempts`` - Max backoff (default: 60s)

  **Legacy Modes (Deprecated)**

  Direct Bedrock access is deprecated. The following still work but emit
  deprecation warnings:

  - ``LITELLM_PROXY_API_BASE`` - Use LiteLLM proxy
  - ``AUTONOMY_MODEL_PROVIDER=bedrock`` - Direct Bedrock access (deprecated)
  - Automatic Bedrock detection via AWS credentials (deprecated)

  The model automatically detects the provider based on environment variables
  and routes requests to the appropriate client implementation.
  """

  def __init__(
    self,
    name: str,
    max_input_tokens: Optional[int] = None,
    # Layer 1: HTTP Client Timeouts
    request_timeout: float = 120.0,
    connect_timeout: float = 10.0,
    stream_timeout: float = 300.0,
    # Layer 2-4: Throttling (rate limiting, queue, retries)
    throttle: bool = False,
    throttle_requests_per_minute: float = 60.0,
    throttle_max_requests_in_progress: int = 10,
    throttle_max_requests_waiting_in_queue: int = 1000,
    throttle_max_seconds_to_wait_in_queue: float = 300.0,
    throttle_max_retry_attempts: int = 3,
    throttle_initial_seconds_between_retry_attempts: float = 5.0,
    throttle_max_seconds_between_retry_attempts: float = 60.0,
    **kwargs,
  ):
    """
    Initialize a Model instance.

    :param name: Model name or alias (e.g., 'claude-sonnet-4', 'gpt-4o')
    :param max_input_tokens: Maximum input tokens for the model

    :param request_timeout: Default timeout for LLM calls in seconds (default: 120.0).
        Reasoning models like o1 can take 60-120s, so this is set conservatively.
    :param connect_timeout: Connection establishment timeout in seconds (default: 10.0).
        If connection can't be established in 10s, fail fast.
    :param stream_timeout: Timeout for streaming responses in seconds (default: 300.0).
        Long streaming responses may need 5+ minutes.

    :param throttle: Enable client-side throttling with rate limiting, queuing, and retries
        (default: False). When False, requests are sent directly without queueing.
    :param throttle_requests_per_minute: Initial requests per minute for rate limiter
        (default: 60.0). The AIMD algorithm will adjust this based on 429 responses.
    :param throttle_max_requests_in_progress: Maximum concurrent requests (default: 10).
        Balances parallelism vs overwhelming the backend.
    :param throttle_max_requests_waiting_in_queue: Maximum queue depth (default: 1000).
        Provides generous backlog without memory runaway.
    :param throttle_max_seconds_to_wait_in_queue: Queue timeout in seconds (default: 300.0).
        5 minutes handles typical backlog clearing.
    :param throttle_max_retry_attempts: Maximum retry attempts on rate limit (default: 3).
        Covers transient failures without hammering the backend.
    :param throttle_initial_seconds_between_retry_attempts: Initial retry backoff in seconds
        (default: 5.0). Matches gateway's default retry-after.
    :param throttle_max_seconds_between_retry_attempts: Maximum retry backoff in seconds
        (default: 60.0). Caps exponential backoff (5→10→20→40→60).

    :param kwargs: Additional parameters to pass to the underlying client
    """
    self.name = name
    self.logger = get_logger("model")
    self.max_input_tokens = max_input_tokens

    # Store timeout configuration
    self.request_timeout = request_timeout
    self.connect_timeout = connect_timeout
    self.stream_timeout = stream_timeout

    # Store throttle configuration
    self.throttle = throttle
    self.throttle_requests_per_minute = throttle_requests_per_minute
    self.throttle_max_requests_in_progress = throttle_max_requests_in_progress
    self.throttle_max_requests_waiting_in_queue = throttle_max_requests_waiting_in_queue
    self.throttle_max_seconds_to_wait_in_queue = throttle_max_seconds_to_wait_in_queue
    self.throttle_max_retry_attempts = throttle_max_retry_attempts
    self.throttle_initial_seconds_between_retry_attempts = throttle_initial_seconds_between_retry_attempts
    self.throttle_max_seconds_between_retry_attempts = throttle_max_seconds_between_retry_attempts

    self.kwargs = kwargs

    # Pick the appropriate client
    self.client = self._pick_client(name, max_input_tokens, **kwargs)

  def _pick_client(self, model_name: str, max_input_tokens: Optional[int] = None, **kwargs):
    """Pick the appropriate client for the given model."""
    # Check for gateway mode first (new preferred method)
    if os.environ.get("AUTONOMY_USE_EXTERNAL_APIS_GATEWAY") == "1":
      return self._pick_gateway_client(model_name, max_input_tokens, **kwargs)

    # Check explicit model mappings
    if model_name in MODEL_CLIENTS:
      client_name = MODEL_CLIENTS[model_name]
      if client_name == "litellm":
        return LiteLLMClient(model_name, max_input_tokens, **kwargs)

    # Try LiteLLM as fallback
    try:
      return LiteLLMClient(model_name, max_input_tokens, **kwargs)
    except ValueError as e:
      raise ValueError(f"Model '{model_name}' is not supported. {str(e)}")

  def _pick_gateway_client(self, model_name: str, max_input_tokens: Optional[int] = None, **kwargs):
    """
    Pick the appropriate gateway client based on model type.

    For Claude models, use AnthropicGatewayClient to leverage native Anthropic SDK.
    For all other models, use GatewayClient with OpenAI SDK.
    """
    # Build common configuration dict for gateway clients
    gateway_config = {
      # Timeouts
      "request_timeout": self.request_timeout,
      "connect_timeout": self.connect_timeout,
      "stream_timeout": self.stream_timeout,
      # Throttling
      "throttle": self.throttle,
      "throttle_requests_per_minute": self.throttle_requests_per_minute,
      "throttle_max_requests_in_progress": self.throttle_max_requests_in_progress,
      "throttle_max_requests_waiting_in_queue": self.throttle_max_requests_waiting_in_queue,
      "throttle_max_seconds_to_wait_in_queue": self.throttle_max_seconds_to_wait_in_queue,
      "throttle_max_retry_attempts": self.throttle_max_retry_attempts,
      "throttle_initial_seconds_between_retry_attempts": self.throttle_initial_seconds_between_retry_attempts,
      "throttle_max_seconds_between_retry_attempts": self.throttle_max_seconds_between_retry_attempts,
    }
    # Merge with any additional kwargs
    gateway_config.update(kwargs)

    # Check if we should use Anthropic SDK for this model
    if use_anthropic_sdk() and is_anthropic_model(model_name):
      self.logger.debug(f"Using AnthropicGatewayClient for model '{model_name}'")
      return AnthropicGatewayClient(model_name, max_input_tokens, **gateway_config)
    else:
      self.logger.debug(f"Using GatewayClient for model '{model_name}'")
      return GatewayClient(model_name, max_input_tokens, **gateway_config)

  def count_tokens(
    self, messages: List[dict] | List[ConversationMessage], is_thinking: bool = False, tools=None
  ) -> int:
    """
    Count the number of tokens in the given messages.

    :param messages: List of messages to count tokens for
    :param is_thinking: Whether the model is in thinking mode
    :param tools: Tools available to the model
    :return: Number of tokens
    """
    return self.client.count_tokens(messages, is_thinking, tools)

  def support_tools(self) -> bool:
    """Check if the model supports tools/function calling."""
    return self.client.support_tools()

  def support_forced_assistant_answer(self) -> bool:
    """Check if the model supports forced assistant answers."""
    return self.client.support_forced_assistant_answer()

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
    Send a chat completion request to the model.

    :param messages: List of messages to send to the model
    :param stream: Whether to stream the response
    :param is_thinking: Whether the model is in thinking mode
    :param agent_name: Optional agent name for transcript correlation
    :param scope: Optional scope identifier for transcript correlation
    :param conversation: Optional conversation identifier for transcript correlation
    :param kwargs: Additional parameters for the model
    :return: Model response (streamed or complete)
    """
    result = self.client.complete_chat(
      messages, stream, is_thinking, agent_name=agent_name, scope=scope, conversation=conversation, **kwargs
    )

    if stream:
      # For streaming, return the async generator directly
      return result
    else:
      # For non-streaming, return an awaitable coroutine
      async def _await_result():
        return await result

      return _await_result()

  async def embeddings(self, text: List[str], **kwargs) -> List[List[float]]:
    """
    Generate embeddings for the given text.

    :param text: List of text strings to embed
    :param kwargs: Additional parameters
    :return: List of embedding vectors
    """
    return await self.client.embeddings(text, **kwargs)

  async def text_to_speech(self, text: str, voice: str = "alloy", response_format: str = "mp3", **kwargs) -> bytes:
    """
    Convert text to speech audio.

    :param text: Text to convert to speech
    :param voice: Voice to use (e.g., 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')
    :param response_format: Audio format (e.g., 'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm')
    :param kwargs: Additional parameters
    :return: Audio bytes
    """
    return await self.client.text_to_speech(text, voice, response_format, **kwargs)

  async def speech_to_text(self, audio_file, language: Optional[str] = None, **kwargs):
    """
    Transcribe audio to text.

    :param audio_file: Audio file or bytes to transcribe
    :param language: Optional language code (e.g., 'en', 'es', 'fr')
    :param kwargs: Additional parameters including:
      - model: Model to use (default uses instance model, can use "gpt-4o-transcribe-diarize" for diarization)
      - response_format: Response format (use "diarized_json" for speaker diarization with gpt-4o-transcribe-diarize)
    :return: Transcribed text (str) or full response object if response_format is specified
    """
    return await self.client.speech_to_text(audio_file, language, **kwargs)

  @staticmethod
  def list() -> List[str]:
    """
    List all available model names/aliases.

    When gateway mode is enabled (AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1),
    fetches the list of models from the gateway and deduplicates them by
    showing aliases instead of multiple full model IDs.

    :return: List of supported model names
    """
    # Check for gateway mode
    if os.environ.get("AUTONOMY_USE_EXTERNAL_APIS_GATEWAY") == "1":
      gateway_models = _fetch_gateway_models()
      if gateway_models:
        return gateway_models
      # Fall back to static list if gateway fetch fails
      logger.warning("Falling back to static model list")

    return list(MODEL_CLIENTS.keys())

  @staticmethod
  def list_by_provider() -> dict[str, List[str]]:
    """
    List available models organized by provider.

    :return: Dictionary mapping provider names to their available models
    """
    result = {}
    for provider_name, models in PROVIDER_ALIASES.items():
      result[provider_name] = list(models.keys())

    # Add gateway as a provider option
    result["gateway"] = list(ANTHROPIC_MODELS) + ["gpt-4o", "gpt-4o-mini", "text-embedding-3-small"]
    return result

  @property
  def original_name(self) -> str:
    """Get the original model name used during initialization."""
    return getattr(self.client, "original_name", self.name)

  @property
  def resolved_name(self) -> str:
    """Get the resolved model name used by the provider."""
    return getattr(self.client, "name", self.name)

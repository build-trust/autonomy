from typing import List, Optional
import os
from ..logs import get_logger, InfoContext, DebugContext
from ..nodes.message import ConversationMessage
from .clients.litellm_client import LiteLLMClient, PROVIDER_ALIASES, ALL_PROVIDER_ALLOWED_FULL_NAMES
from .clients.bedrock_client import BedrockClient, BEDROCK_MODELS

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

# Add direct Bedrock models
for model_alias in BEDROCK_MODELS.keys():
  MODEL_CLIENTS[f"bedrock-direct/{model_alias}"] = "bedrock_direct"
  # Don't set default here - check at runtime in _pick_client


class Model(InfoContext, DebugContext):
  """
  Model class that provides a unified interface to different LLM providers.

  This class supports multiple providers including:
  - LiteLLM Proxy
  - AWS Bedrock
  - Ollama

  The model automatically detects the provider based on environment variables
  and routes requests to the appropriate client implementation.
  """

  def __init__(self, name: str, max_input_tokens: Optional[int] = None, **kwargs):
    """
    Initialize a Model instance.

    :param name: Model name or alias (e.g., 'claude-3-5-sonnet-v2', 'llama3.3')
    :param max_input_tokens: Maximum input tokens for the model
    :param kwargs: Additional parameters to pass to the underlying client
    """
    self.name = name
    self.logger = get_logger("model")
    self.max_input_tokens = max_input_tokens
    self.kwargs = kwargs

    # Pick the appropriate client
    self.client = self._pick_client(name, max_input_tokens, **kwargs)

  def _pick_client(self, model_name: str, max_input_tokens: Optional[int] = None, **kwargs):
    """Pick the appropriate client for the given model."""
    # Check for global direct Bedrock setting first
    if os.environ.get("AUTONOMY_USE_DIRECT_BEDROCK") == "1":
      if model_name in BEDROCK_MODELS or any(
        provider in model_name for provider in ["anthropic", "meta", "amazon", "cohere"]
      ):
        clean_name = model_name.replace("bedrock-direct/", "")
        return BedrockClient(clean_name, max_input_tokens, **kwargs)

    # Check explicit model mappings
    if model_name in MODEL_CLIENTS:
      client_name = MODEL_CLIENTS[model_name]
      if client_name == "litellm":
        return LiteLLMClient(model_name, max_input_tokens, **kwargs)
      elif client_name == "bedrock_direct":
        # Strip bedrock-direct/ prefix if present
        clean_name = model_name.replace("bedrock-direct/", "")
        return BedrockClient(clean_name, max_input_tokens, **kwargs)

    # If model not found in our supported list, try different clients
    # Check if it's a direct Bedrock model
    if model_name in BEDROCK_MODELS:
      return BedrockClient(model_name, max_input_tokens, **kwargs)

    # Try LiteLLM as fallback
    try:
      return LiteLLMClient(model_name, max_input_tokens, **kwargs)
    except ValueError as e:
      # Last resort: try direct Bedrock if it might be a model ID
      if any(provider in model_name for provider in ["anthropic", "meta", "amazon", "cohere"]):
        try:
          return BedrockClient(model_name, max_input_tokens, **kwargs)
        except Exception:
          pass
      raise ValueError(f"Model '{model_name}' is not supported. {str(e)}")

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

  @staticmethod
  def list() -> List[str]:
    """
    List all available model names/aliases.

    :return: List of supported model names
    """
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

    # Add direct Bedrock models
    result["bedrock_direct"] = list(BEDROCK_MODELS.keys())
    return result

  @property
  def original_name(self) -> str:
    """Get the original model name used during initialization."""
    return getattr(self.client, "original_name", self.name)

  @property
  def resolved_name(self) -> str:
    """Get the resolved model name used by the provider."""
    return getattr(self.client, "name", self.name)

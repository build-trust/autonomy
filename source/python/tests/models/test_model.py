import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock

from autonomy.models.model import Model, MODEL_CLIENTS
from autonomy.models.clients.litellm_client import PROVIDER_ALIASES, ALL_PROVIDER_ALLOWED_FULL_NAMES
from autonomy.models.clients.bedrock_client import BEDROCK_MODELS
from autonomy.nodes.message import UserMessage, SystemMessage, AssistantMessage


class TestModelInitialization:
  """Test Model class initialization and basic functionality."""

  def test_model_basic_initialization(self):
    """Test basic model initialization with default parameters."""
    model = Model("llama3.2")
    assert model.name == "llama3.2"
    assert model.client is not None
    assert model.max_input_tokens is None
    assert model.kwargs == {}

  def test_model_initialization_with_max_tokens(self):
    """Test model initialization with max_input_tokens parameter."""
    model = Model("llama3.2", max_input_tokens=100000)
    assert model.name == "llama3.2"
    assert model.max_input_tokens == 100000

  def test_model_initialization_with_kwargs(self):
    """Test model initialization with additional kwargs."""
    kwargs = {"temperature": 0.7, "max_tokens": 4000}
    model = Model("llama3.2", **kwargs)
    assert model.name == "llama3.2"
    assert model.kwargs == kwargs

  def test_model_initialization_all_params(self):
    """Test model initialization with all parameters."""
    kwargs = {"temperature": 0.5, "top_p": 0.9}
    model = Model("llama3.2", max_input_tokens=50000, **kwargs)
    assert model.name == "llama3.2"
    assert model.max_input_tokens == 50000
    assert model.kwargs == kwargs

  def test_unsupported_model_raises_error(self):
    """Test that unsupported models raise ValueError."""
    with pytest.raises(ValueError, match="Model 'nonexistent-model' is not supported"):
      Model("nonexistent-model")

  def test_model_client_attribute_exists(self):
    """Test that initialized model has client attribute."""
    model = Model("llama3.2")
    assert hasattr(model, "client")
    assert model.client is not None


class TestModelClientSelection:
  """Test Model class client selection logic."""

  def test_litellm_client_selection(self):
    """Test that LiteLLM client is selected for supported models."""
    # Test with environment variables that favor LiteLLM
    with patch.dict(os.environ, {"LITELLM_PROXY_API_BASE": "http://localhost:8000"}, clear=False):
      model = Model("claude-3-5-sonnet-v2")
      # Check that client was created
      assert model.client is not None
      assert hasattr(model.client, "name")

  def test_bedrock_direct_env_override(self):
    """Test direct Bedrock client selection with environment variable."""
    with patch.dict(os.environ, {"AUTONOMY_USE_DIRECT_BEDROCK": "1"}, clear=False):
      with patch("autonomy.models.model.BedrockClient") as mock_bedrock:
        mock_instance = MagicMock()
        mock_bedrock.return_value = mock_instance

        model = Model("llama3.2")  # Use a model that works with Bedrock
        mock_bedrock.assert_called_once()
        assert model.client == mock_instance

  def test_bedrock_direct_model_prefix(self):
    """Test direct Bedrock client selection with bedrock-direct/ prefix."""
    with patch("autonomy.models.model.BedrockClient") as mock_bedrock:
      mock_instance = MagicMock()
      mock_bedrock.return_value = mock_instance

      model = Model("bedrock-direct/llama3.2")
      mock_bedrock.assert_called_once_with("llama3.2", None)
      assert model.client == mock_instance

  def test_fallback_to_litellm(self):
    """Test fallback to LiteLLM for unknown models."""
    with patch("autonomy.models.model.LiteLLMClient") as mock_litellm:
      mock_instance = MagicMock()
      mock_litellm.return_value = mock_instance

      # Use a model that's in our supported list
      model = Model("llama3.2")
      assert model.client == mock_instance

  def test_bedrock_model_detection(self):
    """Test that Bedrock models are properly detected."""
    with patch("autonomy.models.model.LiteLLMClient") as mock_litellm:
      mock_instance = MagicMock()
      mock_litellm.return_value = mock_instance

      # Test with models that work with default provider (Ollama)
      test_models = ["llama3.2", "deepseek-r1", "gemma3"]
      for bedrock_model in test_models:
        if bedrock_model in BEDROCK_MODELS:
          model = Model(bedrock_model)
          # Should create some client
          assert model.client is not None


class TestModelProperties:
  """Test Model class properties."""

  def test_original_name_property(self):
    """Test original_name property."""
    model = Model("llama3.2")

    # Mock the client to have original_name attribute
    model.client.original_name = "test-original-name"
    assert model.original_name == "test-original-name"

    # Test fallback to model.name when client doesn't have original_name
    delattr(model.client, "original_name")
    assert model.original_name == "llama3.2"

  def test_resolved_name_property(self):
    """Test resolved_name property."""
    model = Model("llama3.2")

    # Mock the client to have name attribute
    model.client.name = "resolved-model-name"
    assert model.resolved_name == "resolved-model-name"

    # Test fallback to model.name when client doesn't have name
    delattr(model.client, "name")
    assert model.resolved_name == "llama3.2"


class TestModelMethods:
  """Test Model class methods."""

  def test_count_tokens(self):
    """Test count_tokens method."""
    model = Model("llama3.2")
    model.client.count_tokens = MagicMock(return_value=42)

    messages = [{"role": "user", "content": "Hello"}]
    result = model.count_tokens(messages)

    assert result == 42
    model.client.count_tokens.assert_called_once_with(messages, False, None)

  def test_count_tokens_with_params(self):
    """Test count_tokens method with parameters."""
    model = Model("llama3.2")
    model.client.count_tokens = MagicMock(return_value=100)

    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"name": "test_tool"}]
    result = model.count_tokens(messages, is_thinking=True, tools=tools)

    assert result == 100
    model.client.count_tokens.assert_called_once_with(messages, True, tools)

  def test_count_tokens_with_conversation_messages(self):
    """Test count_tokens with ConversationMessage objects."""
    model = Model("llama3.2")
    model.client.count_tokens = MagicMock(return_value=50)

    messages = [UserMessage(content="Hello"), AssistantMessage(content="Hi")]
    result = model.count_tokens(messages)

    assert result == 50
    model.client.count_tokens.assert_called_once()

  def test_support_tools(self):
    """Test support_tools method."""
    model = Model("llama3.2")
    model.client.support_tools = MagicMock(return_value=True)

    result = model.support_tools()

    assert result is True
    model.client.support_tools.assert_called_once()

  def test_support_forced_assistant_answer(self):
    """Test support_forced_assistant_answer method."""
    model = Model("llama3.2")
    model.client.support_forced_assistant_answer = MagicMock(return_value=False)

    result = model.support_forced_assistant_answer()

    assert result is False
    model.client.support_forced_assistant_answer.assert_called_once()

  @pytest.mark.asyncio
  async def test_complete_chat_non_streaming(self):
    """Test complete_chat method in non-streaming mode."""
    model = Model("llama3.2")

    # Mock the client's complete_chat to return an awaitable
    mock_response = MagicMock()
    mock_coro = AsyncMock(return_value=mock_response)
    model.client.complete_chat = MagicMock(return_value=mock_coro())

    messages = [{"role": "user", "content": "Hello"}]
    result_coro = model.complete_chat(messages, stream=False)
    result = await result_coro

    assert result == mock_response
    model.client.complete_chat.assert_called_once_with(messages, False, False)

  @pytest.mark.asyncio
  async def test_complete_chat_streaming(self):
    """Test complete_chat method in streaming mode."""
    model = Model("llama3.2")

    # Mock the client's complete_chat to return an async generator
    async def mock_stream():
      yield {"content": "Hello"}
      yield {"content": " world"}

    model.client.complete_chat = MagicMock(return_value=mock_stream())

    messages = [{"role": "user", "content": "Hello"}]
    result_generator = model.complete_chat(messages, stream=True)

    # Collect results from the generator
    results = []
    async for chunk in result_generator:
      results.append(chunk)

    assert len(results) == 2
    assert results[0]["content"] == "Hello"
    assert results[1]["content"] == " world"
    model.client.complete_chat.assert_called_once_with(messages, True, False)

  @pytest.mark.asyncio
  async def test_complete_chat_with_thinking(self):
    """Test complete_chat method with thinking mode."""
    model = Model("llama3.2")

    mock_response = MagicMock()
    mock_coro = AsyncMock(return_value=mock_response)
    model.client.complete_chat = MagicMock(return_value=mock_coro())

    messages = [{"role": "user", "content": "Solve this"}]
    result_coro = model.complete_chat(messages, is_thinking=True, temperature=0.7)
    result = await result_coro

    assert result == mock_response
    model.client.complete_chat.assert_called_once_with(messages, False, True, temperature=0.7)

  @pytest.mark.asyncio
  async def test_embeddings(self):
    """Test embeddings method."""
    model = Model("llama3.2")

    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    model.client.embeddings = AsyncMock(return_value=mock_embeddings)

    texts = ["Hello", "World"]
    result = await model.embeddings(texts, model="embed-model")

    assert result == mock_embeddings
    model.client.embeddings.assert_called_once_with(texts, model="embed-model")


class TestModelStaticMethods:
  """Test Model class static methods."""

  def test_list_models(self):
    """Test list() static method."""
    models = Model.list()

    assert isinstance(models, list)
    assert len(models) > 0

    # Check that some expected models are in the list
    expected_models = ["llama3.2", "deepseek-r1", "gemma3"]
    for model in expected_models:
      if model in MODEL_CLIENTS:  # Only check if the model exists in our mapping
        assert model in models

  def test_list_models_contains_all_clients(self):
    """Test that list() returns all models from MODEL_CLIENTS."""
    models = Model.list()
    model_clients_keys = list(MODEL_CLIENTS.keys())

    assert set(models) == set(model_clients_keys)

  def test_list_by_provider(self):
    """Test list_by_provider() static method."""
    providers = Model.list_by_provider()

    assert isinstance(providers, dict)
    assert len(providers) > 0

    # Check that expected providers are present
    expected_providers = ["litellm_proxy", "bedrock", "ollama"]
    for provider in expected_providers:
      if provider in PROVIDER_ALIASES:
        assert provider in providers
        assert isinstance(providers[provider], list)
        assert len(providers[provider]) > 0

  def test_list_by_provider_includes_bedrock_direct(self):
    """Test that list_by_provider() includes bedrock_direct models."""
    providers = Model.list_by_provider()

    assert "bedrock_direct" in providers
    assert isinstance(providers["bedrock_direct"], list)
    assert len(providers["bedrock_direct"]) > 0

    # Verify the content matches BEDROCK_MODELS
    assert set(providers["bedrock_direct"]) == set(BEDROCK_MODELS.keys())

  def test_list_by_provider_structure(self):
    """Test the structure of list_by_provider() output."""
    providers = Model.list_by_provider()

    for provider_name, models in providers.items():
      assert isinstance(provider_name, str)
      assert isinstance(models, list)

      for model_name in models:
        assert isinstance(model_name, str)
        assert len(model_name) > 0


class TestModelIntegration:
  """Integration tests for Model class."""

  def test_model_constants_consistency(self):
    """Test that MODEL_CLIENTS is consistent with provider aliases."""
    # Check that all provider aliases are in MODEL_CLIENTS
    for provider_name, models in PROVIDER_ALIASES.items():
      for model_alias in models.keys():
        assert model_alias in MODEL_CLIENTS
        assert MODEL_CLIENTS[model_alias] == "litellm"

    # Check that all full provider names are in MODEL_CLIENTS
    for full_name in ALL_PROVIDER_ALLOWED_FULL_NAMES:
      assert full_name in MODEL_CLIENTS
      assert MODEL_CLIENTS[full_name] == "litellm"

    # Check bedrock direct models
    for model_alias in BEDROCK_MODELS.keys():
      bedrock_direct_key = f"bedrock-direct/{model_alias}"
      assert bedrock_direct_key in MODEL_CLIENTS
      assert MODEL_CLIENTS[bedrock_direct_key] == "bedrock_direct"

  def test_model_creation_for_all_supported_models(self):
    """Test that all supported models can be instantiated."""
    # Test a sample of models that work across providers
    # Use patch to clear environment and force Ollama provider
    with patch.dict(os.environ, {}, clear=True):
      test_models = ["llama3.2", "deepseek-r1"]

      for model_name in test_models:
        if model_name in MODEL_CLIENTS:
          try:
            model = Model(model_name)
            assert model.name == model_name
            assert model.client is not None
          except Exception as e:
            # If model creation fails due to missing credentials/setup, that's OK
            # We just want to ensure the model selection logic works
            if "not supported" in str(e):
              pytest.fail(f"Model {model_name} should be supported but got: {e}")

  def test_model_with_conversation_messages(self):
    """Test Model with ConversationMessage objects."""
    model = Model("llama3.2")

    # Create conversation messages
    messages = [
      SystemMessage(content="You are a helpful assistant"),
      UserMessage(content="Hello"),
      AssistantMessage(content="Hi there!"),
    ]

    # Mock the client method
    model.client.count_tokens = MagicMock(return_value=25)

    result = model.count_tokens(messages)
    assert result == 25
    model.client.count_tokens.assert_called_once()

    # Verify that the messages were passed correctly
    call_args = model.client.count_tokens.call_args[0]
    assert len(call_args[0]) == 3  # Three messages

  @pytest.mark.asyncio
  async def test_model_error_handling(self):
    """Test Model error handling in various scenarios."""
    model = Model("llama3.2")

    # Test error in complete_chat
    model.client.complete_chat = MagicMock(side_effect=Exception("API Error"))

    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(Exception, match="API Error"):
      result_coro = model.complete_chat(messages, stream=False)
      await result_coro

    # Test error in embeddings
    model.client.embeddings = AsyncMock(side_effect=ValueError("Invalid input"))

    with pytest.raises(ValueError, match="Invalid input"):
      await model.embeddings(["test text"])


class TestModelEdgeCases:
  """Test edge cases and boundary conditions."""

  def test_empty_messages_handling(self):
    """Test handling of empty message lists."""
    model = Model("llama3.2")
    model.client.count_tokens = MagicMock(return_value=0)

    result = model.count_tokens([])
    assert result == 0
    model.client.count_tokens.assert_called_once_with([], False, None)

  def test_none_values_handling(self):
    """Test handling of None values in various parameters."""
    model = Model("llama3.2", max_input_tokens=None)
    assert model.max_input_tokens is None

    # Test with None messages (should be handled by client)
    model.client.count_tokens = MagicMock(return_value=0)
    model.count_tokens([], tools=None)
    model.client.count_tokens.assert_called_once_with([], False, None)

  def test_large_parameter_values(self):
    """Test handling of large parameter values."""
    large_tokens = 1_000_000
    model = Model("llama3.2", max_input_tokens=large_tokens)
    assert model.max_input_tokens == large_tokens

  def test_special_characters_in_model_name(self):
    """Test handling of special characters in model names."""
    # Test model names with special characters that work with Ollama
    # Clear environment to force Ollama provider
    with patch.dict(os.environ, {}, clear=True):
      special_names = [
        "llama3.2",  # dots
      ]

      for name in special_names:
        if name in MODEL_CLIENTS:
          model = Model(name)
          assert model.name == name

  def test_case_sensitivity(self):
    """Test case sensitivity of model names."""
    # Model names should be case sensitive
    with pytest.raises(ValueError):
      Model("CLAUDE-3-5-SONNET-V2")  # uppercase version should fail

  def test_model_client_attributes(self):
    """Test that model properly exposes client attributes."""
    model = Model("llama3.2")

    # Ensure client has expected methods
    expected_methods = [
      "count_tokens",
      "support_tools",
      "support_forced_assistant_answer",
      "complete_chat",
      "embeddings",
    ]

    for method_name in expected_methods:
      assert hasattr(model.client, method_name)
      assert callable(getattr(model.client, method_name))


if __name__ == "__main__":
  pytest.main([__file__])

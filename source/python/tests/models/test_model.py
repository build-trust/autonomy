import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock

from autonomy.models.model import Model
from autonomy.models.clients.litellm_client import PROVIDER_ALIASES, normalize_messages
from autonomy.nodes.message import UserMessage, AssistantMessage


@pytest.fixture(autouse=True)
def clean_environment():
  """Clean environment variables between tests to prevent pollution."""
  # Store original environment
  original_env = os.environ.copy()
  yield
  # Restore original environment after each test
  os.environ.clear()
  os.environ.update(original_env)


class TestModel:
  def test_model_initialization(self):
    """Test basic model initialization."""
    with patch.dict(os.environ, {"LITELLM_PROXY_API_BASE": "http://127.0.0.1:4000"}):
      model = Model("claude-3-5-sonnet-v2")
      assert model.name == "claude-3-5-sonnet-v2"
      assert model.client is not None

  def test_model_list(self):
    """Test that model listing works."""
    models = Model.list()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "claude-3-5-sonnet-v2" in models

  def test_model_list_by_provider(self):
    """Test provider-based model listing."""
    providers = Model.list_by_provider()
    assert isinstance(providers, dict)
    assert "litellm_proxy" in providers
    assert "bedrock" in providers
    assert "ollama" in providers

  def test_unsupported_model_error(self):
    """Test that unsupported models raise ValueError."""
    with patch.dict(os.environ, {"LITELLM_PROXY_API_BASE": "http://127.0.0.1:4000"}):
      with pytest.raises(ValueError, match="Model 'invalid-model' is not supported"):
        Model("invalid-model")

  def test_provider_detection_litellm_proxy(self):
    """Test provider detection for LiteLLM proxy."""
    with patch.dict(os.environ, {"LITELLM_PROXY_API_BASE": "http://127.0.0.1:4000"}):
      model = Model("claude-3-5-sonnet-v2")
      assert hasattr(model.client, "name")
      assert model.client.name.startswith("litellm_proxy/")

  def test_provider_detection_bedrock(self):
    """Test provider detection for AWS Bedrock."""
    with patch.dict(os.environ, {
      "AWS_WEB_IDENTITY_TOKEN_FILE": "/tmp/token",
      "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/test-role",
      "AWS_DEFAULT_REGION": "us-east-1"
    }):
      model = Model("claude-3-5-sonnet-v2")
      assert hasattr(model.client, "name")
      assert model.client.name.startswith("bedrock/")

  def test_provider_detection_ollama(self):
    """Test provider detection for Ollama (fallback)."""
    with patch.dict(os.environ, {}, clear=True):
      with patch("autonomy.models.clients.litellm_client.LiteLLMClient._has_bedrock_access", return_value=False):
        model = Model("llama3.2")
        assert hasattr(model.client, "name")
        assert model.client.name.startswith("ollama")

  @patch("autonomy.models.clients.litellm_client.router")
  @patch.dict(os.environ, {"LITELLM_PROXY_API_BASE": "http://127.0.0.1:4000"})
  async def test_complete_chat_non_streaming(self, mock_router):
    """Test non-streaming chat completion."""
    # Mock the router response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello! How can I help you?"
    mock_router.acompletion = AsyncMock(return_value=mock_response)

    model = Model("claude-3-5-sonnet-v2")
    messages = [{"role": "user", "content": "Hello"}]

    response = await model.complete_chat(messages)
    assert response is not None
    mock_router.acompletion.assert_called_once()

  @patch.dict(os.environ, {"LITELLM_PROXY_API_BASE": "http://127.0.0.1:4000"})
  def test_token_counting(self):
    """Test token counting functionality."""
    model = Model("claude-3-5-sonnet-v2")
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]

    # Mock litellm token_counter
    with patch("autonomy.models.clients.litellm_client.litellm.token_counter", return_value=10):
      token_count = model.count_tokens(messages)
      assert token_count == 10

  @patch.dict(os.environ, {"LITELLM_PROXY_API_BASE": "http://127.0.0.1:4000"})
  def test_conversation_message_support(self):
    """Test support for ConversationMessage objects."""
    model = Model("claude-3-5-sonnet-v2")

    # Create ConversationMessage objects
    user_msg = UserMessage(content="Hello")
    assistant_msg = AssistantMessage(content="Hi there!")
    messages = [user_msg, assistant_msg]

    # This should not raise an error
    with patch("autonomy.models.clients.litellm_client.litellm.token_counter", return_value=10):
      token_count = model.count_tokens(messages)
      assert token_count == 10

  @patch.dict(os.environ, {"LITELLM_PROXY_API_BASE": "http://127.0.0.1:4000"})
  def test_model_properties(self):
    """Test model property accessors."""
    model = Model("claude-3-5-sonnet-v2")
    assert model.name == "claude-3-5-sonnet-v2"
    assert hasattr(model, "client")
    assert model.resolved_name.endswith("claude-3-5-sonnet-20241022-v2:0")

  @patch.dict(os.environ, {"LITELLM_PROXY_API_BASE": "http://127.0.0.1:4000"})
  def test_tools_support_detection(self):
    """Test tools support detection."""
    # Most models support tools
    model = Model("claude-3-5-sonnet-v2")
    assert model.support_tools() == True

    # DeepSeek models don't support tools
    with patch.dict(os.environ, {}, clear=True):
      with patch("autonomy.models.clients.litellm_client.LiteLLMClient._has_bedrock_access", return_value=False):
        deepseek_model = Model("deepseek-r1")
        assert deepseek_model.support_tools() == False

  def test_forced_assistant_answer_support(self):
    """Test forced assistant answer support detection."""
    # Ollama models support forced assistant answers
    with patch.dict(os.environ, {}, clear=True):
      with patch("autonomy.models.clients.litellm_client.LiteLLMClient._has_bedrock_access", return_value=False):
        model = Model("llama3.2")
        assert model.support_forced_assistant_answer() == True

    # Bedrock models don't support forced assistant answers
    with patch.dict(os.environ, {
      "AWS_WEB_IDENTITY_TOKEN_FILE": "/tmp/token",
      "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/test-role",
      "AWS_DEFAULT_REGION": "us-east-1"
    }):
      model = Model("claude-3-5-sonnet-v2")
      assert model.support_forced_assistant_answer() == False


class TestMessageNormalization:
  def test_normalize_dict_messages(self):
    """Test message normalization with dict messages."""
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]

    normalized = normalize_messages(messages, False, True, True)
    assert len(normalized) == 2
    assert normalized[0]["role"] == "user"
    assert normalized[1]["role"] == "assistant"

  def test_normalize_conversation_messages(self):
    """Test message normalization with ConversationMessage objects."""
    messages = [UserMessage(content="Hello"), AssistantMessage(content="Hi!")]

    normalized = normalize_messages(messages, False, True, True)
    assert len(normalized) == 2
    assert normalized[0]["role"] == "user"
    assert normalized[1]["role"] == "assistant"

  def test_normalize_empty_content_removal(self):
    """Test that messages with empty content are handled properly."""
    messages = [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": ""},
      {"role": "user", "content": "Are you there?"},
    ]

    normalized = normalize_messages(messages, False, True, True)
    # Empty assistant message should be removed
    assert len(normalized) == 2
    assert normalized[0]["content"] == "Hello"
    assert normalized[1]["content"] == "Are you there?"

  def test_normalize_thinking_mode(self):
    """Test message normalization in thinking mode."""
    messages = [{"role": "user", "content": "Solve this problem"}]

    normalized = normalize_messages(messages, True, True, True)
    assert len(normalized) == 1
    assert normalized[0]["content"] == "<think>Solve this problem"

  def test_normalize_tool_support_disabled(self):
    """Test message normalization when tools are not supported."""
    messages = [
      {"role": "user", "content": "Hello"},
      {"role": "tool", "content": "Tool response"},
      {
        "role": "assistant",
        "content": "Response",
        "tool_calls": [{"id": "1", "function": {"name": "test"}}],
      },
    ]

    normalized = normalize_messages(messages, False, False, True)
    # Tool message should be converted to assistant and compacted with the next assistant message
    # Tool calls should be removed
    assert len(normalized) == 2
    assert normalized[0]["role"] == "user"
    assert normalized[1]["role"] == "assistant"
    assert normalized[1]["content"] == "Tool responseResponse"  # compacted
    assert "tool_calls" not in normalized[1]

  def test_normalize_forced_assistant_answer_disabled(self):
    """Test message normalization when forced assistant answers are not supported."""
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]

    normalized = normalize_messages(messages, False, True, False)
    # Last assistant message should be converted to user
    assert len(normalized) == 2
    assert normalized[1]["role"] == "user"

  def test_normalize_message_compaction(self):
    """Test that consecutive messages with same role are compacted."""
    messages = [
      {"role": "user", "content": "Part 1"},
      {"role": "user", "content": " Part 2"},
      {"role": "assistant", "content": "Response"},
    ]

    normalized = normalize_messages(messages, False, True, True)
    assert len(normalized) == 2
    assert normalized[0]["role"] == "user"
    assert normalized[0]["content"] == "Part 1 Part 2"


class TestProviderAliases:
  def test_provider_aliases_structure(self):
    """Test that provider aliases are properly structured."""
    assert isinstance(PROVIDER_ALIASES, dict)
    assert "litellm_proxy" in PROVIDER_ALIASES
    assert "bedrock" in PROVIDER_ALIASES
    assert "ollama" in PROVIDER_ALIASES

    # Check that each provider has model mappings
    for provider, models in PROVIDER_ALIASES.items():
      assert isinstance(models, dict)
      assert len(models) > 0

      # Check that model mappings are strings
      for alias, full_name in models.items():
        assert isinstance(alias, str)
        assert isinstance(full_name, str)
        assert len(alias) > 0
        assert len(full_name) > 0

  def test_claude_models_available(self):
    """Test that Claude models are available in multiple providers."""
    claude_models = ["claude-3-5-haiku-v1", "claude-3-5-sonnet-v1", "claude-3-5-sonnet-v2"]

    for model in claude_models:
      assert model in PROVIDER_ALIASES["litellm_proxy"]
      assert model in PROVIDER_ALIASES["bedrock"]

  def test_llama_models_available(self):
    """Test that Llama models are available in multiple providers."""
    llama_models = ["llama3.2", "llama3.3"]

    for model in llama_models:
      assert model in PROVIDER_ALIASES["litellm_proxy"]
      assert model in PROVIDER_ALIASES["bedrock"]
      assert model in PROVIDER_ALIASES["ollama"]

  def test_embedding_models_available(self):
    """Test that embedding models are available."""
    embedding_models = [
      "embed-english-v3",
      "embed-multilingual-v3",
      "titan-embed-text-v1",
      "nomic-embed-text",
    ]

    for model in embedding_models:
      found_in_provider = False
      for provider_models in PROVIDER_ALIASES.values():
        if model in provider_models:
          found_in_provider = True
          break
      assert found_in_provider, f"Embedding model {model} not found in any provider"

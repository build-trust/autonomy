import pytest
import asyncio
from unittest.mock import patch, MagicMock

from autonomy.models.clients.default import DefaultModelClient
from autonomy.nodes.message import UserMessage, AssistantMessage


class TestDefaultClientInitialization:
  """Test DefaultModelClient initialization and setup."""

  def test_basic_initialization(self):
    """Test basic client initialization."""
    client = DefaultModelClient()
    assert client.name == "default"
    assert client.original_name == "default"
    assert client.max_input_tokens is None
    assert client.kwargs == {}

  def test_initialization_with_name(self):
    """Test client initialization with custom name."""
    client = DefaultModelClient("test-model")
    assert client.name == "test-model"
    assert client.original_name == "test-model"

  def test_initialization_with_max_tokens(self):
    """Test client initialization with max tokens."""
    client = DefaultModelClient("test-model", max_input_tokens=50000)
    assert client.max_input_tokens == 50000

  def test_initialization_with_kwargs(self):
    """Test client initialization with additional kwargs."""
    kwargs = {"temperature": 0.7, "top_p": 0.9, "custom_param": "value"}
    client = DefaultModelClient("test-model", **kwargs)
    assert client.kwargs == kwargs

  def test_initialization_all_params(self):
    """Test client initialization with all parameters."""
    kwargs = {"temperature": 0.5, "max_tokens": 2000}
    client = DefaultModelClient("custom-model", max_input_tokens=100000, **kwargs)
    assert client.name == "custom-model"
    assert client.original_name == "custom-model"
    assert client.max_input_tokens == 100000
    assert client.kwargs == kwargs

  def test_logger_initialization(self):
    """Test that logger is properly initialized."""
    client = DefaultModelClient()
    assert hasattr(client, "logger")
    assert client.logger is not None


class TestDefaultClientTokenCounting:
  """Test DefaultModelClient token counting functionality."""

  def test_count_tokens_dict_messages(self):
    """Test token counting with dictionary messages."""
    client = DefaultModelClient()

    messages = [
      {"role": "user", "content": "Hello world"},  # 11 chars
      {"role": "assistant", "content": "Hi there!"},  # 9 chars
    ]

    # Total: 20 chars, should be 20/4 = 5 tokens
    result = client.count_tokens(messages)
    assert result == 5

  def test_count_tokens_conversation_messages(self):
    """Test token counting with ConversationMessage objects."""
    client = DefaultModelClient()

    messages = [
      UserMessage(content="Hello"),  # 5 chars
      AssistantMessage(content="Hi"),  # 2 chars
    ]

    # Total: 7 chars, should be 7/4 = 1 token (integer division)
    result = client.count_tokens(messages)
    assert result == 1

  def test_count_tokens_mixed_messages(self):
    """Test token counting with mixed message types."""
    client = DefaultModelClient()

    messages = [
      {"role": "user", "content": "Hello"},  # 5 chars
      AssistantMessage(content="Hi there"),  # 8 chars
      UserMessage(content="How are you?"),  # 12 chars
    ]

    # Count tokens - need to handle TextContent objects properly
    result = client.count_tokens(messages)
    assert isinstance(result, int)
    assert result > 0

  def test_count_tokens_empty_messages(self):
    """Test token counting with empty message list."""
    client = DefaultModelClient()
    result = client.count_tokens([])
    assert result == 0

  def test_count_tokens_empty_content(self):
    """Test token counting with empty content."""
    client = DefaultModelClient()

    messages = [
      {"role": "user", "content": ""},
      {"role": "assistant", "content": "Hello"},  # 5 chars
    ]

    result = client.count_tokens(messages)
    assert result == 1  # 5/4 = 1

  def test_count_tokens_with_thinking_mode(self):
    """Test token counting with thinking mode (should work the same)."""
    client = DefaultModelClient()

    messages = [{"role": "user", "content": "Think about this"}]  # 15 chars

    result_normal = client.count_tokens(messages, is_thinking=False)
    result_thinking = client.count_tokens(messages, is_thinking=True)

    # Should be the same since DefaultClient doesn't handle thinking differently
    assert result_normal == result_thinking
    assert isinstance(result_normal, int)
    assert result_normal > 0

  def test_count_tokens_with_tools(self):
    """Test token counting with tools (should work the same)."""
    client = DefaultModelClient()

    messages = [{"role": "user", "content": "Use tools"}]  # 9 chars
    tools = [{"name": "test_tool"}]  # Should not affect count in default implementation

    result = client.count_tokens(messages, tools=tools)
    assert result == 2  # 9/4 = 2

  def test_count_tokens_missing_content(self):
    """Test token counting with missing content field."""
    client = DefaultModelClient()

    messages = [
      {"role": "user"},  # No content field
      {"role": "assistant", "content": "Hello"},  # 5 chars
    ]

    result = client.count_tokens(messages)
    assert result == 1  # 5/4 = 1

  def test_count_tokens_none_content(self):
    """Test token counting with None content."""
    client = DefaultModelClient()

    messages = [
      {"role": "user", "content": None},
      {"role": "assistant", "content": "Hello"},  # 5 chars
    ]

    result = client.count_tokens(messages)
    assert result == 1  # 5/4 = 1

  def test_count_tokens_non_string_content(self):
    """Test token counting with non-string content."""
    client = DefaultModelClient()

    # Mock a content object with text attribute
    mock_content = MagicMock()
    mock_content.text = "Hello world"  # 11 chars

    messages = [{"role": "user", "content": mock_content}]

    result = client.count_tokens(messages)
    assert result == 2  # 11/4 = 2

  def test_count_tokens_large_content(self):
    """Test token counting with large content."""
    client = DefaultModelClient()

    large_content = "A" * 10000  # 10000 chars
    messages = [{"role": "user", "content": large_content}]

    result = client.count_tokens(messages)
    assert result == 2500  # 10000/4 = 2500

  def test_count_tokens_unicode_content(self):
    """Test token counting with Unicode content."""
    client = DefaultModelClient()

    unicode_content = "Hello 世界! 🌍"  # Mix of ASCII, Chinese, and emoji
    messages = [{"role": "user", "content": unicode_content}]

    result = client.count_tokens(messages)
    assert isinstance(result, int)
    assert result >= 0


class TestDefaultClientCapabilities:
  """Test DefaultModelClient capability methods."""

  def test_support_tools_false(self):
    """Test that DefaultClient doesn't support tools."""
    client = DefaultModelClient()
    assert client.support_tools() is False

  def test_support_forced_assistant_answer_false(self):
    """Test that DefaultClient doesn't support forced assistant answers."""
    client = DefaultModelClient()
    assert client.support_forced_assistant_answer() is False


class TestDefaultClientChatCompletion:
  """Test DefaultModelClient chat completion functionality."""

  @pytest.mark.asyncio
  async def test_complete_chat_non_streaming(self):
    """Test non-streaming chat completion."""
    client = DefaultModelClient()

    with patch.object(client, "logger") as mock_logger:
      messages = [{"role": "user", "content": "Hello"}]
      result_coro = client.complete_chat(messages, stream=False)
      result = await result_coro

      # Should return a mock response
      assert hasattr(result, "choices")
      assert len(result.choices) == 1
      assert result.choices[0].message.role == "assistant"
      assert "DefaultModelClient" in result.choices[0].message.content

      # Should log a warning
      mock_logger.warning.assert_called_once()
      assert "DefaultModelClient.complete_chat called" in mock_logger.warning.call_args[0][0]

  @pytest.mark.asyncio
  async def test_complete_chat_streaming(self):
    """Test streaming chat completion."""
    client = DefaultModelClient()

    with patch.object(client, "logger") as mock_logger:
      messages = [{"role": "user", "content": "Hello"}]
      result_stream = client.complete_chat(messages, stream=True)

      # Collect all chunks
      chunks = []
      async for chunk in result_stream:
        chunks.append(chunk)

      # Should return at least 2 chunks (content + finish)
      assert len(chunks) == 2

      # First chunk should have content
      assert hasattr(chunks[0], "choices")
      assert chunks[0].choices[0].delta.content is not None

      # Last chunk should have finish_reason
      assert chunks[1].choices[0].finish_reason == "stop"

      # Should log a warning
      mock_logger.warning.assert_called_once()

  @pytest.mark.asyncio
  async def test_complete_chat_with_parameters(self):
    """Test chat completion with various parameters."""
    client = DefaultModelClient()

    messages = [{"role": "user", "content": "Hello"}]
    result_coro = client.complete_chat(messages, stream=False, is_thinking=True, temperature=0.7, max_tokens=100)
    result = await result_coro

    # Should still work regardless of parameters
    assert hasattr(result, "choices")
    assert len(result.choices) == 1

  @pytest.mark.asyncio
  async def test_complete_chat_empty_messages(self):
    """Test chat completion with empty messages."""
    client = DefaultModelClient()

    result_coro = client.complete_chat([], stream=False)
    result = await result_coro

    assert hasattr(result, "choices")
    assert len(result.choices) == 1

  @pytest.mark.asyncio
  async def test_complete_chat_conversation_messages(self):
    """Test chat completion with ConversationMessage objects."""
    client = DefaultModelClient()

    messages = [UserMessage(content="Hello"), AssistantMessage(content="Hi there!")]

    result_coro = client.complete_chat(messages, stream=False)
    result = await result_coro

    assert hasattr(result, "choices")
    assert len(result.choices) == 1


class TestDefaultClientEmbeddings:
  """Test DefaultModelClient embeddings functionality."""

  @pytest.mark.asyncio
  async def test_embeddings_basic(self):
    """Test basic embeddings functionality."""
    client = DefaultModelClient()

    with patch.object(client, "logger") as mock_logger:
      texts = ["Hello", "World"]
      result = await client.embeddings(texts)

      # Should return zero vectors of dimension 1024
      assert isinstance(result, list)
      assert len(result) == 2
      assert len(result[0]) == 1024
      assert len(result[1]) == 1024
      assert all(x == 0.0 for x in result[0])
      assert all(x == 0.0 for x in result[1])

      # Should log a warning
      mock_logger.warning.assert_called_once()
      assert "DefaultModelClient.embeddings called" in mock_logger.warning.call_args[0][0]

  @pytest.mark.asyncio
  async def test_embeddings_single_text(self):
    """Test embeddings with single text."""
    client = DefaultModelClient()

    result = await client.embeddings(["Hello world"])

    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result[0]) == 1024
    assert all(x == 0.0 for x in result[0])

  @pytest.mark.asyncio
  async def test_embeddings_empty_list(self):
    """Test embeddings with empty text list."""
    client = DefaultModelClient()

    result = await client.embeddings([])

    assert isinstance(result, list)
    assert len(result) == 0

  @pytest.mark.asyncio
  async def test_embeddings_with_kwargs(self):
    """Test embeddings with additional kwargs."""
    client = DefaultModelClient()

    texts = ["Hello"]
    result = await client.embeddings(texts, model="custom-embed", dimensions=512)

    # Should still return 1024-dimensional vectors (default implementation)
    assert len(result[0]) == 1024

  @pytest.mark.asyncio
  async def test_embeddings_many_texts(self):
    """Test embeddings with many texts."""
    client = DefaultModelClient()

    texts = [f"Text {i}" for i in range(100)]
    result = await client.embeddings(texts)

    assert len(result) == 100
    assert all(len(vec) == 1024 for vec in result)
    assert all(all(x == 0.0 for x in vec) for vec in result)


class TestDefaultClientMessagePreparation:
  """Test DefaultModelClient message preparation."""

  def test_prepare_llm_call_dict_messages(self):
    """Test prepare_llm_call with dictionary messages."""
    client = DefaultModelClient()

    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    kwargs = {"temperature": 0.7}

    result_messages, result_kwargs = client.prepare_llm_call(messages, is_thinking=False, **kwargs)

    assert result_messages == messages  # Should be unchanged
    assert result_kwargs == kwargs

  def test_prepare_llm_call_conversation_messages(self):
    """Test prepare_llm_call with ConversationMessage objects."""
    client = DefaultModelClient()

    messages = [UserMessage(content="Hello"), AssistantMessage(content="Hi there!")]

    result_messages, result_kwargs = client.prepare_llm_call(messages, is_thinking=True)

    # Should convert ConversationMessage to dict
    assert isinstance(result_messages, list)
    assert len(result_messages) == 2
    assert all(isinstance(msg, dict) for msg in result_messages)
    assert result_messages[0]["role"] == "user"
    # Content might be a TextContent object, so check the actual content
    content = result_messages[0]["content"]
    if hasattr(content, "text"):
      assert content.text == "Hello"
    else:
      assert content == "Hello"

  def test_prepare_llm_call_mixed_messages(self):
    """Test prepare_llm_call with mixed message types."""
    client = DefaultModelClient()

    messages = [{"role": "user", "content": "Hello"}, AssistantMessage(content="Hi there!")]

    result_messages, result_kwargs = client.prepare_llm_call(messages, is_thinking=False)

    # Should convert all to dict format
    assert isinstance(result_messages, list)
    assert len(result_messages) == 2

  def test_prepare_llm_call_empty_messages(self):
    """Test prepare_llm_call with empty messages."""
    client = DefaultModelClient()

    result_messages, result_kwargs = client.prepare_llm_call([], is_thinking=False)

    assert result_messages == []
    assert result_kwargs == {}

  def test_prepare_llm_call_with_kwargs(self):
    """Test prepare_llm_call with various kwargs."""
    client = DefaultModelClient()

    messages = [{"role": "user", "content": "Hello"}]
    kwargs = {"temperature": 0.8, "max_tokens": 1000, "top_p": 0.9, "custom_param": "value"}

    result_messages, result_kwargs = client.prepare_llm_call(messages, is_thinking=True, **kwargs)

    assert result_messages == messages
    assert result_kwargs == kwargs


class TestDefaultClientEdgeCases:
  """Test edge cases and error conditions."""

  def test_malformed_conversation_message(self):
    """Test handling of malformed ConversationMessage."""
    client = DefaultModelClient()

    # Mock a ConversationMessage without content attribute
    mock_message = MagicMock()
    mock_message.role.value = "user"
    del mock_message.content  # Remove content attribute

    messages = [mock_message]

    # Should handle gracefully and return empty content
    result_messages, _ = client.prepare_llm_call(messages, is_thinking=False)
    assert len(result_messages) == 1
    assert result_messages[0]["content"] == ""

  def test_none_messages_input(self):
    """Test handling of None messages input."""
    client = DefaultModelClient()

    # Should handle None input gracefully
    try:
      result = client.count_tokens(None)
      # If it doesn't crash, result should be reasonable
      assert isinstance(result, int)
    except (TypeError, AttributeError):
      # It's acceptable to raise an error for None input
      pass

  def test_very_large_message_count(self):
    """Test handling of very large number of messages."""
    client = DefaultModelClient()

    # Create many short messages
    messages = [{"role": "user", "content": "Hi"}] * 1000

    result = client.count_tokens(messages)
    assert isinstance(result, int)
    assert result == 500  # 1000 * 2 chars / 4 = 500

  def test_unicode_and_special_characters(self):
    """Test handling of Unicode and special characters."""
    client = DefaultModelClient()

    messages = [
      {"role": "user", "content": "Hello 世界! 🚀 Special chars: <>&\"'"},
      {"role": "assistant", "content": "Ñoño €£¥ §±"},
    ]

    result = client.count_tokens(messages)
    assert isinstance(result, int)
    assert result > 0

  @pytest.mark.asyncio
  async def test_concurrent_operations(self):
    """Test that operations can be run concurrently."""
    client = DefaultModelClient()

    async def run_operation():
      messages = [{"role": "user", "content": "Hello"}]
      result = await client.complete_chat(messages, stream=False)
      return result

    # Run multiple operations concurrently
    tasks = [run_operation() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # All should complete successfully
    assert len(results) == 10
    assert all(hasattr(r, "choices") for r in results)

  @pytest.mark.asyncio
  async def test_embeddings_concurrent(self):
    """Test concurrent embeddings operations."""
    client = DefaultModelClient()

    async def get_embeddings(text):
      return await client.embeddings([text])

    # Run multiple embedding requests concurrently
    tasks = [get_embeddings(f"Text {i}") for i in range(5)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 5
    assert all(len(r) == 1 and len(r[0]) == 1024 for r in results)

  def test_inheritance_compatibility(self):
    """Test that DefaultClient can be properly subclassed."""

    class CustomDefaultClient(DefaultModelClient):
      def support_tools(self):
        return True

      def count_tokens(self, messages, is_thinking=False, tools=None):
        return 42

    custom_client = CustomDefaultClient("custom")
    assert custom_client.support_tools() is True
    assert custom_client.count_tokens([]) == 42
    assert custom_client.name == "custom"


class TestDefaultClientMockObjects:
  """Test the internal mock objects used by DefaultClient."""

  @pytest.mark.asyncio
  async def test_mock_response_structure(self):
    """Test the structure of mock response objects."""
    client = DefaultModelClient()

    messages = [{"role": "user", "content": "Hello"}]
    result = await client.complete_chat(messages, stream=False)

    # Test response structure
    assert hasattr(result, "choices")
    assert len(result.choices) == 1

    choice = result.choices[0]
    assert hasattr(choice, "message")
    assert hasattr(choice, "finish_reason")

    message = choice.message
    assert hasattr(message, "role")
    assert hasattr(message, "content")
    assert hasattr(message, "reasoning_content")
    assert message.role == "assistant"
    assert isinstance(message.content, str)
    assert message.reasoning_content is None

  @pytest.mark.asyncio
  async def test_mock_streaming_structure(self):
    """Test the structure of mock streaming objects."""
    client = DefaultModelClient()

    messages = [{"role": "user", "content": "Hello"}]
    result_stream = client.complete_chat(messages, stream=True)

    chunks = []
    async for chunk in result_stream:
      chunks.append(chunk)

    assert len(chunks) == 2

    # First chunk (content)
    content_chunk = chunks[0]
    assert hasattr(content_chunk, "choices")
    assert hasattr(content_chunk.choices[0], "delta")
    assert hasattr(content_chunk.choices[0].delta, "content")
    assert content_chunk.choices[0].delta.content is not None

    # Final chunk (finish)
    finish_chunk = chunks[1]
    assert hasattr(finish_chunk, "choices")
    assert hasattr(finish_chunk.choices[0], "finish_reason")
    assert finish_chunk.choices[0].finish_reason == "stop"


if __name__ == "__main__":
  pytest.main([__file__])

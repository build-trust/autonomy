from typing import List, Optional
from ...logs import get_logger, InfoContext, DebugContext
from ...nodes.message import ConversationMessage


class DefaultModelClient(InfoContext, DebugContext):
  """
  Default model client that serves as a fallback implementation.

  This client provides basic functionality and can be used as a template
  for implementing custom model clients.
  """

  def __init__(self, name: str = "default", max_input_tokens: Optional[int] = None, **kwargs):
    self.logger = get_logger("model")
    self.name = name
    self.original_name = name
    self.max_input_tokens = max_input_tokens
    self.kwargs = kwargs

  def count_tokens(
    self, messages: List[dict] | List[ConversationMessage], is_thinking: bool = False, tools=None
  ) -> int:
    """
    Basic token counting implementation.
    This is a rough estimation - actual implementations should use proper tokenizers.
    """
    total_chars = 0

    # Convert ConversationMessage to dict if needed
    if messages and not isinstance(messages[0], dict):
      messages = [
        {
          "role": message.role.value,
          "content": getattr(message, "content", ""),
        }
        for message in messages
      ]

    for message in messages:
      if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
          total_chars += len(content)
        elif hasattr(content, "text"):
          # Handle TextContent objects
          total_chars += len(content.text)

    # Rough estimation: 1 token ≈ 4 characters
    return total_chars // 4

  def support_tools(self) -> bool:
    """Check if the model supports tools/function calling."""
    return False

  def support_forced_assistant_answer(self) -> bool:
    """Check if the model supports forced assistant answers."""
    return False

  def complete_chat(
    self,
    messages: List[dict] | List[ConversationMessage],
    stream: bool = False,
    is_thinking: bool = False,
    **kwargs,
  ):
    """
    Default implementation that returns a simple response.

    Real implementations should call actual model APIs.
    """
    self.logger.warning(
      f"DefaultModelClient.complete_chat called with {len(messages)} messages. "
      "This is a fallback implementation that returns mock responses."
    )

    # Create a mock response structure
    class MockChoice:
      def __init__(self, finish_reason=None):
        self.message = MockMessage()
        self.finish_reason = finish_reason
        self.delta = MockDelta()

    class MockDelta:
      def __init__(self, content=None):
        self.content = content

    class MockMessage:
      def __init__(self):
        self.role = "assistant"
        self.content = "This is a default response from the DefaultModelClient. Please configure a proper model client for actual functionality."
        self.reasoning_content = None

    class MockResponse:
      def __init__(self, finish_reason=None, content=None):
        self.choices = [MockChoice(finish_reason)]
        if content is not None:
          self.choices[0].delta.content = content

    if stream:
      # For streaming, return an async generator
      async def mock_stream():
        # First chunk with content
        yield MockResponse(
          content="This is a default response from the DefaultModelClient. Please configure a proper model client for actual functionality."
        )
        # Final chunk with finish_reason
        yield MockResponse(finish_reason="stop")

      return mock_stream()
    else:

      async def _complete():
        return MockResponse()

      return _complete()

  async def embeddings(self, text: List[str], **kwargs) -> List[List[float]]:
    """
    Default embeddings implementation.

    Returns zero vectors - real implementations should call embedding APIs.
    """
    self.logger.warning(
      f"DefaultModelClient.embeddings called with {len(text)} texts. "
      "This is a fallback implementation that returns zero vectors."
    )

    # Return zero vectors of dimension 1024 (common embedding size)
    return [[0.0] * 1024 for _ in text]

  def prepare_llm_call(self, messages: List[dict] | List[ConversationMessage], is_thinking: bool, **kwargs):
    """
    Prepare messages for LLM call.

    Basic implementation that just converts ConversationMessage to dict.
    """
    if messages and not isinstance(messages[0], dict):
      messages = [
        {
          "role": message.role.value,
          "content": getattr(message, "content", ""),
        }
        for message in messages
      ]

    return messages, kwargs

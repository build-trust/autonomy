from typing import Optional
from ..tools.protocol import InvokableTool
from .protocol import KnowledgeProvider


class KnowledgeTool(InvokableTool):
  """
  Tool that allows agents to search knowledge bases.
  Wraps a KnowledgeProvider for tool-based access.
  """

  def __init__(self, knowledge: KnowledgeProvider, name: str = "search_knowledge"):
    self.knowledge = knowledge
    self.name = name
    self._spec = {
      "type": "function",
      "function": {
        "name": name,
        "description": "Search the knowledge base for relevant information",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string", "description": "The search query to find relevant information"}},
          "required": ["query"],
        },
      },
    }

  async def spec(self) -> dict:
    """Return tool specification for the LLM."""
    return self._spec

  async def invoke(self, json_argument: Optional[str]) -> str:
    """Execute knowledge search."""
    import json

    # Parse arguments
    if not json_argument:
      return "Error: No query provided"

    try:
      args = json.loads(json_argument)
      query = args.get("query", "")
    except (json.JSONDecodeError, AttributeError):
      return "Error: Invalid arguments"

    if not query:
      return "Error: No query provided"

    # Execute knowledge search
    # Note: We don't have scope/conversation in the invoke signature
    # so we pass None for both - knowledge providers should handle this
    result = await self.knowledge.search_knowledge(None, None, query)
    if result:
      return result
    return "No relevant information found"

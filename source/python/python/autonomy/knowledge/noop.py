from typing import Optional

from .protocol import KnowledgeProvider


class NoopKnowledge(KnowledgeProvider):
  async def search_knowledge(self, scope: Optional[str], conversation: Optional[str], query: str) -> Optional[str]:
    return None

from autonomy import Agent, HttpServer, Model, Node
from fastapi import FastAPI, HTTPException, status, Security
from fastapi.security.api_key import APIKeyHeader
from asyncio import gather, create_task
from dataclasses import dataclass
from os import environ
from typing import List, Dict


async def analyze_item(node: Node, item: str) -> Dict[str, str]:
  agent = None
  try:
    agent = await Agent.start(
      node=node,
      name=f"translator_{id(item)}",
      instructions="""
        You are an agent that specializes in translating text
        from English to Hindi. When you're given text in English,
        output the corresponding translation in Hindi written
        using the Latin alphabet (Roman script).

        Provide only the translation, no additional explanation.
      """,
      model=Model("claude-sonnet-4-v1"),
    )

    response = await agent.send(f"English:\n\n{item}", timeout=60)
    return {"item": item, "analysis": response[-1].content.text}
  except Exception as e:
    return {"item": item, "error": str(e)}
  finally:
    if agent:
      create_task(Agent.stop(node, agent.name))


async def analyze(node: Node, items: List[str]) -> List[Dict[str, str]]:
  return await gather(*(analyze_item(node, item) for item in items))


class App:
  def __init__(self):
    self.api = FastAPI()

  def routes(self, node: Node):
    app = self.api
    api_key_env = environ["API_KEY"]
    api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

    def key(api_key_header: str = Security(api_key_header)) -> str:
      if api_key_header == api_key_env:
        return api_key_header
      raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Please provide a valid API key.",
      )

    @dataclass
    class GetAnalysesRequest:
      items: List[str]

    @dataclass
    class GetAnalysesResponse:
      analyses: List[Dict[str, str]]

    @app.post("/analyses")
    async def get_analyses(
      request: GetAnalysesRequest, key: str = Security(key)
    ) -> GetAnalysesResponse:
      analyses = await analyze(node, request.items)
      return GetAnalysesResponse(analyses=analyses)


Node.start(http_server=HttpServer(api=App()))

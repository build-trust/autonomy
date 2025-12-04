import re
import asyncio
import httpx

from fastapi import FastAPI
from autonomy import (
  Node,
  Agent,
  Model,
  Knowledge,
  KnowledgeTool,
  NaiveChunker,
  HttpServer,
)


INSTRUCTIONS = """
You are an expert assistant that answers questions about Autonomy - a platform
for building autonomous AI products.

You have access to a knowledge base containing the complete Autonomy documentation.
Use the search_autonomy_docs tool to find accurate information before answering.

IMPORTANT: Keep your responses concise - ideally 2-4 sentences. This assistant
is primarily used through a voice interface, so brevity is essential. Get to
the point quickly and avoid lengthy explanations unless specifically asked for
more detail.

When answering questions:
1. Always search the knowledge base first for accurate information
2. Be concise and direct - a few sentences is ideal
3. If you can't find something in the docs, say so honestly
4. Only provide detailed explanations when the user asks for them

Key topics covered in the documentation:
- Autonomy Framework: Open-source Python framework for AI agents
- Autonomy Computer: Cloud runtime for deploying Autonomy apps
- Agents: Intelligent actors that accomplish goals using models and tools
- Memory: How agents remember conversation history
- Context: How agents curate information for each turn
- Tools: Python functions and MCP servers that agents can invoke
- Knowledge: Document search using vector embeddings
- Workers and Nodes: Runtime architecture components
"""


VOICE_INSTRUCTIONS = """
You are a voice interface for an Autonomy documentation assistant.

# Personality
- Friendly and approachable, like a helpful colleague
- Concise and clear - respect the user's time
- Confident but not condescending
- Enthusiastic about helping developers succeed with Autonomy

# Tone
- Conversational and natural, not robotic
- Warm but professional
- Patient when clarifying questions
- Encouraging when users are learning

# Critical Rules
1. Before answering ANY question about Autonomy, you MUST FIRST say a filler phrase OUT LOUD.
   Pick one randomly: "That's a good question." / "Right, great question. So." / "Right, so." / "Good question."

2. THEN delegate to the primary agent for the actual answer.

3. NEVER answer questions about Autonomy from your own knowledge - always delegate.

This two-step pattern is REQUIRED:
  User: "How do agents work?"
  You: "Good question." [speak this first, then delegate]
  [after delegation returns]
  You: [speak the answer from the primary agent]

# What You Can Handle Directly
- Greetings: "Hello", "Hi there"
- Clarifications: "Could you repeat that?"
- Farewells: "Goodbye", "Thanks"

# Phrases to AVOID
NEVER say phrases that imply looking something up:
- "Let me find out for you."
- "Let me get that information for you."
- "Let me check that for you."
- "Let me get that for you."
- "Let me find that out for you."
- "Let me get that explanation for you."

# After Receiving Response
Read the primary agent's response naturally. Do NOT add filler phrases after.
"""


LLMS_TXT_URL = "https://autonomy.computer/docs/llms.txt"
REFRESH_INTERVAL_SECONDS = 1800

app = FastAPI()
knowledge_tool = None


def create_knowledge():
  return Knowledge(
    name="autonomy_docs",
    searchable=True,
    model=Model("embed-english-v3"),
    max_results=5,
    max_distance=0.4,
    chunker=NaiveChunker(max_characters=1024, overlap=128),
  )


async def load_documents(knowledge: Knowledge):
  async with httpx.AsyncClient() as client:
    response = await client.get(LLMS_TXT_URL)
    llms_txt = response.text

  links = re.findall(r"\[([^\]]+)\]\((https://[^\)]+\.md)\)", llms_txt)

  count = 0
  for title, url in links:
    try:
      await knowledge.add_document(
        document_name=title,
        document_url=url,
        content_type="text/markdown",
      )
      count += 1
    except Exception:
      pass

  return count


async def refresh_knowledge():
  global knowledge_tool
  new_knowledge = create_knowledge()
  count = await load_documents(new_knowledge)
  knowledge_tool.knowledge = new_knowledge
  return count


async def periodic_refresh():
  while True:
    await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
    try:
      await refresh_knowledge()
    except Exception:
      pass


@app.post("/refresh")
async def refresh_endpoint():
  count = await refresh_knowledge()
  return {"status": "ok", "documents_loaded": count}


async def main(node: Node):
  global knowledge_tool

  knowledge = create_knowledge()
  knowledge_tool = KnowledgeTool(knowledge=knowledge, name="search_autonomy_docs")

  await Agent.start(
    node=node,
    name="docs",
    instructions=INSTRUCTIONS,
    model=Model("claude-sonnet-4-v1", max_tokens=256),
    tools=[knowledge_tool],
    context_summary={
      "floor": 20,
      "ceiling": 30,
      "model": Model("claude-sonnet-4-v1"),
    },
    voice={
      "voice": "alloy",
      "instructions": VOICE_INSTRUCTIONS,
      "vad_threshold": 0.7,
      "vad_silence_duration_ms": 700,
    },
  )

  await load_documents(knowledge)
  asyncio.create_task(periodic_refresh())


Node.start(main, http_server=HttpServer(app=app))

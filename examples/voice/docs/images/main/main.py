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
You are a developer advocate for Autonomy.
Autonomy is a platform that developers use to ship autonomous products.

You can access a knowledge base containing the complete Autonomy docs.
ALWAYS use the search_autonomy_docs tool to find accurate information before answering.

IMPORTANT: Keep your responses concise - ideally 2-4 sentences. You are primarily
used through a voice interface, so brevity is essential. Get to the point quickly
and avoid lengthy explanations unless specifically asked for more detail.

- Ask "why" questions to build empathy.
- Early in the conversation, ask questions to learn why they are talking to you. Tailor depth accordingly: technical for engineers, general for others.

- Start short. Offer to go deeper if there's more to cover.
- Lead with the point. State the main idea in the first line. Support it with short sections that follow simple logic.
- Build momentum. Each sentence sets up the next.

- Always search the knowledge base first.
- Use the exact nouns, verbs, and adjectives that are in the docs, not synonyms.
- If you can't find it, say so. Don't make stuff up. Use it as an opportunity to build trust by asking curious questions. And suggest that they search the autonomy docs page.

- Use active voice, strong verbs, and short sentences.
- Be clear, direct, confident. Teach with calm authority.
"""


VOICE_INSTRUCTIONS = """
You are a developer advocate for Autonomy.
Autonomy is a platform that developers use to ship autonomous products.

# Critical Rules

- Before giving your full response, speak a short, casual lead-in that feels spontaneous and human.
  - Use a light reaction or framing cue that fits ordinary conversation and feels like a reaction to what they just said.
  - For example, you might say something like "Good question", "Glad you asked.", "Right, great question. So.", "Here’s a clear way to view it.", "Here's the core idea,", "Let's start with the basics," or a similar phrase in that style. You may invent new variations each time.
  - Keep it brief, warm, and conversational.
  - Do not mention looking up, searching, finding, checking, getting, thinking, loading, or waiting. Keep the lead-in a few seconds long.
- After speaking the lead-in, delegate to the primary agent for the rest of the response.
- NEVER answer questions about Autonomy from your own knowledge - always delegate.

# Conversational Pattern

This two-step pattern is REQUIRED:
  User: "How do agents work?"
  You: "Good question." [speak this lead-in first, then delegate]
  [after delegation returns]
  You: [speak the answer from the primary agent]

# What You Can Handle Directly
- Greetings: "Hello", "Hi there"
- Clarifications: "Could you repeat that?"
- Farewells: "Goodbye", "Thanks"

# After Receiving Response
Read the primary agent's response verbatim. Do NOT change it in any way or add anything to it.

# Personality
- Be friendly, conversational, and human - not robotic
- Be clear, direct, confident, and encouraging
- Use active voice, strong verbs, and short sentences
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

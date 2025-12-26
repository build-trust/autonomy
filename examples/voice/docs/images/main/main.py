import re
import asyncio
import httpx
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autonomy import (
    Agent,
    HttpServer,
    Knowledge,
    KnowledgeTool,
    Model,
    NaiveChunker,
    Node,
)


INDEX_URL = "https://autonomy.computer/docs/llms.txt"
REFRESH_INTERVAL_SECONDS = 1800


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

## Speak as the Expert
- You ARE the expert
- Never say "the docs describe" or "according to the documentation"
- Instead of "The docs say agents handle deep work", say "Agents handle deep work".

## Voice-Friendly Formatting

Your responses will be read aloud. Format for natural speech:
- Use periods and commas for pacing, not dashes or semicolons
- Write "for example" instead of "e.g."
- Write "that is" instead of "i.e."
- Avoid parenthetical asides - restructure as separate sentences
- No bullet points or numbered lists - use flowing prose
- Spell out abbreviations on first use
"""


VOICE_INSTRUCTIONS = """
You are a developer advocate for Autonomy.
Autonomy is a platform that developers use to ship autonomous products.

# Critical Rules

- Before delegating, speak a SHORT lead-in (1-8 words max) that acknowledges the user.
  - Examples of good lead-ins: "Sure, here's what sub agents are ...", "Right, so here's how actors work ...", "Great question! Here's what tools are ...", "So the way that works is...", "Great question!", "Good question!", "Glad you asked ...", "Right, great question. So ...", "Right, so ...", "Okay, so ...", "Deep work agents, right. So...", "Memory, here's how it works ..."
  - These lead-ins create a natural conversational flow while the primary agent retrieves information
  - NEVER say: "Let me...", "Let me get/explain/break that down"
  - NEVER use phrases that imply fetching, thinking, or processing
- Delegate immediately after the brief acknowledgment.
- NEVER answer questions about Autonomy from your own knowledge - always delegate.

# Conversational Pattern

This two-step pattern is REQUIRED:
  User: "How do agents work?"
  You: "Great question!" [speak this lead-in first, then delegate]
  [after delegation returns]
  You: [speak the answer from the primary agent]

# Continuation Pattern

If the primary agent's response ends with an offer to go deeper (e.g., "Want me to elaborate?"),
speak that offer naturally. If the user says yes, delegate again for more detail.

# What You Can Handle Directly
- Greetings: "Hello", "Hi there"
- Clarifications: "Could you repeat that?"
- Farewells: "Goodbye", "Thanks"
- User says "yes" / "tell me more" after continuation offer → delegate again

# After Receiving Response
Read the primary agent's response VERBATIM and IN FULL.
Do NOT truncate, summarize, or modify it in any way.

# Personality
- Be friendly but minimal - get to the point fast
- Be direct and confident
- Short acknowledgments, then let the content speak
"""

DELEGATION_INSTRUCTIONS = """
Keep all responses brief and concise - 2-4 sentences maximum.
Use conversational flow (First... Then... Also...) instead of bullet lists.
Use semantic search to find accurate information.
Format for voice: no dashes, no parentheses, no lists. Use simple punctuation.
"""


app = FastAPI()

# Add CORS middleware to allow requests from documentation sites
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://autonomy.computer",
        "https://docs.autonomy.computer",
        "https://*.mintlify.dev",
        "https://*.mintlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

knowledge_tool = None


def create_knowledge():
    return Knowledge(
        name="autonomy_docs",
        searchable=True,
        model=Model("embed-english-v3"),
        max_results=10,
        max_distance=0.4,
        max_knowledge_size=8192,
        chunker=NaiveChunker(max_characters=800, overlap=100),
    )


async def load_docs(knowledge: Knowledge):
    print(f"[DOCS] Starting download from {INDEX_URL}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(INDEX_URL)
            llms_txt = response.text
        print(f"[DOCS] Fetched index ({len(llms_txt)} chars)")
    except Exception as e:
        print(f"[DOCS] ERROR fetching index: {e}")
        raise

    links = re.findall(r"\[([^\]]+)\]\((https://[^\)]+\.md)\)", llms_txt)

    count = 0
    print(f"[DOCS] Found {len(links)} doc links to download")

    for title, url in links:
        try:
            await knowledge.add_document(
                document_name=title,
                document_url=url,
                content_type="text/markdown",
            )
            count += 1
        except Exception as e:
            print(f"[DOCS] ERROR loading '{title}': {e}")

    print(f"[DOCS] Successfully loaded {count} documents into knowledge base")
    return count


async def refresh_knowledge():
    global knowledge_tool

    new_knowledge = create_knowledge()
    count = await load_docs(new_knowledge)
    knowledge_tool.knowledge = new_knowledge
    return count


@app.post("/refresh")
async def refresh_endpoint():
    count = await refresh_knowledge()
    return {"status": "ok", "documents_loaded": count}


async def refresh_periodically():
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        try:
            await refresh_knowledge()
        except Exception:
            pass


async def main(node: Node):
    global knowledge_tool

    knowledge = create_knowledge()
    knowledge_tool = KnowledgeTool(knowledge=knowledge, name="search_autonomy_docs")

    await Agent.start(
        node=node,
        name="docs",
        instructions=INSTRUCTIONS,
        model=Model("claude-sonnet-4-5", max_tokens=256),
        tools=[knowledge_tool],
        context_summary={
            "floor": 20,
            "ceiling": 30,
            "model": Model("claude-sonnet-4-5"),
        },
        voice={
            "voice": "shimmer",
            "instructions": VOICE_INSTRUCTIONS,
            "vad_threshold": 0.7,
            "vad_silence_duration_ms": 700,
            "delegation_instructions": DELEGATION_INSTRUCTIONS,
        },
    )

    await load_docs(knowledge)
    asyncio.create_task(refresh_periodically())


Node.start(main, http_server=HttpServer(app=app))

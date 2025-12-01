"""
Autonomy Documentation Voice Assistant

This example demonstrates a voice agent that answers questions about Autonomy
by searching a knowledge base built from the Autonomy documentation.

Architecture:
    User ←→ VoiceAgent (gpt-4o-realtime, fast) ←→ Audio
                  ↓ (delegate_to_primary)
           Primary Agent (claude-sonnet, has knowledge tool)
                  ↓
           Knowledge Base (Autonomy docs)

The knowledge base is populated from individual docs listed at:
    https://autonomy.computer/docs/llms.txt

Usage:
    # Run locally
    cd autonomy/examples/voice/docs/images/main
    AUTONOMY_USE_DIRECT_BEDROCK=1 \
    AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
        uv run --active main.py

    # Or deploy with autonomy command
    cd autonomy/examples/voice/docs
    autonomy --rm

    # Connect via browser
    Open http://localhost:8000/ or http://localhost:32100/
"""

import asyncio
import os

from autonomy import Node, Agent, Model, Knowledge, KnowledgeTool, NaiveChunker


# Primary agent instructions
PRIMARY_AGENT_INSTRUCTIONS = """
You are an expert on Autonomy - a platform for building autonomous AI products.

You have access to a knowledge base containing the complete Autonomy documentation.
Use the search_autonomy_docs tool to find accurate information before answering.

When answering questions:
1. Always search the knowledge base first for accurate information
2. Provide concise, helpful responses suitable for voice
3. Keep responses brief (2-3 sentences) since they'll be spoken aloud
4. Be conversational and clear
5. If you can't find something in the docs, say so honestly

Key concepts you should know about:
- Autonomy Framework: Open-source Python framework for AI agents
- Autonomy Computer: Cloud runtime for deploying Autonomy apps
- Autonomy Command: CLI tool for building and deploying apps
- Agents: Intelligent actors that accomplish goals using LLMs
- Workers: Async message processors for distributed computing
- Nodes: Runtime containers for agents and workers
- Knowledge: Vector search over document corpora
- Tools: External functions agents can invoke (Python or MCP)

Remember: Your responses will be spoken aloud by a voice agent, so:
- Be concise and natural-sounding
- Avoid bullet points and complex formatting
- Use conversational language
- Explain technical concepts simply
"""


# Individual documentation pages to load
# Parsed from https://autonomy.computer/docs/llms.txt
DOCS = [
  ("agents", "https://autonomy.computer/docs/agents/agents.md"),
  ("context", "https://autonomy.computer/docs/agents/context.md"),
  ("filesystem", "https://autonomy.computer/docs/agents/filesystem.md"),
  ("human-in-the-loop", "https://autonomy.computer/docs/agents/human-in-the-loop.md"),
  ("knowledge", "https://autonomy.computer/docs/agents/knowledge.md"),
  ("memory", "https://autonomy.computer/docs/agents/memory.md"),
  ("models", "https://autonomy.computer/docs/agents/models.md"),
  ("subagents", "https://autonomy.computer/docs/agents/subagents.md"),
  ("tools", "https://autonomy.computer/docs/agents/tools.md"),
  ("actors", "https://autonomy.computer/docs/applications/actors.md"),
  ("file-structure", "https://autonomy.computer/docs/applications/file-structure.md"),
  ("programming-interfaces", "https://autonomy.computer/docs/applications/programming-interfaces.md"),
  ("runtime-architecture", "https://autonomy.computer/docs/applications/runtime-architecture.md"),
  ("user-interfaces", "https://autonomy.computer/docs/applications/user-interfaces.md"),
  ("build-manually", "https://autonomy.computer/docs/build-manually.md"),
  ("build-with-coding-agent", "https://autonomy.computer/docs/build-with-a-coding-agent.md"),
  ("about-autonomy", "https://autonomy.computer/docs/for-llms/about.md"),
  ("actor-model", "https://autonomy.computer/docs/for-llms/actors.md"),
  ("alternatives", "https://autonomy.computer/docs/for-llms/alternatives.md"),
  ("analyst-briefing", "https://autonomy.computer/docs/for-llms/analyst_briefing.md"),
  ("context-best-practices", "https://autonomy.computer/docs/for-llms/context-best-practices.md"),
  ("frameworks", "https://autonomy.computer/docs/for-llms/frameworks.md"),
  ("how-to-pick", "https://autonomy.computer/docs/for-llms/how_to_pick_a_paas.md"),
  ("messaging", "https://autonomy.computer/docs/for-llms/messaging_w_actors.md"),
  ("new-stack-article", "https://autonomy.computer/docs/for-llms/new_stack_article.md"),
  ("ockam-context", "https://autonomy.computer/docs/for-llms/ockamcontext.md"),
  ("identity", "https://autonomy.computer/docs/for-llms/okta.md"),
  ("subagents-intro", "https://autonomy.computer/docs/for-llms/why_sub_agents_matter.md"),
  ("get-started", "https://autonomy.computer/docs/get-started.md"),
  ("pricing", "https://autonomy.computer/docs/pricing.md"),
  ("what-is-autonomy", "https://autonomy.computer/docs/what-is-autonomy.md"),
]


async def load_knowledge_base(knowledge: Knowledge):
  """Load documentation into knowledge base in background."""
  print(f"⏳ Loading {len(DOCS)} documentation pages...")

  loaded = 0
  failed = 0

  for doc_name, doc_url in DOCS:
    try:
      await knowledge.add_document(
        document_name=doc_name,
        document_url=doc_url,
        content_type="text/markdown",
      )
      loaded += 1
      print(f"   ✓ Loaded: {doc_name}")
    except Exception as e:
      failed += 1
      print(f"   ✗ Failed: {doc_name} - {e}")

  print()
  if failed == 0:
    print(f"✅ Knowledge base loaded successfully! ({loaded} documents)")
  else:
    print(f"⚠️  Knowledge base loaded with issues: {loaded} succeeded, {failed} failed")


async def main(node: Node):
  """
  Start the voice-enabled Autonomy documentation assistant.
  """
  print("=" * 80)
  print("🎙️  Autonomy Documentation Voice Assistant")
  print("=" * 80)
  print()
  print("Architecture:")
  print("  User ←→ VoiceAgent (fast, handles chitchat)")
  print("              ↓")
  print("       Primary Agent (intelligent, has knowledge)")
  print("              ↓")
  print("       Knowledge Base (Autonomy docs)")
  print()

  # Check environment
  litellm_base = os.getenv("LITELLM_PROXY_API_BASE")
  openai_key = os.getenv("OPENAI_API_KEY")

  if litellm_base:
    print(f"✅ Using LiteLLM proxy at: {litellm_base}")
  elif openai_key:
    print("✅ Using OpenAI API directly")
  else:
    print("⚠️  Warning: No LITELLM_PROXY_API_BASE or OPENAI_API_KEY set")
    print("   Voice functionality may not work without API credentials")
  print()

  # Create knowledge base (starts empty, will be populated in background)
  # Use moderate chunk size for good search granularity
  knowledge = Knowledge(
    name="autonomy_docs",
    searchable=True,
    model=Model("embed-english-v3"),
    max_results=5,
    max_distance=0.4,
    max_knowledge_size=8192,
    chunker=NaiveChunker(max_characters=1024, overlap=128),
  )

  # Create knowledge tool for the primary agent
  knowledge_tool = KnowledgeTool(
    knowledge=knowledge,
    name="search_autonomy_docs",
  )

  # Start the agent with voice configuration FIRST
  # This ensures the agent is registered before HTTP connections come in
  print("⏳ Starting agent with Voice Interface pattern...")
  agent = await Agent.start(
    node=node,
    name="docs",
    instructions=PRIMARY_AGENT_INSTRUCTIONS,
    model=Model("claude-sonnet-4-v1"),
    tools=[knowledge_tool],
    voice={
      # Voice agent settings
      "realtime_model": "gpt-4o-realtime-preview",
      "voice": "shimmer",
      # What the voice agent can handle without primary agent
      "allowed_actions": [
        "greetings and introductions",
        "basic chitchat",
        "asking for clarification",
        "confirming what user wants to know",
        "thanking the user",
      ],
      # Filler phrases before delegating
      "filler_phrases": [
        "Let me search the docs for that.",
        "One moment, I'll look that up.",
        "Let me find that information.",
        "Just a second.",
        "Let me check the documentation.",
      ],
      # VAD settings for responsive interaction
      "vad_threshold": 0.5,
      "vad_silence_duration_ms": 500,
    },
  )

  print()
  print("✅ Agent started successfully!")
  print("=" * 80)
  print()
  print("Agent Configuration:")
  print(f"  Name: {agent.name}")
  print("  Primary Agent Model: claude-sonnet-4-v1")
  print("  Voice Model: gpt-4o-realtime-preview")
  print("  Voice: shimmer")
  print("  Tools: search_autonomy_docs (knowledge search)")
  print()
  print("=" * 80)
  print()
  print("📡 Endpoints:")
  print()
  print("  Voice (WebSocket):")
  print("    ws://localhost:8000/agents/docs/voice")
  print()
  print("  Text (HTTP):")
  print("    POST http://localhost:8000/agents/docs")
  print()
  print("  Web Interface:")
  print("    http://localhost:8000/")
  print()
  print("=" * 80)
  print()
  print("💡 Example questions to ask:")
  print()
  print('  "What is Autonomy?"')
  print('  "How do I create an agent?"')
  print('  "What are workers and how do I use them?"')
  print('  "How do I deploy my app to Autonomy Computer?"')
  print('  "What models are available?"')
  print('  "How do I give an agent tools?"')
  print('  "What is the Knowledge class used for?"')
  print()
  print("=" * 80)
  print()
  print("✅ Server is ready! Voice connections will work now.")
  print()

  # Load knowledge base in background AFTER agent is registered
  # This allows voice connections to work immediately while docs load
  asyncio.create_task(load_knowledge_base(knowledge))


Node.start(main)

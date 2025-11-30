"""
Voice Interface Example

This example demonstrates the Voice Interface pattern where:
- A fast VoiceAgent (using OpenAI Realtime API) handles user interaction
- Complex tasks are delegated to the Primary Agent (powerful text model with tools)

Architecture:
    User ←→ VoiceAgent (gpt-4o-realtime, fast) ←→ Audio
                  ↓ (delegate_to_primary)
           Primary Agent (gpt-4.1, has tools)

Benefits:
- Low latency voice interaction (realtime model responds immediately)
- High intelligence for complex tasks (primary agent uses powerful text model)
- All tools available via primary agent delegation
- Natural conversation flow with filler phrases

Usage:
    # Set up environment
    export LITELLM_PROXY_API_BASE=http://localhost:4000
    export LITELLM_PROXY_API_KEY=your-key  # if required
    export AWS_REGION=us-west-2

    # Run the example
    cd autonomy/examples/voice-interface
    AUTONOMY_USE_IN_MEMORY_DATABASE=1 uv run --active main.py

    # Connect via WebSocket
    Open index.html in a browser, or connect to:
    ws://localhost:8000/agents/interviewer/voice

Note:
    The VoiceAgent handles basic chitchat directly and delegates
    complex questions to the primary agent. The primary agent has access
    to tools and provides high-quality responses that the VoiceAgent
    speaks verbatim.
"""

import os

from autonomy import Node, Agent, Model, Tool


# Example tool for the primary agent
async def lookup_candidate_info(name: str) -> dict:
  """
  Look up information about a candidate.

  Args:
      name: The candidate's name

  Returns:
      Dictionary with candidate information
  """
  # Simulated candidate database
  candidates = {
    "alice": {
      "name": "Alice Johnson",
      "experience": "5 years",
      "skills": ["Python", "Machine Learning", "System Design"],
      "previous_companies": ["Google", "Meta"],
      "education": "MS Computer Science, Stanford",
    },
    "bob": {
      "name": "Bob Smith",
      "experience": "3 years",
      "skills": ["JavaScript", "React", "Node.js"],
      "previous_companies": ["Startup XYZ"],
      "education": "BS Computer Science, MIT",
    },
  }

  name_lower = name.lower()
  for key, info in candidates.items():
    if key in name_lower or name_lower in info["name"].lower():
      return info

  return {"error": f"No candidate found with name: {name}"}


async def get_interview_questions(topic: str) -> list:
  """
  Get interview questions for a specific topic.

  Args:
      topic: The topic to get questions for (e.g., "algorithms", "system design")

  Returns:
      List of interview questions
  """
  questions = {
    "algorithms": [
      "Can you explain the time complexity of quicksort?",
      "How would you detect a cycle in a linked list?",
      "What's the difference between BFS and DFS?",
    ],
    "system design": [
      "How would you design a URL shortener?",
      "Walk me through designing a real-time chat system.",
      "How would you scale a database to handle millions of users?",
    ],
    "python": [
      "What are decorators and how do they work?",
      "Explain the GIL and its implications.",
      "What's the difference between a list and a tuple?",
    ],
    "behavioral": [
      "Tell me about a challenging project you worked on.",
      "How do you handle disagreements with teammates?",
      "What's your approach to learning new technologies?",
    ],
  }

  topic_lower = topic.lower()
  for key, qs in questions.items():
    if key in topic_lower or topic_lower in key:
      return qs

  return [f"I don't have specific questions for '{topic}', but let's discuss it!"]


# Primary agent instructions - detailed guidance for handling complex tasks
PRIMARY_AGENT_INSTRUCTIONS = """
You are an expert technical interviewer assistant. Your role is to help conduct
interviews by providing thoughtful questions, evaluating responses, and offering
guidance.

You have access to tools:
- lookup_candidate_info: Get information about a candidate
- get_interview_questions: Get questions for specific topics

When asked about candidates or interview topics:
1. Use the appropriate tool to get information
2. Provide concise, helpful responses suitable for voice
3. Keep responses brief (2-3 sentences max) since they'll be spoken aloud
4. Be professional but conversational

For technical questions:
- Provide clear, accurate explanations
- Give concrete examples when helpful
- Adjust complexity based on the context

Remember: Your responses will be spoken aloud by a voice agent, so:
- Be concise and natural-sounding
- Avoid bullet points and complex formatting
- Use conversational language
"""


async def main(node: Node):
  """
  Start the voice-enabled interviewer agent using Voice Interface pattern.
  """
  print("=" * 80)
  print("🎙️  Voice Interface Example")
  print("=" * 80)
  print()
  print("Architecture:")
  print("  User ←→ VoiceAgent (fast, handles chitchat)")
  print("              ↓")
  print("       Primary Agent (intelligent, has tools)")
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

  # Create tools for the primary agent
  tools = [
    Tool(lookup_candidate_info),
    Tool(get_interview_questions),
  ]

  # Start the agent with voice configuration
  print("⏳ Starting agent with Voice Interface pattern...")
  agent = await Agent.start(
    node=node,
    name="interviewer",
    instructions=PRIMARY_AGENT_INSTRUCTIONS,
    model=Model("claude-sonnet-4-v1"),  # Primary agent uses powerful text model
    tools=tools,
    voice={
      # Voice agent settings
      "realtime_model": "gpt-4o-realtime-preview",  # Fast realtime model
      "voice": "alloy",  # Voice ID

      # What the voice agent can handle without primary agent
      "allowed_actions": [
        "greetings and introductions",
        "basic chitchat",
        "asking for clarification",
        "confirming information",
        "thanking the user",
      ],

      # Filler phrases before delegating
      "filler_phrases": [
        "Let me check on that.",
        "One moment.",
        "Let me look that up.",
        "Just a second.",
        "Let me think about that.",
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
  print(f"  Primary Agent Model: claude-sonnet-4-v1")
  print(f"  Voice Model: gpt-4o-realtime-preview")
  print(f"  Tools: lookup_candidate_info, get_interview_questions")
  print()
  print("=" * 80)
  print()
  print("📡 Endpoints:")
  print()
  print("  Voice (WebSocket):")
  print("    ws://localhost:8000/agents/interviewer/voice")
  print()
  print("  Text (HTTP):")
  print("    POST http://localhost:8000/agents/interviewer")
  print()
  print("  Web Interface:")
  print("    http://localhost:8000/")
  print()
  print("=" * 80)
  print()
  print("💡 How it works:")
  print()
  print("  1. VoiceAgent receives audio via WebSocket")
  print("  2. For simple requests (greetings, chitchat), responds directly")
  print("  3. For complex requests, says a filler phrase then delegates")
  print("  4. Primary agent processes request using tools if needed")
  print("  5. VoiceAgent speaks primary agent's response verbatim")
  print()
  print("Example interactions:")
  print('  "Hello!" → VoiceAgent responds directly')
  print('  "Tell me about Alice" → Delegates to primary agent (uses tool)')
  print('  "What questions should I ask about algorithms?" → Delegates')
  print()
  print("=" * 80)
  print()
  print("✅ Server is ready! Open index.html or connect via WebSocket.")
  print()


Node.start(main)

"""
Example 010: Agent Transcripts - Debugging Agent/Model Interactions

This example demonstrates the new agent transcript logging features that let you
see exactly what's sent to the model and what comes back.

Agent transcripts show:
- Human-readable context (messages, tools, instructions)
- Raw API payloads (exact JSON sent to model providers)
- Transformation steps (how internal format becomes API format)
- Model responses (text, tool calls, reasoning)

This is invaluable for:
- Debugging agent behavior
- Understanding model responses
- Validating API payloads
- Prompt engineering
- Production monitoring

Environment Variables:
  AUTONOMY_TRANSCRIPTS=1          - Enable transcript logging (human-readable)
  AUTONOMY_TRANSCRIPTS_RAW=1      - Also show raw API payloads/responses
  AUTONOMY_TRANSCRIPTS_RAW_ONLY=1 - Show ONLY raw JSON (no human-readable)
  AUTONOMY_TRANSCRIPTS_FILE=/path - Write output to file (appends)

Usage Examples:

1. Basic transcript (human-readable):
   AUTONOMY_TRANSCRIPTS=1 \
   AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
   uv run --active examples/010_agent_transcripts.py

2. With raw API payload:
   AUTONOMY_TRANSCRIPTS=1 \
   AUTONOMY_TRANSCRIPTS_RAW=1 \
   AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
   uv run --active examples/010_agent_transcripts.py

3. Only raw API payloads (pure JSON output - includes requests AND responses):
   AUTONOMY_TRANSCRIPTS_RAW_ONLY=1 \
   AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
   uv run --active examples/010_agent_transcripts.py 2>/dev/null

4. Save to file:
   AUTONOMY_TRANSCRIPTS=1 \
   AUTONOMY_TRANSCRIPTS_FILE=/tmp/agent_transcript.log \
   AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
   uv run --active examples/010_agent_transcripts.py

5. Using AWS Bedrock (to see provider-specific transformations):
   AWS_PROFILE=PowerUserAccess-demo-a \
   AUTONOMY_USE_DIRECT_BEDROCK=1 \
   AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
   AUTONOMY_TRANSCRIPTS=1 \
   AUTONOMY_TRANSCRIPTS_RAW=1 \
   CLUSTER="$(autonomy cluster show)" \
   uv run --active examples/010_agent_transcripts.py
"""

from autonomy import Agent, Model, Node, Tool, info
import asyncio
import os


# Define tools for the agent
@Tool
async def get_weather(city: str) -> str:
  """
  Get current weather for a city.

  Args:
    city: The name of the city to get weather for

  Returns:
    Weather information as a string
  """
  # Simulate weather API call
  weather_data = {
    "Tokyo": "Sunny, 22°C",
    "London": "Cloudy, 15°C",
    "New York": "Rainy, 18°C",
    "Paris": "Partly cloudy, 19°C",
    "Sydney": "Clear, 25°C",
  }
  return weather_data.get(city, f"Weather data for {city} not available")


@Tool
async def get_time(city: str) -> str:
  """
  Get current time for a city.

  Args:
    city: The name of the city to get time for

  Returns:
    Current time information as a string
  """
  # Simulate time API call
  time_data = {
    "Tokyo": "15:30 JST",
    "London": "07:30 GMT",
    "New York": "02:30 EST",
    "Paris": "08:30 CET",
    "Sydney": "17:30 AEDT",
  }
  return time_data.get(city, f"Time data for {city} not available")


@Tool
async def calculate_distance(city1: str, city2: str) -> str:
  """
  Calculate distance between two cities in kilometers.

  Args:
    city1: First city name
    city2: Second city name

  Returns:
    Distance information as a string
  """
  # Simulate distance calculation
  distances = {
    ("Tokyo", "London"): "9,580 km",
    ("Tokyo", "New York"): "10,850 km",
    ("London", "New York"): "5,570 km",
    ("London", "Paris"): "340 km",
    ("New York", "Sydney"): "16,000 km",
  }

  key = (city1, city2)
  reverse_key = (city2, city1)

  if key in distances:
    return distances[key]
  elif reverse_key in distances:
    return distances[reverse_key]
  else:
    return f"Distance between {city1} and {city2} not available"


async def main(node):
  """Run the agent transcript example."""

  # Check if in raw_only mode (suppress all non-JSON output)
  raw_only = os.environ.get("AUTONOMY_TRANSCRIPTS_RAW_ONLY", "0") == "1"

  if not raw_only:
    # Print environment variable status
    print("\n" + "=" * 80)
    print("AGENT TRANSCRIPT CONFIGURATION")
    print("=" * 80)
    print(f"AUTONOMY_TRANSCRIPTS:          {os.environ.get('AUTONOMY_TRANSCRIPTS', '0')}")
    print(f"AUTONOMY_TRANSCRIPTS_RAW:      {os.environ.get('AUTONOMY_TRANSCRIPTS_RAW', '0')}")
    print(f"AUTONOMY_TRANSCRIPTS_RAW_ONLY: {os.environ.get('AUTONOMY_TRANSCRIPTS_RAW_ONLY', '0')}")
    print(f"AUTONOMY_TRANSCRIPTS_FILE:     {os.environ.get('AUTONOMY_TRANSCRIPTS_FILE', 'none')}")
    print("=" * 80)
    print()

    # Check if any transcript logging is enabled
    transcript_enabled = (
      os.environ.get("AUTONOMY_TRANSCRIPTS", "0") == "1"
      or os.environ.get("AUTONOMY_TRANSCRIPTS_RAW", "0") == "1"
      or os.environ.get("AUTONOMY_TRANSCRIPTS_RAW_ONLY", "0") == "1"
    )

    if not transcript_enabled:
      print("⚠️  No transcript logging enabled!")
      print("💡 Try running with AUTONOMY_TRANSCRIPTS=1 to see transcripts")
      print()
      print("Examples:")
      print("  AUTONOMY_TRANSCRIPTS=1 uv run --active examples/010_agent_transcripts.py")
      print("  AUTONOMY_TRANSCRIPTS=1 AUTONOMY_TRANSCRIPTS_RAW=1 uv run --active examples/010_agent_transcripts.py")
      print()
    else:
      print("✅ Transcript logging is enabled")
      print("📝 Watch below for transcript output...")
      print()

  # Create agent with tools
  agent = await Agent.start(
    node=node,
    name="travel-info-agent",
    instructions="""
    You are a helpful travel information assistant.

    Your capabilities:
    - Check current weather in cities
    - Get current time in cities
    - Calculate distances between cities

    When users ask questions, use the available tools to gather accurate
    information and provide helpful, detailed answers.
    """,
    model=Model("claude-sonnet-4-v1"),
    tools=[get_weather, get_time, calculate_distance],
  )

  agent_id = await agent.identifier()

  if not raw_only:
    info(f"Agent created with ID: {agent_id}")

    # Test conversation with multiple turns
    info("=" * 80)
    info("CONVERSATION START")
    info("=" * 80)
    info("")

  # Message 1
  if not raw_only:
    info("👤 User: What's the weather in Tokyo?")
    info("")
  response1 = await agent.send("What's the weather in Tokyo?")
  if not raw_only:
    info(f"🤖 Agent: {response1}")
    info("")
    info("-" * 80)
    info("")

  # Message 2
  if not raw_only:
    info("👤 User: What time is it there?")
    info("")
  response2 = await agent.send("What time is it there?")
  if not raw_only:
    info(f"🤖 Agent: {response2}")
    info("")
    info("-" * 80)
    info("")

  # Message 3
  if not raw_only:
    info("👤 User: How far is it from Tokyo to London?")
    info("")
  response3 = await agent.send("How far is it from Tokyo to London?")
  if not raw_only:
    info(f"🤖 Agent: {response3}")
    info("")

    info("-" * 80)
    info("CONVERSATION END")
    info("-" * 80)
    info("")

    info("Example complete")

  if not raw_only and os.environ.get("AUTONOMY_TRANSCRIPTS_FILE"):
    print()
    print("=" * 80)
    print("TRANSCRIPT SAVED TO FILE")
    print("=" * 80)
    print(f"File: {os.environ.get('AUTONOMY_TRANSCRIPTS_FILE')}")
    print(f"View with: cat {os.environ.get('AUTONOMY_TRANSCRIPTS_FILE')}")
    print()

  info("Example complete")


Node.start(main)

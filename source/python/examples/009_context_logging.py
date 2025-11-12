"""
Example 009: Context Logging with Multi-Step Tool Usage

This example demonstrates how to use context logging to inspect the exact
context sent to the model during multi-step agent execution with tool calls.

The agent will:
1. Break down a complex task into steps
2. Use multiple tools to gather information
3. Reason through the results
4. Provide a final answer

RECOMMENDED: Use transcript mode for clean output (no logging noise):
  AWS_PROFILE=PowerUserAccess-demo-a AUTONOMY_USE_DIRECT_BEDROCK=1 AUTONOMY_USE_IN_MEMORY_DATABASE=1 AUTONOMY_TRANSCRIPTS=1 CLUSTER="$(autonomy cluster show)" uv run --active examples/009_context_logging.py

With raw API payloads:
  AWS_PROFILE=PowerUserAccess-demo-a AUTONOMY_USE_DIRECT_BEDROCK=1 AUTONOMY_USE_IN_MEMORY_DATABASE=1 AUTONOMY_TRANSCRIPTS=1 AUTONOMY_TRANSCRIPTS_RAW=1 CLUSTER="$(autonomy cluster show)" uv run --active examples/009_context_logging.py

Or use in-memory database for testing:
  AWS_PROFILE=PowerUserAccess-demo-a AUTONOMY_USE_DIRECT_BEDROCK=1 AUTONOMY_USE_IN_MEMORY_DATABASE=1 AUTONOMY_TRANSCRIPTS=1 uv run --active examples/009_context_logging.py
"""

from autonomy import Agent, Model, Node, Tool, info
import random
import asyncio


# ============================================================================
# TOOLS - Simulating real-world data sources
# ============================================================================


@Tool
async def get_weather(city: str) -> str:
  """
  Get current weather information for a city.

  Args:
    city: The name of the city to get weather for

  Returns:
    Weather information as a string
  """
  # Simulate API delay
  await asyncio.sleep(0.5)

  # Simulate weather data
  weather_data = {
    "San Francisco": {"temp": 65, "condition": "Foggy", "humidity": 75},
    "New York": {"temp": 72, "condition": "Sunny", "humidity": 60},
    "London": {"temp": 58, "condition": "Rainy", "humidity": 85},
    "Tokyo": {"temp": 68, "condition": "Cloudy", "humidity": 70},
  }

  data = weather_data.get(city, {"temp": 70, "condition": "Clear", "humidity": 65})
  return f"Weather in {city}: {data['condition']}, {data['temp']}°F, Humidity: {data['humidity']}%"


@Tool
async def get_flight_price(origin: str, destination: str) -> str:
  """
  Get estimated flight price between two cities.

  Args:
    origin: Departure city
    destination: Arrival city

  Returns:
    Flight price information
  """
  # Simulate API delay
  await asyncio.sleep(0.5)

  # Simulate price calculation
  base_price = random.randint(200, 800)
  return f"Flight from {origin} to {destination}: approximately ${base_price} (economy class)"


@Tool
async def get_hotel_availability(city: str, nights: int) -> str:
  """
  Check hotel availability and pricing in a city.

  Args:
    city: The city to check hotels in
    nights: Number of nights to stay

  Returns:
    Hotel availability and pricing information
  """
  # Simulate API delay
  await asyncio.sleep(0.5)

  # Simulate hotel data
  price_per_night = random.randint(100, 300)
  total = price_per_night * nights

  return f"Hotels in {city}: Available from ${price_per_night}/night. Total for {nights} nights: ${total}"


@Tool
async def calculate_total_cost(items: list) -> str:
  """
  Calculate total cost from a list of expense items.

  Args:
    items: List of expense amounts (as numbers)

  Returns:
    Total cost calculation
  """
  try:
    total = sum(float(item) for item in items)
    return f"Total calculated: ${total:.2f}"
  except (ValueError, TypeError) as e:
    return f"Error calculating total: {str(e)}"


@Tool
async def get_travel_tips(destination: str) -> str:
  """
  Get travel tips and recommendations for a destination.

  Args:
    destination: The destination city

  Returns:
    Travel tips and recommendations
  """
  # Simulate API delay
  await asyncio.sleep(0.5)

  tips = {
    "San Francisco": [
      "Visit the Golden Gate Bridge early morning",
      "Try clam chowder at Fisherman's Wharf",
      "Ride a cable car",
    ],
    "New York": [
      "Visit Central Park in spring",
      "See a Broadway show",
      "Try authentic NYC pizza",
    ],
    "London": [
      "Visit the British Museum (free entry)",
      "Take afternoon tea",
      "Use the Tube for transportation",
    ],
    "Tokyo": [
      "Get a JR Pass for trains",
      "Visit temples in Asakusa",
      "Try authentic ramen",
    ],
  }

  city_tips = tips.get(destination, ["Enjoy local cuisine", "Visit popular landmarks", "Learn basic local phrases"])
  return f"Travel tips for {destination}:\n" + "\n".join(f"- {tip}" for tip in city_tips)


# ============================================================================
# MAIN EXAMPLE
# ============================================================================


async def main(node):
  info("=" * 80)
  info("EXAMPLE 009: Context Logging with Multi-Step Tool Usage")
  info("=" * 80)
  info("")

  # Check if transcript logging is enabled
  import os

  transcript_enabled = os.environ.get("AUTONOMY_TRANSCRIPTS", "0") == "1"
  transcript_raw = os.environ.get("AUTONOMY_TRANSCRIPTS_RAW", "0") == "1"

  if transcript_enabled:
    info("✓ Transcript Mode ENABLED")
    info("  You will see human-readable context and model responses")
    if transcript_raw:
      info("  You will also see raw API payloads and responses")
  else:
    info("✗ Transcript logging is DISABLED")
    info("  To enable: AUTONOMY_TRANSCRIPTS=1")
    info("  To also see raw API: AUTONOMY_TRANSCRIPTS=1 AUTONOMY_TRANSCRIPTS_RAW=1")

  info("")
  info("-" * 80)
  info("Setting up Travel Planning Agent with Tools")
  info("-" * 80)
  info("")

  # Create an agent with travel planning capabilities
  agent = await Agent.start(
    node=node,
    name="travel-planner",
    instructions="""
      You are an expert travel planning assistant. Your goal is to help users
      plan their trips by providing comprehensive information about weather,
      flights, hotels, and travel tips.

      When planning a trip:
      1. First, check the weather at the destination
      2. Look up flight prices
      3. Check hotel availability
      4. Calculate total estimated costs
      5. Provide travel tips
      6. Summarize everything in a clear, organized way

      Always break down complex requests into smaller steps and use the
      available tools to gather accurate information.
    """,
    model=Model(name="claude-sonnet-4-v1"),
    tools=[
      get_weather,
      get_flight_price,
      get_hotel_availability,
      calculate_total_cost,
      get_travel_tips,
    ],
  )

  agent_id = await agent.identifier()
  info(f"✓ Agent created with ID: {agent_id}")
  info(f"✓ Agent configured with 5 tools")
  info("")

  info("-" * 80)
  info("CONVERSATION START")
  info("-" * 80)
  info("")

  # Example 1: Simple query (will trigger multiple tool calls)
  info("👤 User: I'm planning a 3-day trip from San Francisco to Tokyo. Can you help me plan?")
  info("")

  response1 = await agent.send("I'm planning a 3-day trip from San Francisco to Tokyo. Can you help me plan?")
  info(f"🤖 Agent: {response1}")
  info("")

  info("-" * 80)
  info("")

  # Example 2: Follow-up question (will show conversation history in context)
  info("👤 User: What if I stay for 5 days instead?")
  info("")

  response2 = await agent.send("What if I stay for 5 days instead?")
  info(f"🤖 Agent: {response2}")
  info("")

  info("-" * 80)
  info("")

  # Example 3: Another follow-up (showing growing context)
  info("👤 User: Can you give me some travel tips for Tokyo?")
  info("")

  response3 = await agent.send("Can you give me some travel tips for Tokyo?")
  info(f"🤖 Agent: {response3}")
  info("")

  info("-" * 80)
  info("CONVERSATION END")
  info("-" * 80)
  info("")

  # Summary
  info("=" * 80)
  info("EXAMPLE SUMMARY")
  info("=" * 80)
  info("")
  info("This example demonstrated:")
  info("  ✓ Multi-step agent reasoning")
  info("  ✓ Multiple tool calls per turn")
  info("  ✓ Context accumulation across conversation")
  info("  ✓ Tool results being added to context")
  info("")

  if transcript_enabled:
    info("You saw the complete context logged before each model call, including:")
    info("  • System instructions (agent's capabilities)")
    info("  • Available tools")
    info("  • Conversation history (all messages)")
    info("  • Tool calls and results")
    info("")
    if transcript_raw:
      info("You also saw raw API payloads showing:")
      info("  • Provider-specific transformations")
      info("  • Actual JSON sent to the model API")
      info("  • Raw responses from the model")
      info("")
  else:
    info("To see transcript logging, run with:")
    info("  AUTONOMY_TRANSCRIPTS=1 uv run --active examples/009_context_logging.py")
    info("")
    info("To also see raw API payloads:")
    info("  AUTONOMY_TRANSCRIPTS=1 AUTONOMY_TRANSCRIPTS_RAW=1 uv run --active examples/009_context_logging.py")
    info("")

  info("Transcript logging helps you:")
  info("  • Debug unexpected agent behavior")
  info("  • Understand what information the model sees")
  info("  • Validate context templates")
  info("  • Optimize context size")
  info("  • Inspect raw API payloads")
  info("")


Node.start(main)

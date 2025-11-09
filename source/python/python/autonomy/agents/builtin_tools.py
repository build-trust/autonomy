"""
Built-in tools automatically available to all agents.

These tools are registered with every agent instance and provide
core functionality for human-in-the-loop interactions, debugging,
and other standard operations.
"""

import json
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional
from ..tools.protocol import InvokableTool


class AskUserForInputTool(InvokableTool):
  """
  Built-in tool that allows agents to request user input.

  When an agent calls this tool, the state machine transitions to
  WAITING_FOR_INPUT state, sends the prompt to the user, and pauses
  execution until the user responds.

  Example usage by agent:
    Agent: "I need to know your budget. Let me ask..."
    Tool call: ask_user_for_input(prompt="What is your budget?")
    → State machine pauses
    → User sees: "What is your budget?"
    User: "$500"
    → State machine resumes
    → Tool result: "$500"
    Agent: "Based on your budget of $500..."

  Args:
    prompt: The question or prompt to show the user

  Returns:
    Special marker dict with _waiting_for_input flag and prompt.
    The state machine detects this marker and transitions to WAITING_FOR_INPUT.
  """

  def __init__(self):
    super().__init__()
    self.name = "ask_user_for_input"
    self.description = (
      "REQUIRED: Use this tool whenever you need information from the user that you don't have. "
      "This includes any questions, clarifications, or requests for details. "
      "DO NOT ask questions in your text response - always use this tool to ask questions. "
      "The conversation will pause and wait for the user's response before continuing. "
      "After receiving the user's answer, you can proceed with their request."
    )

  async def invoke(self, json_argument: Optional[str]) -> Dict[str, Any]:
    """
    Mark that we're waiting for user input.

    The actual pausing happens in StateMachine._handle_acting_state()
    when it detects this special return value.

    Args:
      json_argument: JSON string containing 'prompt' or 'question' key with the question for the user

    Returns:
      Dict with _waiting_for_input marker flag and prompt text
    """
    # Parse JSON argument string into dictionary
    if json_argument is None or json_argument.strip() == "":
      params = {}
    else:
      try:
        params = json.loads(json_argument)
        if not isinstance(params, dict):
          params = {}
      except json.JSONDecodeError:
        params = {}

    # Accept both 'question' and 'prompt' for flexibility
    prompt = params.get("question") or params.get("prompt") or "Please provide input:"

    # Return special marker that state machine recognizes
    return {
      "_waiting_for_input": True,
      "prompt": prompt,
    }

  async def spec(self) -> dict:
    """
    Return OpenAI-compatible tool specification.

    Returns:
      Tool spec dict in OpenAI function calling format
    """
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": {
          "type": "object",
          "properties": {
            "question": {
              "type": "string",
              "description": "The exact question or prompt to display to the user. Be clear and specific about what information you need."
            }
          },
          "required": ["question"],
        },
      },
    }


class GetCurrentTimeUtcTool(InvokableTool):
  """
  Built-in tool that returns the current UTC time.

  Returns the current time in UTC timezone formatted as ISO 8601.
  This is useful for agents that need to know the current time for
  timestamping, scheduling, or time-sensitive operations.

  Example usage by agent:
    Agent: "Let me check the current time..."
    Tool call: get_current_time_utc()
    → Tool result: "2024-01-15T14:30:45.123456+00:00"
    Agent: "The current UTC time is 2:30 PM..."

  Returns:
    ISO 8601 formatted UTC timestamp string
  """

  def __init__(self):
    super().__init__()
    self.name = "get_current_time_utc"
    self.description = (
      "Get the current time in UTC timezone. Returns the time in ISO 8601 format."
    )

  async def invoke(self, json_argument: Optional[str]) -> str:
    """
    Return the current UTC time as ISO 8601 string.

    Args:
      json_argument: Not used for this tool (accepts empty or None)

    Returns:
      Current UTC time as ISO 8601 formatted string
    """
    return datetime.now(UTC).isoformat()

  async def spec(self) -> dict:
    """
    Return OpenAI-compatible tool specification.

    Returns:
      Tool spec dict in OpenAI function calling format
    """
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": {
          "type": "object",
          "properties": {},
          "required": [],
        },
      },
    }


class GetCurrentTimeTool(InvokableTool):
  """
  Built-in tool that returns the current time in a specific timezone.

  Returns the current time in the specified timezone formatted as ISO 8601.
  This is useful for agents that need to work with times in different
  timezones or provide localized time information.

  Example usage by agent:
    Agent: "Let me check the current time in New York..."
    Tool call: get_current_time(timezone="America/New_York")
    → Tool result: "2024-01-15T09:30:45.123456-05:00"
    Agent: "The current time in New York is 9:30 AM..."

  Args:
    timezone: IANA timezone name (e.g., "America/New_York", "Europe/London", "Asia/Tokyo")

  Returns:
    ISO 8601 formatted timestamp string in the specified timezone
  """

  def __init__(self):
    super().__init__()
    self.name = "get_current_time"
    self.description = (
      "Get the current time in a specific timezone. "
      "Provide a timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'). "
      "Returns the time in ISO 8601 format."
    )

  async def invoke(self, json_argument: Optional[str]) -> str:
    """
    Return the current time in the specified timezone as ISO 8601 string.

    Args:
      json_argument: JSON string containing 'timezone' key with IANA timezone name

    Returns:
      Current time in specified timezone as ISO 8601 formatted string
    """
    # Parse JSON argument string into dictionary
    if json_argument is None or json_argument.strip() == "":
      params = {}
    else:
      try:
        params = json.loads(json_argument)
        if not isinstance(params, dict):
          params = {}
      except json.JSONDecodeError:
        params = {}

    timezone_name = params.get("timezone", "UTC")

    try:
      tz = ZoneInfo(timezone_name)
      return datetime.now(tz).isoformat()
    except Exception as e:
      return f"Error: Invalid timezone '{timezone_name}'. Please use a valid IANA timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'). Error: {str(e)}"

  async def spec(self) -> dict:
    """
    Return OpenAI-compatible tool specification.

    Returns:
      Tool spec dict in OpenAI function calling format
    """
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": {
          "type": "object",
          "properties": {
            "timezone": {
              "type": "string",
              "description": "IANA timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo')",
            }
          },
          "required": ["timezone"],
        },
      },
    }

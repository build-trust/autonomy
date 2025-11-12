"""
Built-in tools automatically available to all agents.

These tools are registered with every agent instance and provide
core functionality for human-in-the-loop interactions, debugging,
subagent management, and other standard operations.
"""

import json
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from ..tools.protocol import InvokableTool

if TYPE_CHECKING:
  from .agent import Agent


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
              "description": "The exact question or prompt to display to the user. Be clear and specific about what information you need.",
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
  Useful for agents that need to know the current time for scheduling,
  timestamps, or time-based decisions.

  Returns:
    Current UTC time as ISO 8601 string (e.g., "2024-01-15T14:30:00Z")
  """

  def __init__(self):
    super().__init__()
    self.name = "get_current_time_utc"
    self.description = (
      "Get the current time in UTC timezone. "
      "Returns the time formatted as ISO 8601 (e.g., '2024-01-15T14:30:00Z'). "
      "Use this when you need to know the current time for scheduling, "
      "timestamps, or time-based decisions."
    )

  async def invoke(self, json_argument: Optional[str]) -> str:
    """
    Return current UTC time.

    Returns:
      ISO 8601 formatted timestamp string
    """
    current_time = datetime.now(UTC)
    return current_time.isoformat()

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
        },
      },
    }


class StartSubagentTool(InvokableTool):
  """
  Built-in tool that allows agents to start configured subagents on-demand.

  When an agent calls this tool, it starts a subagent using one of the
  predefined subagent configurations from the parent agent's config.

  Args:
    role: The role/name of the subagent to start (must match a key in subagents config)

  Returns:
    Success message with subagent identifier or error message
  """

  def __init__(self, agent: "Agent"):
    super().__init__()
    self.agent = agent
    self.name = "start_subagent"
    self.description = (
      "Start a configured subagent by role name. The subagent will be initialized "
      "and ready to receive tasks via delegate_to_subagent."
    )

  async def invoke(self, json_argument: Optional[str]) -> str:
    """Start a subagent by role."""
    if json_argument is None or json_argument.strip() == "":
      return "Error: 'role' parameter is required"

    try:
      params = json.loads(json_argument)
      if not isinstance(params, dict):
        return "Error: Invalid parameters format"
    except json.JSONDecodeError:
      return "Error: Invalid JSON in parameters"

    role = params.get("role")
    if not role:
      return "Error: 'role' parameter is required"

    try:
      subagent_ref = await self.agent.subagents.start_subagent(role)
      return f"Successfully started subagent '{role}' (name: {subagent_ref.name})"
    except Exception as e:
      return f"Error starting subagent: {str(e)}"

  async def spec(self) -> dict:
    """Return OpenAI-compatible tool specification."""
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": {
          "type": "object",
          "properties": {
            "role": {
              "type": "string",
              "description": "The role of the subagent to start",
            }
          },
          "required": ["role"],
        },
      },
    }


class DelegateToSubagentTool(InvokableTool):
  """
  Built-in tool that delegates a task to a specific subagent.

  Sends a task to a running subagent and waits for its response.
  The subagent must be started first (either via start_subagent or auto_start).

  Args:
    role: The role of the subagent to delegate to
    task: The task description or prompt to send
    timeout: Optional timeout in seconds (default: 60)

  Returns:
    The subagent's response text or error message
  """

  def __init__(self, agent: "Agent"):
    super().__init__()
    self.agent = agent
    self.name = "delegate_to_subagent"
    self.description = (
      "Delegate a task to a subagent and wait for its response. "
      "The subagent must be started first (or configured with auto_start)."
    )

  async def invoke(self, json_argument: Optional[str]) -> str:
    """Delegate a task to a subagent."""
    if json_argument is None or json_argument.strip() == "":
      return "Error: 'role' and 'task' parameters are required"

    try:
      params = json.loads(json_argument)
      if not isinstance(params, dict):
        return "Error: Invalid parameters format"
    except json.JSONDecodeError:
      return "Error: Invalid JSON in parameters"

    role = params.get("role")
    task = params.get("task")
    timeout = params.get("timeout", 60)

    if not role or not task:
      return "Error: Both 'role' and 'task' parameters are required"

    try:
      result = await self.agent.subagents.delegate_to_subagent(role, task, timeout=timeout)
      return result
    except Exception as e:
      return f"Error delegating to subagent: {str(e)}"

  async def spec(self) -> dict:
    """Return OpenAI-compatible tool specification."""
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": {
          "type": "object",
          "properties": {
            "role": {
              "type": "string",
              "description": "The role of the subagent to delegate to",
            },
            "task": {
              "type": "string",
              "description": "The task description or prompt to send to the subagent",
            },
            "timeout": {
              "type": "number",
              "description": "Optional timeout in seconds (default: 60)",
            },
          },
          "required": ["role", "task"],
        },
      },
    }


class DelegateToSubagentsParallelTool(InvokableTool):
  """
  Built-in tool that delegates multiple tasks to parallel subagent instances.

  Creates one subagent instance per task and processes them concurrently.
  Useful for batch processing or parallel research/analysis tasks.

  Args:
    role: The role of subagents to use
    tasks: List of task descriptions to process in parallel
    timeout: Optional timeout in seconds per task (default: 60)
    runner_filter: Optional filter for targeting specific runner nodes

  Returns:
    JSON string with list of results, each containing task, result/error, and success status
  """

  def __init__(self, agent: "Agent"):
    super().__init__()
    self.agent = agent
    self.name = "delegate_to_subagents_parallel"
    self.description = (
      "Delegate multiple tasks to parallel subagent instances. "
      "Creates one subagent per task for concurrent processing. "
      "Returns results for all tasks."
    )

  async def invoke(self, json_argument: Optional[str]) -> str:
    """Delegate tasks to parallel subagents."""
    if json_argument is None or json_argument.strip() == "":
      return "Error: 'role' and 'tasks' parameters are required"

    try:
      params = json.loads(json_argument)
      if not isinstance(params, dict):
        return "Error: Invalid parameters format"
    except json.JSONDecodeError:
      return "Error: Invalid JSON in parameters"

    role = params.get("role")
    tasks = params.get("tasks")
    timeout = params.get("timeout", 60)
    runner_filter = params.get("runner_filter")

    if not role or not tasks:
      return "Error: Both 'role' and 'tasks' parameters are required"

    try:
      results = await self.agent.subagents.delegate_to_subagents_parallel(
        role, tasks, timeout=timeout, runner_filter=runner_filter
      )
      return json.dumps(results, indent=2)
    except Exception as e:
      return f"Error in parallel delegation: {str(e)}"

  async def spec(self) -> dict:
    """Return OpenAI-compatible tool specification."""
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": {
          "type": "object",
          "properties": {
            "role": {
              "type": "string",
              "description": "The role of subagents to use for parallel execution",
            },
            "tasks": {
              "type": "array",
              "items": {"type": "string"},
              "description": "List of task descriptions to process in parallel",
            },
            "timeout": {
              "type": "number",
              "description": "Optional timeout in seconds per task (default: 60)",
            },
            "runner_filter": {
              "type": "string",
              "description": "Optional filter for Zone.nodes() to target specific runners",
            },
          },
          "required": ["role", "tasks"],
        },
      },
    }


class ListSubagentsTool(InvokableTool):
  """
  Built-in tool that lists configured and running subagents.

  Returns information about all subagent configurations and their current status.

  Returns:
    JSON string with configured roles, running roles, and detailed status
  """

  def __init__(self, agent: "Agent"):
    super().__init__()
    self.agent = agent
    self.name = "list_subagents"
    self.description = "List all configured and running subagents with their status."

  async def invoke(self, json_argument: Optional[str]) -> str:
    """List subagents."""
    try:
      result = self.agent.subagents.list_subagents()
      return json.dumps(result, indent=2)
    except Exception as e:
      return f"Error listing subagents: {str(e)}"

  async def spec(self) -> dict:
    """Return OpenAI-compatible tool specification."""
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": {
          "type": "object",
          "properties": {},
        },
      },
    }


class GetCurrentTimeTool(InvokableTool):
  """
  Built-in tool that returns the current time in a specific timezone.

  Returns the current time in the specified timezone formatted as ISO 8601.
  Useful for agents that need to work with different timezones.

  Args:
    timezone: IANA timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo')

  Returns:
    Current time in the specified timezone as ISO 8601 string
  """

  def __init__(self):
    super().__init__()
    self.name = "get_current_time"
    self.description = (
      "Get the current time in a specific timezone. "
      "Returns the time formatted as ISO 8601. "
      "Accepts IANA timezone names like 'America/New_York', 'Europe/London', 'Asia/Tokyo'. "
      "Use this when you need to know the current time in a specific location."
    )

  async def invoke(self, json_argument: Optional[str]) -> str:
    """
    Return current time in specified timezone.

    Args:
      json_argument: JSON string with 'timezone' field (defaults to UTC if not provided)

    Returns:
      ISO 8601 formatted timestamp string or error message
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
      current_time = datetime.now(tz)
      return current_time.isoformat()
    except Exception as e:
      return f"Error: Invalid timezone '{timezone_name}'. Use IANA timezone names like 'America/New_York', 'Europe/London', 'Asia/Tokyo'. Error: {str(e)}"

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
              "description": "IANA timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'). Defaults to UTC if not provided.",
            }
          },
          "required": [],
        },
      },
    }


class StopSubagentTool(InvokableTool):
  """
  Built-in tool that allows agents to stop running subagents.

  Args:
    role: The role of the subagent to stop

  Returns:
    Success or error message
  """

  def __init__(self, agent: "Agent"):
    super().__init__()
    self.agent = agent
    self.name = "stop_subagent"
    self.description = "Stop a running subagent by role name."

  async def invoke(self, json_argument: Optional[str]) -> str:
    """Stop a subagent."""
    if json_argument is None or json_argument.strip() == "":
      return "Error: 'role' parameter is required"

    try:
      params = json.loads(json_argument)
      if not isinstance(params, dict):
        return "Error: Invalid parameters format"
    except json.JSONDecodeError:
      return "Error: Invalid JSON in parameters"

    role = params.get("role")
    if not role:
      return "Error: 'role' parameter is required"

    try:
      await self.agent.subagents.stop_subagent(role)
      return f"Successfully stopped subagent '{role}'"
    except Exception as e:
      return f"Error stopping subagent: {str(e)}"

  async def spec(self) -> dict:
    """Return OpenAI-compatible tool specification."""
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": {
          "type": "object",
          "properties": {
            "role": {
              "type": "string",
              "description": "The role of the subagent to stop",
            }
          },
          "required": ["role"],
        },
      },
    }

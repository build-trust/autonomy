"""
Built-in tools automatically available to all agents.

These tools are registered with every agent instance and provide
core functionality for human-in-the-loop interactions, debugging,
and other standard operations.
"""

from typing import Any, Dict
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
      "Ask the user for additional information. Use this when you need "
      "clarification or more details from the user. The agent will pause "
      "and wait for the user's response before continuing."
    )

  async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mark that we're waiting for user input.

    The actual pausing happens in StateMachine._handle_acting_state()
    when it detects this special return value.

    Args:
      params: Must contain 'prompt' or 'question' key with the question for the user

    Returns:
      Dict with _waiting_for_input marker flag and prompt text
    """
    # Accept both 'question' and 'prompt' for flexibility
    prompt = params.get("question") or params.get("prompt") or "Please provide input:"

    # Return special marker that state machine recognizes
    return {
      "_waiting_for_input": True,
      "prompt": prompt,
    }

  def spec(self) -> dict:
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
              "description": "The question to ask the user"
            }
          },
          "required": ["question"]
        }
      }
    }

"""
Custom exception classes for agents.

This module provides exception classes with helpful context for debugging
timeout and other agent-related errors.
"""

import asyncio
from typing import Optional, Dict, Any


class AgentTimeoutError(asyncio.TimeoutError):
  """
  Base class for agent-related timeout errors.

  Provides helpful context about what operation timed out and suggestions
  for resolution.

  Attributes:
    method: The method or operation that timed out
    timeout: The timeout value in seconds
    context: Additional context about the operation
    message: Human-readable error message
  """

  def __init__(
    self,
    method: str,
    timeout: float,
    context: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
  ):
    self.method = method
    self.timeout = timeout
    self.context = context or {}

    if message is None:
      message = self._build_message()

    self.message = message
    super().__init__(message)

  def _build_message(self) -> str:
    """Build a helpful error message with context."""
    parts = [f"{self.method} timed out after {self.timeout}s."]

    if self.context:
      context_parts = [f"{k}: {v}" for k, v in self.context.items() if v is not None]
      if context_parts:
        parts.append(f"Context: {', '.join(context_parts)}.")

    parts.append(self._get_suggestion())

    return " ".join(parts)

  def _get_suggestion(self) -> str:
    """Get a suggestion for resolving the timeout."""
    return "Consider increasing the timeout value."


class AgentStartTimeoutError(AgentTimeoutError):
  """
  Raised when Agent.start() times out.

  This typically indicates network issues, gateway connectivity problems,
  or the node being overloaded.

  Example:
    try:
      agent = await Agent.start(node=node, instructions="...", timeout=30.0)
    except AgentStartTimeoutError as e:
      print(f"Failed to start agent: {e}")
      print(f"Agent name: {e.context.get('agent_name')}")
  """

  def __init__(
    self,
    timeout: float,
    agent_name: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
  ):
    ctx = context or {}
    ctx["agent_name"] = agent_name
    super().__init__(
      method="Agent.start()",
      timeout=timeout,
      context=ctx,
    )
    self.agent_name = agent_name

  def _get_suggestion(self) -> str:
    return (
      "Consider increasing the timeout, checking gateway connectivity, "
      "or verifying the node is running and accessible."
    )


class AgentStopTimeoutError(AgentTimeoutError):
  """
  Raised when Agent.stop() times out.

  This typically indicates the agent is stuck processing a request,
  has entered an invalid state, or there are network issues.

  Example:
    try:
      await Agent.stop(node, agent_name, timeout=10.0, raise_on_timeout=True)
    except AgentStopTimeoutError as e:
      print(f"Failed to stop agent cleanly: {e}")
  """

  def __init__(
    self,
    timeout: float,
    agent_name: str,
    context: Optional[Dict[str, Any]] = None,
  ):
    ctx = context or {}
    ctx["agent_name"] = agent_name
    super().__init__(
      method="Agent.stop()",
      timeout=timeout,
      context=ctx,
    )
    self.agent_name = agent_name

  def _get_suggestion(self) -> str:
    return (
      "The agent may be processing a long-running request. "
      "Consider increasing the timeout or forcefully terminating if necessary."
    )


class SendMessageTimeoutError(AgentTimeoutError):
  """
  Raised when Reference.send_message() times out.

  This typically indicates network issues or the receiving agent
  being unresponsive.

  Example:
    try:
      receive_ref = await agent.send_message("Hello", timeout=30.0)
    except SendMessageTimeoutError as e:
      print(f"Failed to send message: {e}")
  """

  def __init__(
    self,
    timeout: float,
    agent_name: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
  ):
    ctx = context or {}
    ctx["agent_name"] = agent_name
    super().__init__(
      method="Reference.send_message()",
      timeout=timeout,
      context=ctx,
    )
    self.agent_name = agent_name

  def _get_suggestion(self) -> str:
    return (
      "Consider increasing the timeout or checking if the target agent is running. "
      "For request-response patterns, prefer using send() which has built-in timeout support."
    )

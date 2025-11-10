"""
Subagent lifecycle management for hierarchical agent coordination.

This module provides the Subagents class which manages the lifecycle of
subagents including starting, delegating tasks, parallel execution, and cleanup.
"""

import asyncio
import secrets
from typing import Any, Dict, List, Optional

from ..logs import get_logger

logger = get_logger("subagents")


class Subagents:
  """
  Manages subagent lifecycle and task delegation.

  This class handles:
  - Starting subagents from configuration
  - Single task delegation to subagents
  - Parallel task delegation across multiple subagent instances
  - Subagent lifecycle tracking and cleanup
  - Optional distribution across runner nodes

  Example:
    # Created automatically when agent has subagent configs
    agent.subagents.start_subagent(role="researcher")
    result = await agent.subagents.delegate_to_subagent(
      role="researcher",
      task="Research quantum computing"
    )
    results = await agent.subagents.delegate_to_subagents_parallel(
      role="researcher",
      tasks=["Topic 1", "Topic 2", "Topic 3"]
    )
  """

  def __init__(self, agent: "Agent"):
    """
    Initialize Subagents manager.

    Args:
      agent: Parent agent instance that owns these subagents
    """
    self.agent = agent
    self.subagent_configs = agent.subagent_configs or {}
    self.running_subagents: Dict[str, Any] = {}  # Dict[role, AgentReference]
    self.runner_filter = getattr(agent, "subagent_runner_filter", None)
    self._start_locks: Dict[str, asyncio.Lock] = {}  # Per-role locks to prevent race conditions

  async def start_subagent(self, role: str):
    """
    Start a subagent by role name.

    Args:
      role: The role/name of the subagent to start (must match a key in subagent configs)

    Returns:
      AgentReference for the started subagent

    Raises:
      ValueError: If role not found in configuration or already running
    """
    if role not in self.subagent_configs:
      raise ValueError(f"No subagent configuration found for role: {role}")

    # Ensure we have a lock for this role
    if role not in self._start_locks:
      self._start_locks[role] = asyncio.Lock()

    # Use lock to prevent race conditions when multiple concurrent requests try to start the same role
    async with self._start_locks[role]:
      # Double-check inside the lock in case another coroutine started it
      if role in self.running_subagents:
        logger.debug(f"Subagent '{role}' already running, returning existing reference")
        return self.running_subagents[role]

      config = self.subagent_configs[role]
      subagent_name = f"{self.agent.name}__subagent__{role}__{secrets.token_hex(6)}"

      logger.info(f"Starting subagent '{role}' with name '{subagent_name}'")

      # Create full config for subagent
      subagent_config = {
        "instructions": config["instructions"],
        "name": subagent_name,
      }

      # Copy optional fields
      for field in [
        "model",
        "tools",
        "max_iterations",
        "max_execution_time",
        "max_messages_in_short_term_memory",
        "max_tokens_in_short_term_memory",
        "enable_long_term_memory",
      ]:
        if field in config:
          subagent_config[field] = config[field]

      # Import here to avoid circular dependency
      from ..agents.agent import Agent

      # Start the subagent
      subagent_ref = await Agent.start_from_config(node=self.agent.node, config=subagent_config)

      self.running_subagents[role] = subagent_ref
      logger.debug(f"Subagent '{role}' started successfully")
      return subagent_ref

  async def delegate_to_subagent(self, role: str, task: str, timeout: float = 60) -> str:
    """
    Delegate a task to a subagent and wait for response.

    Args:
      role: The role of the subagent to delegate to
      task: The task description or prompt to send
      timeout: Timeout in seconds (default: 60)

    Returns:
      The subagent's response text

    Raises:
      ValueError: If subagent not configured or not running without auto_start
    """
    if role not in self.subagent_configs:
      raise ValueError(f"No subagent configuration found for role: {role}")

    # Auto-start if not running and configured to do so
    if role not in self.running_subagents:
      config = self.subagent_configs[role]
      if config.get("auto_start", False):
        logger.debug(f"Auto-starting subagent '{role}'")
        await self.start_subagent(role)
      else:
        raise ValueError(
          f"Subagent '{role}' is not running. Start it first using start_subagent or set auto_start=True."
        )

    subagent_ref = self.running_subagents[role]
    logger.debug(f"Delegating task to subagent '{role}': {task[:100]}...")

    try:
      response = await subagent_ref.send(task, timeout=timeout)

      if len(response) > 0:
        result = response[-1].content.text
        logger.debug(f"Subagent '{role}' completed task successfully")
        return result
      else:
        logger.warning(f"Subagent '{role}' returned empty response")
        return "Error: Subagent returned empty response"
    except Exception as e:
      logger.error(f"Error delegating to subagent '{role}': {e}")
      raise

  async def delegate_to_subagents_parallel(
    self, role: str, tasks: List[str], timeout: float = 60, runner_filter: Optional[str] = None
  ) -> List[Dict[str, Any]]:
    """
    Delegate multiple tasks to parallel subagent instances.

    Creates one subagent instance per task and processes them concurrently.
    Optionally distributes subagents across runner nodes for true parallelism.

    Args:
      role: The role of subagents to use
      tasks: List of task descriptions to process in parallel
      timeout: Timeout in seconds per task (default: 60)
      runner_filter: Optional filter for Zone.nodes() to target specific runners

    Returns:
      List of result dicts in same order as tasks, each containing:
        - task: The original task
        - result: The response text (if successful)
        - error: Error message (if failed)
        - subagent: Name of the subagent that processed it
        - success: Boolean indicating success/failure

    Raises:
      ValueError: If role not configured, tasks is not a list, or no runners found with filter
    """
    if role not in self.subagent_configs:
      raise ValueError(f"No subagent configuration found for role: {role}")

    if not isinstance(tasks, list):
      raise ValueError("tasks parameter must be a list")

    if len(tasks) == 0:
      raise ValueError("tasks list cannot be empty")

    config = self.subagent_configs[role]
    logger.info(f"Starting parallel delegation of {len(tasks)} tasks to '{role}' subagents")

    # Determine which runner filter to use
    effective_filter = runner_filter or config.get("runner_filter") or self.runner_filter

    # Get runner nodes if filter specified
    runners = None
    if effective_filter:
      from ..clusters.zone import Zone

      runners = await Zone.nodes(self.agent.node, filter=effective_filter)
      if not runners:
        raise ValueError(f"No runner nodes found with filter: {effective_filter}")
      logger.debug(f"Found {len(runners)} runner nodes for parallel execution")

    # Import here to avoid circular dependency
    from ..agents.agent import Agent

    # Create async function to process a single task
    async def process_task(cfg: Dict[str, Any], node, tsk: str, tm: float, index: int) -> Dict[str, Any]:
      """Process a single task with its own subagent instance."""
      subagent_ref = None
      try:
        # Start subagent
        subagent_ref = await Agent.start_from_config(node=node, config=cfg)
        logger.debug(f"Parallel subagent {index} started: {subagent_ref.name}")

        # Send task and wait for response
        response = await subagent_ref.send(tsk, timeout=tm)
        result = response[-1].content.text if response else "Error: Empty response"

        return {
          "task": tsk,
          "result": result,
          "subagent": subagent_ref.name,
          "success": True,
        }
      except Exception as e:
        logger.warning(f"Parallel subagent {index} failed: {e}")
        return {
          "task": tsk,
          "error": str(e),
          "subagent": subagent_ref.name if subagent_ref else "not_started",
          "success": False,
        }
      finally:
        # Cleanup: stop the subagent after task completion
        if subagent_ref:
          try:
            await Agent.stop(node, subagent_ref.name)
            logger.debug(f"Parallel subagent {index} stopped: {subagent_ref.name}")
          except Exception as e:
            logger.warning(f"Error stopping parallel subagent {index}: {e}")

    # Create subagent instances for parallel processing
    subagent_futures = []

    for i, task in enumerate(tasks):
      # Generate unique name for each parallel subagent
      subagent_name = f"{self.agent.name}__subagent__{role}__parallel_{i}__{secrets.token_hex(4)}"

      # Build config for this subagent
      subagent_config = {
        "instructions": config["instructions"],
        "name": subagent_name,
      }

      # Copy optional fields
      for field in [
        "model",
        "tools",
        "max_iterations",
        "max_execution_time",
        "max_messages_in_short_term_memory",
        "max_tokens_in_short_term_memory",
        "enable_long_term_memory",
      ]:
        if field in config:
          subagent_config[field] = config[field]

      # Choose which node to start on (round-robin if runners available)
      target_node = self.agent.node
      if runners:
        target_node = runners[i % len(runners)]
        logger.debug(f"Task {i} assigned to runner: {target_node.name}")

      # Create task future
      future = process_task(subagent_config, target_node, task, timeout, i)
      subagent_futures.append(future)

    # Execute all tasks in parallel using asyncio.gather
    logger.debug(f"Executing {len(tasks)} tasks in parallel with asyncio.gather")
    results = await asyncio.gather(*subagent_futures, return_exceptions=True)

    # Process results, converting exceptions to error dicts
    processed_results = []
    for i, result in enumerate(results):
      if isinstance(result, Exception):
        logger.error(f"Task {i} raised exception: {result}")
        processed_results.append(
          {
            "task": tasks[i],
            "error": str(result),
            "success": False,
          }
        )
      else:
        processed_results.append(result)

    # Log summary
    success_count = sum(1 for r in processed_results if r.get("success", False))
    logger.info(f"Parallel delegation complete: {success_count}/{len(tasks)} tasks succeeded")

    return processed_results

  async def stop_subagent(self, role: str):
    """
    Stop a running subagent.

    Args:
      role: The role of the subagent to stop

    Raises:
      ValueError: If no running subagent found for the role
    """
    if role not in self.running_subagents:
      raise ValueError(f"No running subagent found for role: {role}")

    subagent_ref = self.running_subagents[role]
    logger.info(f"Stopping subagent '{role}': {subagent_ref.name}")

    # Import here to avoid circular dependency
    from ..agents.agent import Agent

    await Agent.stop(self.agent.node, subagent_ref.name)
    del self.running_subagents[role]
    logger.debug(f"Subagent '{role}' stopped successfully")

  def list_subagents(self) -> Dict[str, Any]:
    """
    List all configured and running subagents.

    Returns:
      Dict with:
        - configured: List of configured role names
        - running: List of currently running role names
        - details: Per-role configuration and status information
    """
    result = {
      "configured": list(self.subagent_configs.keys()),
      "running": list(self.running_subagents.keys()),
      "details": {},
    }

    for role, config in self.subagent_configs.items():
      result["details"][role] = {
        "instructions": config["instructions"],
        "auto_start": config.get("auto_start", False),
        "runner_filter": config.get("runner_filter"),
        "status": "running" if role in self.running_subagents else "available",
      }
      if role in self.running_subagents:
        result["details"][role]["subagent_name"] = self.running_subagents[role].name

    return result

  async def cleanup(self):
    """
    Stop all running subagents.

    This is typically called when the parent agent is stopping.
    """
    if not self.running_subagents:
      logger.debug("No running subagents to cleanup")
      return

    logger.info(f"Cleaning up {len(self.running_subagents)} running subagents")

    # Import here to avoid circular dependency
    from ..agents.agent import Agent

    # Stop all running subagents
    for role, subagent_ref in list(self.running_subagents.items()):
      try:
        logger.debug(f"Stopping subagent '{role}': {subagent_ref.name}")
        await Agent.stop(self.agent.node, subagent_ref.name)
      except Exception as e:
        logger.warning(f"Error stopping subagent '{role}': {e}")

    self.running_subagents.clear()
    logger.debug("Subagent cleanup complete")

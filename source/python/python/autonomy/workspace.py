"""
Workspace tools for Autonomy agents.

Provides unified filesystem and terminal access for agents.

Key Features:
- Terminal command execution (non-persistent, each command runs in a new process)
- File operations inherited from FilesystemTools
- Two modes: read-only, standard (default)
- Visibility levels: all, agent, scope, conversation

Workspace Modes:
  read-only: Read files only, no terminal, no write access
  standard:  Read/write workspace, terminal command execution (default)

Usage:
  from autonomy import Agent, Model, Node, Workspace, WorkspaceMode

  async def main(node):
    # Default: workspace with terminal access
    await Agent.start(
      node=node,
      name="code_assistant",
      tools=[Workspace()]
    )

    # Read-only mode (safe for analysis tasks)
    await Agent.start(
      node=node,
      name="reviewer",
      tools=[Workspace(mode=WorkspaceMode.READ_ONLY)]
    )

Runtime Environment:
  - In Autonomy Computer: Uses /workspace directory (emptyDir volume)
  - Locally: Uses /tmp/workspace
  - Environment detection via CLUSTER/ZONE environment variables (injected by provisioner)
"""

import os
import subprocess
from enum import Enum
from typing import Optional, List

from .logs import get_logger
from .tools.filesystem import FilesystemTools

logger = get_logger("workspace")


class WorkspaceMode(Enum):
  """Workspace operation modes."""

  READ_ONLY = "read-only"
  """Read files only, no shell execution, no write access."""

  STANDARD = "standard"
  """Read/write workspace with sandboxed shell execution (default)."""


def _is_running_in_autonomy_computer() -> bool:
  """
  Detect if we're running in Autonomy Computer.

  The provisioner injects CLUSTER and ZONE environment variables into all
  containers running on Autonomy Computer. This is the authoritative way
  to detect Autonomy Computer vs local execution.
  """
  return bool(os.environ.get("CLUSTER") or os.environ.get("ZONE"))


def _get_default_base_dir() -> str:
  """
  Get the default base directory for workspace files.

  In Autonomy Computer: /workspace (emptyDir volume mounted by provisioner)
  Locally: /tmp/workspace

  Falls back to /tmp/workspace if /workspace doesn't exist or isn't writable
  (allows testing with CLUSTER/ZONE env vars set locally).
  """
  if _is_running_in_autonomy_computer():
    cloud_workspace = "/workspace"
    # Check if cloud workspace exists and is writable
    if os.path.isdir(cloud_workspace) and os.access(cloud_workspace, os.W_OK):
      return cloud_workspace
    # Fall back if cloud workspace isn't available (local testing with env vars)
    logger.debug(f"Cloud workspace {cloud_workspace} not available, using local fallback")
  return "/tmp/workspace"


def _get_sandbox_module():
  """Get the sandbox module, returning None if not available."""
  try:
    from .autonomy_in_rust_for_python import sandbox

    return sandbox
  except ImportError:
    return None


class Workspace:
  """
  Unified workspace providing filesystem and shell access with sandboxing.

  Combines file operations (from FilesystemTools) with sandboxed shell
  execution. The sandbox uses Landlock for filesystem isolation and
  seccomp for syscall filtering on Linux.

  Attributes:
    visibility: Visibility level ("all", "agent", "scope", "conversation")
    mode: WorkspaceMode controlling access level
    base_dir: Base directory for workspace files
    agent_name: Agent name (None in factory mode)
    scope: Scope identifier (None in factory mode)
    conversation: Conversation identifier (None in factory mode)

  Examples:
    # Default: sandboxed shell + filesystem
    workspace = Workspace()

    # Read-only mode (no shell, read-only files)
    workspace = Workspace(mode=WorkspaceMode.READ_ONLY)

    # With specific visibility
    workspace = Workspace(visibility="scope")

    # Use with Agent
    agent = await Agent.start(
      node=node,
      tools=[Workspace()],
    )
  """

  def __init__(
    self,
    visibility: str = "conversation",
    mode: WorkspaceMode = WorkspaceMode.STANDARD,
    agent_name: Optional[str] = None,
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
    base_dir: Optional[str] = None,
    command_timeout: int = 60,
  ):
    """
    Initialize workspace with configurable isolation.

    Args:
      visibility: Visibility level - "all", "agent", "scope", or "conversation".
                  Defaults to "conversation".
      mode: WorkspaceMode controlling access level. Defaults to STANDARD.
      agent_name: Agent name (required for agent/scope/conversation visibility in direct mode)
      scope: Scope identifier (required for scope/conversation visibility in direct mode)
      conversation: Conversation identifier (required for conversation visibility in direct mode)
      base_dir: Base directory for workspace files. Auto-detected based on environment.
      command_timeout: Default timeout for shell commands in seconds. Defaults to 60.

    Raises:
      ValueError: If visibility is invalid.
    """
    valid_levels = ["all", "agent", "scope", "conversation"]
    if visibility not in valid_levels:
      raise ValueError(f"Invalid visibility '{visibility}'. Must be one of: {', '.join(valid_levels)}")

    if not isinstance(mode, WorkspaceMode):
      # Allow string mode for convenience
      mode = WorkspaceMode(mode)

    # Use appropriate base directory for the environment
    if base_dir is None:
      base_dir = _get_default_base_dir()

    self.visibility = visibility
    self.mode = mode
    self.base_dir = base_dir
    self.agent_name = agent_name
    self.scope = scope
    self.conversation = conversation
    self.command_timeout = command_timeout

    # Create underlying FilesystemTools for file operations
    self._fs_tools = FilesystemTools(
      visibility=visibility,
      agent_name=agent_name,
      scope=scope,
      conversation=conversation,
      base_dir=base_dir,
    )

    # Determine if we're in factory mode
    self._factory_mode = self._fs_tools.scope_root is None

    if not self._factory_mode:
      logger.debug(
        f"Initialized Workspace in direct mode: visibility='{visibility}', "
        f"mode='{mode.value}', path='{self._fs_tools.scope_root}'"
      )
    else:
      logger.debug(
        f"Initialized Workspace in factory mode: visibility='{visibility}', mode='{mode.value}', base_dir='{base_dir}'"
      )

  @property
  def workspace_path(self) -> Optional[str]:
    """Get the absolute path to the workspace directory."""
    return self._fs_tools.scope_root

  def terminal(self, command: str, timeout: Optional[int] = None) -> str:
    """
    Execute a shell command in the workspace.

    Each command runs in a new process with the workspace as the working
    directory. State does not persist between calls (e.g., `cd` in one call
    does not affect the next call).

    Args:
      command: The shell command to execute.
      timeout: Command timeout in seconds. If the command exceeds this limit,
               it is killed and an error is returned. Default: 60 seconds.

    Returns:
      Combined stdout and stderr output from the command.

    Raises:
      ValueError: If workspace is in read-only mode.
      TimeoutError: If command exceeds the timeout.
      RuntimeError: If command execution fails.
    """
    if self.mode == WorkspaceMode.READ_ONLY:
      raise ValueError("Command execution is disabled in read-only mode")

    if self._factory_mode:
      raise ValueError("Workspace not initialized. Use create_tools() first or provide full context.")

    timeout = timeout or self.command_timeout

    return self._execute_sandboxed(command, timeout)

  def _execute_sandboxed(self, command: str, timeout: int) -> str:
    """Execute command in a sandboxed subprocess using bash."""
    # Check if bash is available
    if not self._is_bash_available():
      raise RuntimeError(
        "bash is not available in this container. "
        "The terminal tool requires bash to execute commands. "
        "Use 'ghcr.io/build-trust/autonomy-python-dev' as your base image instead of "
        "'ghcr.io/build-trust/autonomy-python' to get bash support."
      )

    try:
      logger.debug(f"Executing command: {command[:100]}...")

      result = subprocess.run(
        ["/bin/bash", "-c", command],
        cwd=self.workspace_path,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=self._get_sandboxed_env(),
      )

      output = result.stdout
      if result.stderr:
        output += "\n" + result.stderr if output else result.stderr

      return output.strip() if output else ""

    except subprocess.TimeoutExpired:
      raise TimeoutError(f"Command timed out after {timeout} seconds: {command}")
    except Exception as e:
      raise RuntimeError(f"Command execution failed: {e}")

  def _get_sandboxed_env(self) -> dict:
    """Get environment variables for sandboxed execution."""
    env = os.environ.copy()

    # Set workspace-related variables
    env["HOME"] = self.workspace_path
    env["PWD"] = self.workspace_path
    env["WORKSPACE"] = self.workspace_path

    # Preserve PATH for command execution
    # but ensure workspace is not in PATH to prevent script injection
    if "PATH" in env:
      paths = env["PATH"].split(os.pathsep)
      safe_paths = [p for p in paths if not p.startswith(self.workspace_path)]
      env["PATH"] = os.pathsep.join(safe_paths)

    return env

  def _is_bash_available(self) -> bool:
    """Check if bash is available in the current environment."""
    if not hasattr(self, "_bash_available"):
      import shutil

      self._bash_available = shutil.which("bash") is not None
    return self._bash_available

  # Delegate file operations to FilesystemTools

  def list_directory(self, path: str = ".") -> str:
    """
    List contents of a directory in the workspace.

    Args:
      path: Directory path relative to workspace root. Defaults to ".".

    Returns:
      Formatted listing of directory contents.
    """
    return self._fs_tools.list_directory(path)

  def read_file(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """
    Read contents of a file in the workspace.

    Args:
      path: File path relative to workspace root.
      start_line: Optional starting line (1-indexed).
      end_line: Optional ending line (1-indexed, inclusive).

    Returns:
      File contents (or specified line range).
    """
    return self._fs_tools.read_file(path, start_line=start_line, end_line=end_line)

  def write_file(self, path: str, content: str) -> str:
    """
    Write content to a file in the workspace.

    Args:
      path: File path relative to workspace root.
      content: Content to write.

    Returns:
      Confirmation message.

    Raises:
      ValueError: If workspace is in read-only mode.
    """
    if self.mode == WorkspaceMode.READ_ONLY:
      raise ValueError("Write operations are disabled in read-only mode")
    return self._fs_tools.write_file(path, content)

  def edit_file(self, path: str, old_str: str, new_str: str) -> str:
    """
    Edit a file by replacing text.

    Args:
      path: File path relative to workspace root.
      old_str: Text to find and replace.
      new_str: Replacement text.

    Returns:
      Confirmation message.

    Raises:
      ValueError: If workspace is in read-only mode.
    """
    if self.mode == WorkspaceMode.READ_ONLY:
      raise ValueError("Edit operations are disabled in read-only mode")
    return self._fs_tools.edit_file(path, old_str, new_str)

  def find_files(self, pattern: str) -> str:
    """
    Find files matching a glob pattern.

    Args:
      pattern: Glob pattern to match (e.g., "*.py", "**/*.txt").

    Returns:
      List of matching file paths.
    """
    return self._fs_tools.find_files(pattern)

  def search_in_files(self, pattern: str, path: str = ".", context_lines: int = 2) -> str:
    """
    Search for a regex pattern in files.

    Args:
      pattern: Regex pattern to search for.
      path: Directory to search in. Defaults to workspace root.
      context_lines: Number of context lines around matches.

    Returns:
      Search results with context.
    """
    return self._fs_tools.search_in_files(pattern, path, context_lines)

  def remove_file(self, path: str) -> str:
    """
    Remove a file from the workspace.

    Args:
      path: File path relative to workspace root.

    Returns:
      Confirmation message.

    Raises:
      ValueError: If workspace is in read-only mode.
    """
    if self.mode == WorkspaceMode.READ_ONLY:
      raise ValueError("Remove operations are disabled in read-only mode")
    return self._fs_tools.remove_file(path)

  def remove_directory(self, path: str, recursive: bool = False) -> str:
    """
    Remove a directory from the workspace.

    Args:
      path: Directory path relative to workspace root.
      recursive: If True, remove directory and all contents.

    Returns:
      Confirmation message.

    Raises:
      ValueError: If workspace is in read-only mode.
    """
    if self.mode == WorkspaceMode.READ_ONLY:
      raise ValueError("Remove operations are disabled in read-only mode")
    return self._fs_tools.remove_directory(path, recursive)

  def create_tools(
    self,
    scope: Optional[str],
    conversation: Optional[str],
    agent_name: Optional[str] = None,
  ) -> list:
    """
    Create Workspace tool instances (ToolFactory protocol).

    This method enables Workspace to act as a ToolFactory. The spawner
    calls this method to create properly isolated instances based on
    the configured visibility level.

    Args:
      scope: Scope identifier (e.g., "user-alice").
      conversation: Conversation identifier (e.g., "chat-1").
      agent_name: Agent name. If None, uses "default".

    Returns:
      List of Tool instances for workspace operations.
    """
    from .tools.tool import Tool

    # Create a new Workspace instance with full context
    params = {
      "visibility": self.visibility,
      "mode": self.mode,
      "base_dir": self.base_dir,
      "command_timeout": self.command_timeout,
    }

    if self.visibility in ["agent", "scope", "conversation"]:
      params["agent_name"] = agent_name or self.agent_name or "default"

    if self.visibility in ["scope", "conversation"]:
      params["scope"] = scope or self.scope or "default"

    if self.visibility == "conversation":
      params["conversation"] = conversation or self.conversation or "default"

    ws = Workspace(**params)

    logger.debug(
      f"Created Workspace: visibility='{self.visibility}', "
      f"mode='{self.mode.value}', agent='{params.get('agent_name')}', "
      f"scope='{params.get('scope')}', conversation='{params.get('conversation')}', "
      f"path='{ws.workspace_path}'"
    )

    # Build tool list based on mode
    tools = [
      # File operations (always available)
      Tool(ws.list_directory),
      Tool(ws.read_file),
      Tool(ws.find_files),
      Tool(ws.search_in_files),
    ]

    # Write operations (not in read-only mode)
    if self.mode != WorkspaceMode.READ_ONLY:
      tools.extend(
        [
          Tool(ws.write_file),
          Tool(ws.edit_file),
          Tool(ws.remove_file),
          Tool(ws.remove_directory),
        ]
      )

    # Terminal (not in read-only mode)
    if self.mode != WorkspaceMode.READ_ONLY:
      tools.append(Tool(ws.terminal))

    return tools

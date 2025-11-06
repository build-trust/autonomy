"""
Filesystem tools for Autonomy agents.

Provides file operations with configurable visibility levels.

Key Features:
- Four visibility levels: all, agent, scope, conversation
- Security: Path validation prevents directory traversal attacks
- Virtual filesystem: Agents see "/" as the root of their visibility level
- Core operations: list_directory, read_file, write_file, edit_file, find_files, search_in_files
- Advanced features: partial file reading, string replacement, context-aware search

Visibility Levels:
  all:          {base_dir}/                              - Shared across all agents
  agent:        {base_dir}/{agent_name}/                 - Isolated per agent
  scope:        {base_dir}/{agent_name}/{scope}/         - Isolated per user/tenant
  conversation: {base_dir}/{agent_name}/{scope}/{conv}/  - Isolated per conversation (default)

Usage:
  # Factory mode with visibility level
  fs_tools = FilesystemTools()  # Default: conversation-level visibility
  fs_tools = FilesystemTools(visibility="scope")  # scope-level visibility

  # Use with Agent
  agent = await Agent.start(
    node=node,
    name="assistant",
    tools=[fs_tools],  # Automatically creates visibility-specific instances
  )

Security:
  All paths are validated to prevent directory traversal. Attempts to escape
  the visibility boundary (e.g., "../../../etc/passwd") will raise ValueError.

For practical guides:
- Tool usage: docs/_for-coding-agents/tools.mdx
- Agent deployment: docs/_for-coding-agents/create-a-new-autonomy-app.mdx
"""

import os
import re
import shutil
import glob as glob_module
from typing import Optional

from ..logs import get_logger

logger = get_logger("filesystem_tools")


class FilesystemTools:
  """
  Filesystem operations for Autonomy agents with configurable visibility.

  Can be used in two modes:
  1. Factory mode: Creates visibility-specific instances automatically based on visibility level
  2. Direct mode: Creates tools for specific agent/scope/conversation immediately

  Visibility Levels:
    all:          Base directory - shared across all agents and users
    agent:        Agent directory - separate per agent, shared across users
    scope:        Scope directory - separate per user/tenant
    conversation: Conversation directory - separate per conversation within a user (default)

  Attributes:
    visibility: Visibility level ("all", "agent", "scope", "conversation")
    base_dir: Base directory for all filesystem operations
    agent_name: Agent name (None in factory mode)
    scope: Scope identifier (None in factory mode for scope/conversation visibility)
    conversation: Conversation identifier (None in factory mode for conversation visibility)
    scope_root: Absolute path to the visibility root (None in factory mode)

  Examples:
    # Factory mode with different visibility levels
    fs = FilesystemTools()                           # Default - per-conversation visibility
    fs = FilesystemTools(visibility="conversation")  # Same as default
    fs = FilesystemTools(visibility="scope")         # Per-scope visibility
    fs = FilesystemTools(visibility="agent")         # Per-agent visibility
    fs = FilesystemTools(visibility="all")           # Shared across all

    agent = await Agent.start(
      node=node,
      tools=[fs],  # Spawner creates visibility-specific instances automatically
    )

    # Direct mode - specify all context upfront
    fs = FilesystemTools(
      visibility="scope",
      agent_name="assistant",
      scope="user-alice",
    )
    result = fs.write_file("notes.txt", "Hello World")
  """

  def __init__(
    self,
    visibility: str = "conversation",
    agent_name: Optional[str] = None,
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
    base_dir: str = "/tmp/agent-files",
  ):
    """
    Initialize filesystem tools with configurable visibility.

    Args:
      visibility: Visibility level - "all", "agent", "scope", or "conversation".
                  Defaults to "conversation".
      agent_name: Agent name (required for agent/scope/conversation visibility in direct mode)
      scope: Scope identifier (required for scope/conversation visibility in direct mode)
      conversation: Conversation identifier (required for conversation visibility in direct mode)
      base_dir: Base directory for all operations. Defaults to /tmp/agent-files.

    Raises:
      ValueError: If visibility is invalid or required parameters are missing in direct mode.

    Examples:
      # Factory mode - parameters filled in by spawner
      fs = FilesystemTools()                           # Default: conversation-level visibility
      fs = FilesystemTools(visibility="conversation")  # Same as default
      fs = FilesystemTools(visibility="scope")         # Per-user visibility
      fs = FilesystemTools(visibility="agent")         # Per-agent visibility
      fs = FilesystemTools(visibility="all")           # Shared

      # Direct mode - all parameters specified
      fs = FilesystemTools(
        visibility="scope",
        agent_name="assistant",
        scope="user-alice",
      )
    """
    # Validate visibility level
    valid_levels = ["all", "agent", "scope", "conversation"]
    if visibility not in valid_levels:
      raise ValueError(f"Invalid visibility '{visibility}'. Must be one of: {', '.join(valid_levels)}")

    self.visibility = visibility
    self.base_dir = base_dir
    self.agent_name = agent_name
    self.scope = scope
    self.conversation = conversation

    # Determine if we're in factory mode
    factory_mode = False
    if visibility == "all":
      # 'all' visibility doesn't need any context
      factory_mode = False
    elif visibility == "agent":
      # Need agent_name
      factory_mode = agent_name is None
    elif visibility == "scope":
      # Need agent_name and scope
      factory_mode = agent_name is None or scope is None
    elif visibility == "conversation":
      # Need agent_name, scope, and conversation
      factory_mode = agent_name is None or scope is None or conversation is None

    if factory_mode:
      # Factory mode - will compute scope_root in create_tools()
      self.scope_root = None
      logger.debug(f"Initialized FilesystemTools in factory mode: visibility='{visibility}', base_dir='{base_dir}'")
      return

    # Direct mode - validate and set up scope_root immediately
    self._validate_and_setup_scope_root()

  def _validate_and_setup_scope_root(self):
    """
    Validate parameters and set up scope_root for direct mode.

    Raises:
      ValueError: If parameters are invalid or contain path traversal attempts.
    """
    # Validate agent_name if needed
    if self.visibility in ["agent", "scope", "conversation"]:
      if not self.agent_name or not self.agent_name.strip():
        raise ValueError(f"agent_name is required for visibility='{self.visibility}'")

      agent_name = self.agent_name.strip()
      if ".." in agent_name or "/" in agent_name or "\\" in agent_name:
        raise ValueError(f"Invalid agent_name '{agent_name}': cannot contain '..', '/', or '\\'")
      self.agent_name = agent_name

    # Validate scope if needed
    if self.visibility in ["scope", "conversation"]:
      if not self.scope or not self.scope.strip():
        raise ValueError(f"scope is required for visibility='{self.visibility}'")

      scope = self.scope.strip()
      if ".." in scope or "/" in scope or "\\" in scope:
        raise ValueError(f"Invalid scope '{scope}': cannot contain '..', '/', or '\\'")
      self.scope = scope

    # Validate conversation if needed
    if self.visibility == "conversation":
      if not self.conversation or not self.conversation.strip():
        raise ValueError(f"conversation is required for visibility='conversation'")

      conversation = self.conversation.strip()
      if ".." in conversation or "/" in conversation or "\\" in conversation:
        raise ValueError(f"Invalid conversation '{conversation}': cannot contain '..', '/', or '\\'")
      self.conversation = conversation

    # Build scope_root based on visibility level
    if self.visibility == "all":
      self.scope_root = self.base_dir
    elif self.visibility == "agent":
      self.scope_root = os.path.join(self.base_dir, self.agent_name)
    elif self.visibility == "scope":
      self.scope_root = os.path.join(self.base_dir, self.agent_name, self.scope)
    elif self.visibility == "conversation":
      self.scope_root = os.path.join(self.base_dir, self.agent_name, self.scope, self.conversation)

    # Create scope directory if it doesn't exist
    try:
      os.makedirs(self.scope_root, exist_ok=True)
      logger.debug(
        f"Initialized FilesystemTools: visibility='{self.visibility}', "
        f"agent='{self.agent_name}', scope='{self.scope}', "
        f"conversation='{self.conversation}', path='{self.scope_root}'"
      )
    except Exception as e:
      logger.error(f"Failed to create directory '{self.scope_root}': {e}")
      raise

  def _resolve_path(self, virtual_path: str) -> str:
    """
    Resolve a virtual path to a physical filesystem path.

    The virtual path is relative to the visibility root. Leading slashes are removed
    to ensure paths are relative. The resolved path is validated to ensure it
    stays within the visibility boundary.

    Args:
      virtual_path: Path as seen by the agent (e.g., "notes.txt", "/docs/readme.md")

    Returns:
      Absolute physical path on the filesystem

    Raises:
      ValueError: If path attempts to escape visibility boundary

    Security:
      This method is critical for security. It prevents directory traversal
      attacks by normalizing the path and checking it stays within the visibility root.

    Example:
      virtual_path: "notes/2024/ideas.md"
      Returns: "/tmp/agent-files/user-alice/notes/2024/ideas.md"
    """
    # Handle empty paths
    virtual_path = virtual_path.strip() if virtual_path else "."

    # Remove leading slash to make path relative
    if virtual_path.startswith("/"):
      virtual_path = virtual_path[1:]

    # Resolve to absolute path and normalize
    physical_path = os.path.normpath(os.path.join(self.scope_root, virtual_path))

    # Security check: ensure path is within scope_root
    if not physical_path.startswith(self.scope_root):
      logger.warning(
        f"Path traversal attempt: virtual='{virtual_path}' resolved to '{physical_path}' "
        f"outside scope '{self.scope_root}'"
      )
      raise ValueError(f"Path '{virtual_path}' attempts to escape visibility boundary")

    return physical_path

  def create_tools(
    self,
    scope: Optional[str],
    conversation: Optional[str],
    agent_name: Optional[str] = None,
  ) -> list:
    """
    Create FilesystemTools instance based on visibility level (ToolFactory protocol).

    This method enables FilesystemTools to act as a ToolFactory. The spawner
    calls this method to create visibility-specific instances based on the configured
    visibility level.

    Args:
      scope: Scope identifier (e.g., "user-alice", "tenant-123").
            If None, uses "default" for scope/conversation visibility.
      conversation: Conversation identifier (e.g., "chat-1", "session-abc").
                  If None, uses "default" for conversation visibility.
      agent_name: Agent name. If None, extracted from context or uses "default".

    Returns:
      List of Tool instances wrapping FilesystemTools methods

    Examples:
      # Used internally by the spawner
      fs = FilesystemTools(visibility="scope")
      tools = fs.create_tools(
        agent_name="assistant",
        scope="user-alice",
        conversation="chat-1",
      )

    Security:
      All parameters are validated to prevent directory traversal.
      Invalid values (containing "..", "/", or "\\") will raise ValueError.
    """
    from ..tools.tool import Tool

    # Determine what parameters are needed based on visibility
    params = {
      "visibility": self.visibility,
      "base_dir": self.base_dir,
    }

    if self.visibility in ["agent", "scope", "conversation"]:
      params["agent_name"] = agent_name or self.agent_name or "default"

    if self.visibility in ["scope", "conversation"]:
      params["scope"] = scope or self.scope or "default"

    if self.visibility == "conversation":
      params["conversation"] = conversation or self.conversation or "default"

    # Create FilesystemTools instance with appropriate parameters
    fs = FilesystemTools(**params)

    logger.debug(
      f"Created FilesystemTools: visibility='{self.visibility}', "
      f"agent='{params.get('agent_name')}', scope='{params.get('scope')}', "
      f"conversation='{params.get('conversation')}', path='{fs.scope_root}'"
    )

    # Return all filesystem tools wrapped as Tool instances
    return [
      Tool(fs.list_directory),
      Tool(fs.read_file),
      Tool(fs.write_file),
      Tool(fs.edit_file),
      Tool(fs.find_files),
      Tool(fs.search_in_files),
      Tool(fs.remove_file),
      Tool(fs.remove_directory),
    ]

  def list_directory(self, path: str = ".") -> str:
    """
    List files and directories at the given path.

    This tool displays the contents of a directory, showing all files and subdirectories.
    Directories are marked with a trailing slash (/), and files show their size in bytes.
    Use this to explore the filesystem structure and find files.

    Args:
      path: Directory path to list, relative to your visibility root.
            Default is "." (current/root directory).
            Examples: ".", "docs", "src/components"

    Returns:
      Formatted listing showing:
      - Directory name being listed
      - Each subdirectory with trailing /
      - Each file with its size in bytes
      - Error message if path doesn't exist or isn't a directory

    Examples:
      List root directory:
        >>> list_directory(".")
        Contents of '.':
          docs/
          src/
          README.md (1234 bytes)
          config.yaml (567 bytes)

      List subdirectory:
        >>> list_directory("src")
        Contents of 'src':
          components/
          utils/
          main.py (2048 bytes)

      Empty directory:
        >>> list_directory("empty")
        Empty directory: empty

      Path doesn't exist:
        >>> list_directory("nonexistent")
        Error: Path 'nonexistent' does not exist

    Use Cases:
      - Explore directory structure before reading files
      - Verify file existence before operations
      - Find files when you don't know exact names
      - Check if directories exist before writing files
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      physical_path = self._resolve_path(path)

      if not os.path.exists(physical_path):
        logger.debug(f"ls: Path '{path}' does not exist")
        return f"Error: Path '{path}' does not exist"

      if not os.path.isdir(physical_path):
        logger.debug(f"ls: Path '{path}' is not a directory")
        return f"Error: Path '{path}' is not a directory"

      items = os.listdir(physical_path)

      if not items:
        return f"Empty directory: {path}"

      # Sort and format items
      items.sort()
      result = [f"Contents of '{path}':"]

      for item in items:
        item_path = os.path.join(physical_path, item)
        if os.path.isdir(item_path):
          result.append(f"  {item}/")
        else:
          size = os.path.getsize(item_path)
          result.append(f"  {item} ({size} bytes)")

      logger.debug(f"list_directory: Listed {len(items)} items in '{path}'")
      return "\n".join(result)

    except ValueError as e:
      # Path validation error (security)
      return f"Error: {str(e)}"
    except Exception as e:
      logger.error(f"list_directory: Unexpected error listing '{path}': {e}")
      return f"Error listing '{path}': {str(e)}"

  def read_file(
    self,
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
  ) -> str:
    """
    Read the contents of a file, optionally specifying a line range.

    This tool reads text files and returns their contents. You can read the entire file
    or just a specific range of lines, which is useful for large files. The tool only
    works with UTF-8 encoded text files.

    Args:
      path: File path to read, relative to your visibility root.
            Examples: "notes.txt", "docs/readme.md", "config/app.yaml"

      start_line: Optional line number to start reading from (1-based, inclusive).
                  If specified without end_line, reads from start_line to end of file.
                  Default is None (read from beginning).

      end_line: Optional line number to stop reading at (1-based, inclusive).
                Must be >= start_line if both are specified.
                Default is None (read to end of file).

    Returns:
      File contents as string, or error message if:
      - File doesn't exist
      - Path is a directory, not a file
      - File is not valid UTF-8 text
      - Line range is invalid

    Examples:
      Read entire file:
        >>> read_file("notes.txt")
        "Line 1: My notes\nLine 2: More notes\nLine 3: Final notes"

      Read specific line range (lines 10-20):
        >>> read_file("log.txt", start_line=10, end_line=20)
        "Line 10: [INFO] Starting process\n...\nLine 20: [INFO] Process complete"

      Read from line 50 to end:
        >>> read_file("data.csv", start_line=50)
        "row50,value1,value2\nrow51,value3,value4\n..."

      Read first 100 lines:
        >>> read_file("large_file.txt", start_line=1, end_line=100)
        "First 100 lines of content..."

      File doesn't exist:
        >>> read_file("missing.txt")
        Error: File 'missing.txt' does not exist

      Not a text file:
        >>> read_file("image.jpg")
        Error: File 'image.jpg' is not a valid UTF-8 text file

    Use Cases:
      - Read configuration files to understand settings
      - Examine source code before making changes
      - Preview large log files without loading everything
      - Read documentation or README files
      - Check file contents before editing
      - Extract specific sections from large files

    Note:
      This tool is for text files only. Binary files (images, PDFs, executables)
      will return an error. For large files, use start_line/end_line to avoid
      loading the entire file into memory.
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      physical_path = self._resolve_path(path)

      if not os.path.exists(physical_path):
        logger.debug(f"read_file: File '{path}' does not exist")
        return f"Error: File '{path}' does not exist"

      if not os.path.isfile(physical_path):
        logger.debug(f"read_file: Path '{path}' is not a file")
        return f"Error: Path '{path}' is not a file"

      # Validate line range if specified
      if start_line is not None and start_line < 1:
        return f"Error: start_line must be >= 1, got {start_line}"

      if end_line is not None and end_line < 1:
        return f"Error: end_line must be >= 1, got {end_line}"

      if start_line is not None and end_line is not None and end_line < start_line:
        return f"Error: end_line ({end_line}) must be >= start_line ({start_line})"

      with open(physical_path, "r", encoding="utf-8") as f:
        if start_line is None and end_line is None:
          # Read entire file
          content = f.read()
          logger.debug(f"read_file: Read {len(content)} bytes from '{path}'")
          return content
        else:
          # Read specific line range
          lines = []
          for line_num, line in enumerate(f, 1):
            if start_line is not None and line_num < start_line:
              continue
            if end_line is not None and line_num > end_line:
              break
            lines.append(line)

          content = "".join(lines)
          logger.debug(
            f"read_file: Read lines {start_line or 1}-{end_line or 'end'} ({len(content)} bytes) from '{path}'"
          )
          return content

    except ValueError as e:
      # Path validation error (security)
      return f"Error: {str(e)}"
    except UnicodeDecodeError as e:
      logger.error(f"read_file: Cannot decode '{path}' as UTF-8: {e}")
      return f"Error: File '{path}' is not a valid UTF-8 text file"
    except Exception as e:
      logger.error(f"read_file: Unexpected error reading '{path}': {e}")
      return f"Error reading '{path}': {str(e)}"

  def write_file(self, path: str, content: str) -> str:
    """
    Write content to a file, creating or completely replacing it.

    This tool creates a new file or completely overwrites an existing file with the
    provided content. Parent directories are created automatically if they don't exist.
    Use this when you want to create a new file or replace an entire file's contents.

    WARNING: This completely replaces existing files. If you want to modify part of
    a file, use edit_file() instead. If you want to add to the end of a file, consider
    reading it first, modifying the content, then writing it back.

    Args:
      path: File path to write, relative to your visibility root.
            Parent directories will be created automatically.
            Examples: "config.yaml", "docs/readme.md", "src/new_file.py"

      content: Complete content to write to the file.
              The file will contain exactly this content after the operation.

    Returns:
      Success message showing bytes written, or error message.

    Examples:
      Create new file:
        >>> write_file("notes.txt", "My first note\nMy second note\n")
        Successfully wrote 30 bytes to 'notes.txt'

      Overwrite existing file:
        >>> write_file("config.yaml", "setting: value\nport: 8080\n")
        Successfully wrote 29 bytes to 'config.yaml'

      Create file in new directory:
        >>> write_file("docs/api/readme.md", "# API Documentation\n")
        Successfully wrote 20 bytes to 'docs/api/readme.md'
        (Note: docs/api/ directory created automatically)

      Create empty file:
        >>> write_file("placeholder.txt", "")
        Successfully wrote 0 bytes to 'placeholder.txt'

      Path traversal attempt blocked:
        >>> write_file("../../etc/passwd", "malicious")
        Error: Path '../../etc/passwd' attempts to escape visibility boundary

    Use Cases:
      - Create new configuration files
      - Generate new source code files
      - Create documentation or README files
      - Replace entire file contents after modifications
      - Create empty placeholder files
      - Save generated data or reports

    Best Practices:
      - For existing files, read them first with read_file() to see current contents
      - For partial modifications, use edit_file() to replace specific text
      - For configuration files, preserve formatting when possible
      - Double-check you're writing to the correct path
      - Be careful with existing files - this tool overwrites completely

    Security:
      All paths are validated to prevent directory traversal attacks.
      You cannot write outside your visibility boundary.
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      physical_path = self._resolve_path(path)

      # Create parent directories if needed
      parent_dir = os.path.dirname(physical_path)
      if parent_dir and parent_dir != self.scope_root:
        os.makedirs(parent_dir, exist_ok=True)

      with open(physical_path, "w", encoding="utf-8") as f:
        f.write(content)

      size = len(content)
      logger.debug(f"write_file: Wrote {size} bytes to '{path}'")
      return f"Successfully wrote {size} bytes to '{path}'"

    except ValueError as e:
      # Path validation error (security)
      return f"Error: {str(e)}"
    except Exception as e:
      logger.error(f"write_file: Unexpected error writing to '{path}': {e}")
      return f"Error writing to '{path}': {str(e)}"

  def remove_file(self, path: str) -> str:
    """
    Delete a file.

    This tool permanently deletes a file from the filesystem. Use with caution as
    this operation cannot be undone.

    Args:
      path: File path to delete, relative to your visibility root.
            Examples: "temp.txt", "logs/old.log", "cache/data.json"

    Returns:
      Success message, or error message if:
      - File doesn't exist
      - Path is a directory (use remove_directory instead)
      - Permission denied or other error

    Examples:
      Delete a file:
        >>> remove_file("temp.txt")
        Successfully deleted file 'temp.txt'

      File doesn't exist:
        >>> remove_file("nonexistent.txt")
        Error: File 'nonexistent.txt' does not exist

      Path is a directory:
        >>> remove_file("my_folder")
        Error: 'my_folder' is a directory, use remove_directory() instead

    Use Cases:
      - Clean up temporary files
      - Remove old log files
      - Delete cache files
      - Remove generated files no longer needed

    Warning:
      This operation is permanent and cannot be undone. Make sure you're
      deleting the correct file before calling this tool.
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      physical_path = self._resolve_path(path)

      if not os.path.exists(physical_path):
        return f"Error: File '{path}' does not exist"

      if os.path.isdir(physical_path):
        return f"Error: '{path}' is a directory, use delete_directory() instead"

      os.remove(physical_path)
      logger.debug(f"remove_file: Deleted file '{path}'")
      return f"Successfully deleted file '{path}'"

    except ValueError as e:
      # Path validation error (security)
      return f"Error: {str(e)}"
    except Exception as e:
      logger.error(f"remove_file: Unexpected error deleting '{path}': {e}")
      return f"Error deleting '{path}': {str(e)}"

  def remove_directory(self, path: str, recursive: bool = False) -> str:
    """
    Delete a directory.

    This tool permanently deletes a directory from the filesystem. By default, it only
    deletes empty directories for safety. Set recursive=True to delete directories
    with contents. Use with extreme caution as this operation cannot be undone.

    Args:
      path: Directory path to delete, relative to your visibility root.
            Examples: "old_folder", "temp/cache", "build/output"

      recursive: If False (default), only deletes empty directories (safety feature).
                If True, deletes directory and ALL its contents recursively.
                WARNING: recursive=True will permanently delete all files and
                subdirectories inside the target directory.

    Returns:
      Success message, or error message if:
      - Directory doesn't exist
      - Path is a file (use remove_file instead)
      - Directory not empty and recursive=False
      - Cannot delete root directory
      - Permission denied or other error

    Examples:
      Delete empty directory:
        >>> remove_directory("empty_folder")
        Successfully deleted empty directory 'empty_folder'

      Try to delete non-empty directory without recursive:
        >>> remove_directory("my_docs")
        Error: Directory 'my_docs' is not empty. Use recursive=True to delete non-empty directories

      Delete directory and all contents:
        >>> remove_directory("temp_files", recursive=True)
        Successfully deleted directory 'temp_files' and all its contents

      Directory doesn't exist:
        >>> remove_directory("nonexistent")
        Error: Directory 'nonexistent' does not exist

      Path is a file:
        >>> remove_directory("document.txt")
        Error: 'document.txt' is not a directory, use remove_file() instead

      Cannot delete root:
        >>> remove_directory(".")
        Error: Cannot delete the root directory

    Use Cases:
      - Clean up empty directories after moving files
      - Remove temporary directory trees
      - Delete build output directories
      - Clean up cache directories
      - Remove old project directories

    Safety Features:
      - Default mode (recursive=False) prevents accidental deletion of data
      - Cannot delete the visibility root directory
      - Path validation prevents directory traversal

    Warning:
      When using recursive=True, ALL files and subdirectories will be permanently
      deleted. This operation cannot be undone. Double-check the path before executing.

    Best Practices:
      - List directory contents first with list_directory() to verify
      - Use recursive=False (default) when possible
      - Be absolutely certain before using recursive=True
      - Test on non-critical directories first
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      physical_path = self._resolve_path(path)

      if not os.path.exists(physical_path):
        return f"Error: Directory '{path}' does not exist"

      if not os.path.isdir(physical_path):
        return f"Error: '{path}' is not a directory, use delete_file() instead"

      # Prevent deletion of the visibility root itself
      if os.path.normpath(physical_path) == os.path.normpath(self.scope_root):
        return f"Error: Cannot delete the root directory"

      if recursive:
        shutil.rmtree(physical_path)
        logger.debug(f"remove_directory: Recursively deleted directory '{path}'")
        return f"Successfully deleted directory '{path}' and all its contents"
      else:
        try:
          os.rmdir(physical_path)
          logger.debug(f"remove_directory: Deleted empty directory '{path}'")
          return f"Successfully deleted empty directory '{path}'"
        except OSError:
          return f"Error: Directory '{path}' is not empty. Use recursive=True to delete non-empty directories"

    except ValueError as e:
      # Path validation error (security)
      return f"Error: {str(e)}"
    except Exception as e:
      logger.error(f"remove_directory: Unexpected error deleting '{path}': {e}")
      return f"Error deleting directory '{path}': {str(e)}"

  def edit_file(
    self,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
  ) -> str:
    """
    Replace text in a file using exact string matching.

    This tool performs find-and-replace operations in files. By default, it requires
    the old_string to appear exactly once in the file (safety feature). Set replace_all=True
    to replace all occurrences. This is the recommended way to modify existing files.

    IMPORTANT: This performs exact string matching, not regex. The old_string must match
    exactly, including whitespace, indentation, and line breaks.

    Args:
      path: File path to edit, relative to your visibility root.
            Examples: "config.yaml", "src/main.py", "docs/readme.md"

      old_string: Exact text to find and replace.
                  Must match exactly including all whitespace and newlines.
                  If replace_all=False (default), must appear exactly once in the file.

      new_string: Text to replace old_string with.
                  Can be empty string to delete the old_string.
                  Must be different from old_string.

      replace_all: If False (default), requires old_string to be unique - fails if found
                  multiple times. This is a safety feature to prevent unintended changes.
                  If True, replaces all occurrences of old_string in the file.

    Returns:
      Success message showing number of replacements made, or error message if:
      - File doesn't exist
      - old_string not found in file
      - old_string appears multiple times and replace_all=False
      - old_string and new_string are identical

    Examples:
      Replace single occurrence (safe mode):
        >>> edit_file("config.yaml", "port: 8080", "port: 3000")
        Successfully replaced 1 occurrence in 'config.yaml'

      Replace all occurrences:
        >>> edit_file("app.py", "old_function_name", "new_function_name", replace_all=True)
        Successfully replaced 5 occurrences in 'app.py'

      Delete text (replace with empty string):
        >>> edit_file("readme.md", "TODO: Remove this line\n", "")
        Successfully replaced 1 occurrence in 'readme.md'

      Multi-line replacement:
        >>> edit_file("code.py", "def old():\n    return 1", "def new():\n    return 2")
        Successfully replaced 1 occurrence in 'code.py'

      Update configuration value:
        >>> edit_file("settings.json", '"debug": true', '"debug": false')
        Successfully replaced 1 occurrence in 'settings.json'

      Safety check - multiple occurrences found:
        >>> edit_file("data.txt", "value", "new_value")
        Error: Found 3 occurrences of old_string in 'data.txt'.
        Use replace_all=True to replace all occurrences, or make old_string more specific.

      Text not found:
        >>> edit_file("config.yaml", "nonexistent: value", "new: value")
        Error: old_string not found in 'config.yaml'

    Use Cases:
      - Update configuration values
      - Rename functions or variables
      - Fix typos or bugs in code
      - Update version numbers
      - Change URLs or paths
      - Remove or replace specific lines
      - Refactor code by renaming

    Best Practices:
      - Read the file first with read_file() to see current contents
      - Copy the exact text including all whitespace from read_file() output
      - Use replace_all=False (default) for safety when possible
      - For unique changes, include surrounding context in old_string
      - Test with small, specific strings before larger replacements
      - For multiple different changes, call edit_file() multiple times

    Common Mistakes to Avoid:
      - Mismatched whitespace (spaces vs tabs)
      - Missing or extra newlines
      - Using regex patterns (this is exact string matching only)
      - Not including enough context to make old_string unique
      - Trying to replace old_string with itself (not allowed)

    Safety Features:
      - Default mode requires old_string to be unique (prevents accidental mass changes)
      - old_string and new_string must be different
      - File must exist (won't create new files)
      - All paths validated to prevent directory traversal

    Alternative Tools:
      - For creating/overwriting files: use write_file()
      - For appending to end of file: read_file(), modify, then write_file()
      - For complex transformations: read_file(), process in Python, then write_file()
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      # Validate inputs
      if old_string == new_string:
        return "Error: old_string and new_string must be different"

      if not old_string:
        return "Error: old_string cannot be empty"

      physical_path = self._resolve_path(path)

      # Check file exists
      if not os.path.exists(physical_path):
        return f"Error: File '{path}' does not exist"

      if not os.path.isfile(physical_path):
        return f"Error: Path '{path}' is not a file"

      # Read current content
      try:
        with open(physical_path, "r", encoding="utf-8") as f:
          content = f.read()
      except UnicodeDecodeError:
        return f"Error: File '{path}' is not a valid UTF-8 text file"

      # Check if old_string exists
      if old_string not in content:
        return f"Error: old_string not found in '{path}'"

      # Count occurrences
      count = content.count(old_string)

      # Safety check: if not replace_all, old_string must be unique
      if not replace_all and count > 1:
        return (
          f"Error: Found {count} occurrences of old_string in '{path}'. "
          f"Use replace_all=True to replace all occurrences, or make old_string more specific."
        )

      # Perform replacement
      new_content = content.replace(old_string, new_string)

      # Write back
      with open(physical_path, "w", encoding="utf-8") as f:
        f.write(new_content)

      logger.debug(f"edit_file: Replaced {count} occurrence(s) in '{path}'")
      return f"Successfully replaced {count} occurrence(s) in '{path}'"

    except ValueError as e:
      # Path validation error (security)
      return f"Error: {str(e)}"
    except Exception as e:
      logger.error(f"edit_file: Unexpected error editing '{path}': {e}")
      return f"Error editing '{path}': {str(e)}"

  def find_files(self, pattern: str) -> str:
    """
    Find files matching a glob pattern.

    This tool searches for files using glob patterns (wildcards). It's useful when you
    know the file name pattern but not the exact location. Only returns files, not
    directories. Searches recursively when using ** pattern.

    Args:
      pattern: Glob pattern to match files against. Supports:
              - * : matches any characters in a filename
              - ** : matches any files/directories recursively
              - ? : matches a single character
              - [seq] : matches any character in seq
              - [!seq] : matches any character not in seq

              Examples: "*.txt", "**/*.py", "test_*.py", "docs/**/*.md"

    Returns:
      Formatted list of matching files with their sizes, or message if no matches found.
      Files are sorted alphabetically and show path relative to visibility root.

    Examples:
      Find all Python files recursively:
        >>> find_files("**/*.py")
        Files matching '**/*.py':
          src/main.py (1024 bytes)
          src/utils/helper.py (512 bytes)
          tests/test_main.py (768 bytes)

      Find all text files in current directory:
        >>> find_files("*.txt")
        Files matching '*.txt':
          notes.txt (234 bytes)
          readme.txt (567 bytes)

      Find all markdown files in docs:
        >>> find_files("docs/**/*.md")
        Files matching 'docs/**/*.md':
          docs/readme.md (1200 bytes)
          docs/api/endpoints.md (800 bytes)
          docs/guides/tutorial.md (1500 bytes)

      Find test files:
        >>> find_files("test_*.py")
        Files matching 'test_*.py':
          test_app.py (400 bytes)
          test_utils.py (350 bytes)

      Find config files with specific pattern:
        >>> find_files("*.{yaml,yml}")
        Files matching '*.{yaml,yml}':
          config.yaml (234 bytes)
          docker-compose.yml (456 bytes)

      No matches found:
        >>> find_files("*.nonexistent")
        No files match pattern '*.nonexistent'

    Use Cases:
      - Find all source files of a specific type
      - Locate test files
      - Find configuration files
      - Search for files when you don't know exact location
      - List all files with specific extension
      - Find files matching naming convention

    Pattern Examples:
      *.py          - All Python files in root
      **/*.py       - All Python files recursively
      src/**/*.js   - All JavaScript files under src/
      test_*.py     - All files starting with "test_"
      *.{yml,yaml}  - Files ending in .yml or .yaml
      data/*/*.csv  - CSV files in direct subdirs of data/
      **/*test*.py  - Files containing "test" anywhere in name

    Best Practices:
      - Use ** for recursive search across directories
      - Use specific patterns to limit results
      - Combine with search_in_files() to find files by content
      - Use list_directory() first to understand directory structure

    Note:
      - Only returns files, not directories
      - Results are sorted alphabetically
      - Paths are relative to your visibility root
      - Hidden files (starting with .) are included in results
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      # Remove leading slash from pattern
      if pattern.startswith("/"):
        pattern = pattern[1:]

      physical_pattern = os.path.join(self.scope_root, pattern)

      # Find matching files
      matches = glob_module.glob(physical_pattern, recursive=True)

      # Filter to only files and convert to virtual paths
      virtual_matches = []
      for match in matches:
        if os.path.isfile(match):
          virtual_path = os.path.relpath(match, self.scope_root)
          virtual_matches.append(virtual_path)

      if not virtual_matches:
        logger.debug(f"glob: No files match pattern '{pattern}'")
        return f"No files match pattern '{pattern}'"

      virtual_matches.sort()
      result = [f"Files matching '{pattern}':"]
      for vpath in virtual_matches:
        physical = os.path.join(self.scope_root, vpath)
        size = os.path.getsize(physical)
        result.append(f"  {vpath} ({size} bytes)")

      logger.debug(f"find_files: Found {len(virtual_matches)} files matching '{pattern}'")
      return "\n".join(result)

    except Exception as e:
      logger.error(f"find_files: Unexpected error searching for '{pattern}': {e}")
      return f"Error searching for '{pattern}': {str(e)}"

  def search_in_files(
    self,
    pattern: str,
    path: str = ".",
    case_sensitive: bool = True,
    context_lines: Optional[int] = None,
    show_line_numbers: bool = True,
    max_results: int = 100,
  ) -> str:
    """
    Search for a regex pattern in files.

    This tool searches through file contents using regular expressions. It's useful for
    finding specific text, code patterns, or TODO comments across your codebase.
    Recursively searches all text files in the specified path.

    Args:
      pattern: Regular expression pattern to search for.
              Supports full Python regex syntax.
              Examples: "TODO", "function.*main", "class \\w+", "import (os|sys)"

      path: Directory or file to search in, relative to visibility root.
            Default is "." (search entire visibility root).
            Examples: ".", "src", "docs/api", "main.py"

      case_sensitive: If True (default), search is case-sensitive.
                      If False, search ignores case (finds "TODO", "todo", "Todo", etc.)

      context_lines: Number of lines to show before and after each match.
                    Default is None (show only matching line).
                    Examples: 1, 2, 3 for surrounding context

      show_line_numbers: If True (default), include line numbers in output.
                        If False, show only file paths and matched content.

      max_results: Maximum number of matches to return (default 100).
                  Prevents overwhelming output for common patterns.
                  If exceeded, shows count of additional matches.

    Returns:
      Formatted list of matches showing:
      - File path (relative to visibility root)
      - Line number (if show_line_numbers=True)
      - Matched line content
      - Context lines (if context_lines specified)

      Or message if no matches found or error occurred.

    Examples:
      Simple text search:
        >>> search_in_files("TODO", ".")
        Matches for 'TODO' in '.':
          src/main.py:45: # TODO: Implement error handling
          docs/plan.md:12: TODO: Review architecture
          tests/test_app.py:89: # TODO: Add more test cases

      Case-insensitive search:
        >>> search_in_files("error", ".", case_sensitive=False)
        Matches for 'error' in '.':
          src/app.py:23: raise ValueError("Invalid input")
          src/logger.py:56: def log_error(message):
          tests/test.py:34: # Test ERROR handling

      Search with context lines:
        >>> search_in_files("def main", ".", context_lines=2)
        Matches for 'def main' in '.':
          src/app.py:42:
          src/app.py:43: if __name__ == "__main__":
          src/app.py:44:     def main():
          src/app.py:45:         app.run()
          src/app.py:46:

      Regex pattern search:
        >>> search_in_files("class \\w+:", "src")
        Matches for 'class \\w+:' in 'src':
          src/models.py:5: class User:
          src/models.py:15: class Product:
          src/handlers.py:8: class RequestHandler:

      Search in specific file:
        >>> search_in_files("import", "src/main.py")
        Matches for 'import' in 'src/main.py':
          src/main.py:1: import os
          src/main.py:2: import sys
          src/main.py:3: from typing import Optional

      Find function definitions:
        >>> search_in_files("^def ", ".", case_sensitive=True)
        Matches for '^def ' in '.':
          src/utils.py:10: def helper_function():
          src/utils.py:20: def another_helper():
          src/main.py:30: def main():

      No matches:
        >>> search_in_files("nonexistent_pattern", ".")
        No matches found for pattern 'nonexistent_pattern' in '.'

      Too many matches:
        >>> search_in_files("the", ".")
        Matches for 'the' in '.':
          [... 100 matches shown ...]
          ... (150 more matches)

      Invalid regex:
        >>> search_in_files("[invalid", ".")
        Error: Invalid regex pattern '[invalid': unterminated character set at position 0

    Use Cases:
      - Find TODO/FIXME comments across codebase
      - Search for function or class definitions
      - Find imports or dependencies
      - Locate configuration values
      - Search for error messages or logging statements
      - Find usage of specific variables or functions
      - Locate security issues or code patterns
      - Find documentation mentions

    Regex Pattern Examples:
      "TODO"                  - Find TODO comments
      "function.*\\("         - Find function calls
      "^import "              - Find import statements at line start
      "class \\w+:"           - Find class definitions
      "(error|warning|fail)"  - Find error/warning/fail mentions
      "\\d{3}-\\d{4}"        - Find phone numbers (simple)
      "https?://\\S+"         - Find URLs

    Best Practices:
      - Use specific patterns to avoid too many matches
      - Start with case-insensitive for exploratory searches
      - Use context_lines to understand surrounding code
      - Combine with find_files() to narrow search scope
      - Test regex patterns on small paths first
      - Use ^ and $ anchors for line start/end matching

    Performance Notes:
      - Binary files are automatically skipped
      - Max 100 matches returned by default (configurable)
      - Large files are processed efficiently line-by-line
      - Non-UTF-8 files are silently skipped

    Note:
      - This is regex search, not exact string matching
      - Special regex characters need escaping: . * + ? [ ] ( ) { } ^ $ | \\
      - Case sensitivity matters unless case_sensitive=False
      - Results limited to max_results to prevent overwhelming output
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      physical_path = self._resolve_path(path)

      if not os.path.exists(physical_path):
        logger.debug(f"search_in_files: Path '{path}' does not exist")
        return f"Error: Path '{path}' does not exist"

      # Compile regex pattern with case sensitivity
      try:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
      except re.error as e:
        logger.debug(f"search_in_files: Invalid regex pattern '{pattern}': {e}")
        return f"Error: Invalid regex pattern '{pattern}': {str(e)}"

      matches = []

      # Determine files to search
      if os.path.isfile(physical_path):
        files_to_search = [physical_path]
      else:
        # Recursively find all files in directory
        files_to_search = []
        for root, dirs, files in os.walk(physical_path):
          for filename in files:
            files_to_search.append(os.path.join(root, filename))

      # Search each file
      for filepath in files_to_search:
        try:
          with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

          for line_num, line in enumerate(lines, 1):
            if regex.search(line):
              virtual_path = os.path.relpath(filepath, self.scope_root)

              if context_lines is not None and context_lines > 0:
                # Show context lines before and after
                start_idx = max(0, line_num - 1 - context_lines)
                end_idx = min(len(lines), line_num + context_lines)

                for ctx_line_num in range(start_idx, end_idx):
                  ctx_line = lines[ctx_line_num].rstrip()
                  display_line_num = ctx_line_num + 1

                  if show_line_numbers:
                    if display_line_num == line_num:
                      # Matching line
                      matches.append(f"{virtual_path}:{display_line_num}: {ctx_line}")
                    else:
                      # Context line
                      matches.append(f"{virtual_path}-{display_line_num}- {ctx_line}")
                  else:
                    matches.append(f"{virtual_path}: {ctx_line}")

                # Add separator between matches if showing context
                if end_idx < len(lines):
                  matches.append("")
              else:
                # No context, just the matching line
                if show_line_numbers:
                  matches.append(f"{virtual_path}:{line_num}: {line.rstrip()}")
                else:
                  matches.append(f"{virtual_path}: {line.rstrip()}")

              # Check if we've hit max_results
              if len(matches) >= max_results:
                break

        except (UnicodeDecodeError, PermissionError):
          # Skip binary files or files we can't read
          continue
        except Exception as e:
          logger.warning(f"search_in_files: Error reading file '{filepath}': {e}")
          continue

        # Check if we've hit max_results
        if len(matches) >= max_results:
          break

      if not matches:
        logger.debug(f"search_in_files: No matches for pattern '{pattern}' in '{path}'")
        return f"No matches found for pattern '{pattern}' in '{path}'"

      result = [f"Matches for '{pattern}' in '{path}':"]
      result.extend(matches[:max_results])

      if len(matches) > max_results:
        result.append(f"... ({len(matches) - max_results} more matches)")

      logger.debug(f"search_in_files: Found {len(matches)} matches for '{pattern}' in '{path}'")
      return "\n".join(result)

    except ValueError as e:
      # Path validation error (security)
      return f"Error: {str(e)}"
    except Exception as e:
      logger.error(f"search_in_files: Unexpected error searching for '{pattern}' in '{path}': {e}")
      return f"Error searching for '{pattern}' in '{path}': {str(e)}"

  def copy_file(self, source: str, destination: str, recursive: bool = False) -> str:
    """
    Copy a file or directory to a new location.

    This tool copies files or directories within your visibility boundary. For directories,
    use recursive=True. The destination path is created if it doesn't exist.

    Args:
      source: Source file or directory path, relative to visibility root.
              Examples: "data.json", "src/utils", "docs/readme.md"

      destination: Destination path, relative to visibility root.
                  Examples: "backup/data.json", "lib/utils", "archive/readme.md"

      recursive: Required for copying directories. If False (default), only copies files.
                If True, copies directories and all their contents recursively.

    Returns:
      Success message, or error message if operation fails.

    Examples:
      Copy a file:
        >>> copy_file("config.yaml", "config.backup.yaml")
        Successfully copied 'config.yaml' to 'config.backup.yaml'

      Copy to different directory:
        >>> copy_file("report.pdf", "archive/2024/report.pdf")
        Successfully copied 'report.pdf' to 'archive/2024/report.pdf'

      Copy a directory:
        >>> copy_file("templates", "backup/templates", recursive=True)
        Successfully copied directory 'templates' to 'backup/templates'

      Source doesn't exist:
        >>> copy_file("missing.txt", "copy.txt")
        Error: Source 'missing.txt' does not exist

      Try to copy directory without recursive:
        >>> copy_file("my_folder", "backup/my_folder")
        Error: 'my_folder' is a directory. Use recursive=True to copy directories

    Use Cases:
      - Create backups before modifications
      - Duplicate configuration files
      - Copy templates for new projects
      - Archive files before cleanup
      - Create file copies for testing

    Note:
      Parent directories in the destination path are created automatically.
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      source_physical = self._resolve_path(source)
      dest_physical = self._resolve_path(destination)

      if not os.path.exists(source_physical):
        return f"Error: Source '{source}' does not exist"

      # Check if source is a directory
      if os.path.isdir(source_physical):
        if not recursive:
          return f"Error: '{source}' is a directory. Use recursive=True to copy directories"

        # Copy directory recursively
        shutil.copytree(source_physical, dest_physical, dirs_exist_ok=True)
        logger.debug(f"copy_file: Copied directory '{source}' to '{destination}'")
        return f"Successfully copied directory '{source}' to '{destination}'"
      else:
        # Copy file - create parent directories if needed
        dest_parent = os.path.dirname(dest_physical)
        if dest_parent and dest_parent != self.scope_root:
          os.makedirs(dest_parent, exist_ok=True)

        shutil.copy2(source_physical, dest_physical)
        logger.debug(f"copy_file: Copied file '{source}' to '{destination}'")
        return f"Successfully copied '{source}' to '{destination}'"

    except ValueError as e:
      return f"Error: {str(e)}"
    except Exception as e:
      logger.error(f"copy_file: Unexpected error copying '{source}' to '{destination}': {e}")
      return f"Error copying '{source}' to '{destination}': {str(e)}"

  def move_file(self, source: str, destination: str) -> str:
    """
    Move or rename a file or directory.

    This tool moves files or directories to a new location, or renames them if the
    destination is in the same directory. The operation is atomic for files on the
    same filesystem.

    Args:
      source: Source file or directory path, relative to visibility root.
              Examples: "draft.txt", "old_name.py", "temp_folder"

      destination: Destination path, relative to visibility root.
                  Examples: "final.txt", "new_name.py", "archive/temp_folder"

    Returns:
      Success message, or error message if operation fails.

    Examples:
      Rename a file:
        >>> move_file("draft.md", "final.md")
        Successfully moved 'draft.md' to 'final.md'

      Move file to different directory:
        >>> move_file("report.pdf", "archive/2024/report.pdf")
        Successfully moved 'report.pdf' to 'archive/2024/report.pdf'

      Rename a directory:
        >>> move_file("old_folder", "new_folder")
        Successfully moved 'old_folder' to 'new_folder'

      Move directory to new location:
        >>> move_file("temp", "archive/temp")
        Successfully moved 'temp' to 'archive/temp'

      Source doesn't exist:
        >>> move_file("missing.txt", "new.txt")
        Error: Source 'missing.txt' does not exist

    Use Cases:
      - Rename files or directories
      - Move files to different directories
      - Reorganize project structure
      - Archive old files
      - Move processed files to output directory

    Note:
      - This is a move operation, not a copy - the source will no longer exist
      - Parent directories in destination path are created automatically
      - Works for both files and directories
      - On same filesystem, this is an atomic operation
    """
    if self.scope_root is None:
      return (
        "Error: FilesystemTools is in factory mode. Use with Agent.start() to create visibility-specific instances."
      )

    try:
      source_physical = self._resolve_path(source)
      dest_physical = self._resolve_path(destination)

      if not os.path.exists(source_physical):
        return f"Error: Source '{source}' does not exist"

      # Create parent directories if needed
      dest_parent = os.path.dirname(dest_physical)
      if dest_parent and dest_parent != self.scope_root:
        os.makedirs(dest_parent, exist_ok=True)

      shutil.move(source_physical, dest_physical)
      logger.debug(f"move_file: Moved '{source}' to '{destination}'")
      return f"Successfully moved '{source}' to '{destination}'"

    except ValueError as e:
      return f"Error: {str(e)}"
    except Exception as e:
      logger.error(f"move_file: Unexpected error moving '{source}' to '{destination}': {e}")
      return f"Error moving '{source}' to '{destination}': {str(e)}"


__all__ = ["FilesystemTools"]

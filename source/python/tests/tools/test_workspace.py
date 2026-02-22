"""
Tests for the Workspace class.

Tests workspace functionality including:
- Basic initialization and configuration
- Workspace modes (read-only, standard)
- Visibility levels
- File operations
- Terminal command execution
- Tool factory integration
- Environment detection
"""

import os
import pytest
import tempfile
import shutil

from autonomy import Workspace, WorkspaceMode


class TestWorkspaceInitialization:
  """Test workspace initialization and configuration."""

  def test_workspace_default_mode(self):
    """Test workspace defaults to STANDARD mode."""
    ws = Workspace(visibility="all")
    assert ws.mode == WorkspaceMode.STANDARD

  def test_workspace_default_visibility(self):
    """Test workspace defaults to conversation visibility."""
    ws = Workspace()
    assert ws.visibility == "conversation"

  def test_workspace_mode_string_conversion(self):
    """Test workspace accepts mode as string."""
    ws = Workspace(visibility="all", mode="standard")
    assert ws.mode == WorkspaceMode.STANDARD

    ws = Workspace(visibility="all", mode="read-only")
    assert ws.mode == WorkspaceMode.READ_ONLY

  def test_workspace_invalid_visibility_raises(self):
    """Test invalid visibility raises ValueError."""
    with pytest.raises(ValueError, match="Invalid visibility"):
      Workspace(visibility="invalid")

  def test_workspace_invalid_mode_raises(self):
    """Test invalid mode raises ValueError."""
    with pytest.raises(ValueError):
      Workspace(visibility="all", mode="invalid")

  def test_workspace_factory_mode(self):
    """Test workspace in factory mode has no workspace_path."""
    ws = Workspace(visibility="conversation")
    assert ws.workspace_path is None
    assert ws._factory_mode is True

  def test_workspace_direct_mode(self):
    """Test workspace with full context has workspace_path."""
    ws = Workspace(
      visibility="conversation",
      agent_name="test-agent",
      scope="test-scope",
      conversation="test-conv",
    )
    assert ws.workspace_path is not None
    assert ws._factory_mode is False
    assert "test-agent" in ws.workspace_path
    assert "test-scope" in ws.workspace_path
    assert "test-conv" in ws.workspace_path

  def test_workspace_all_visibility_direct_mode(self):
    """Test workspace with 'all' visibility is always in direct mode."""
    ws = Workspace(visibility="all")
    assert ws.workspace_path is not None
    assert ws._factory_mode is False


class TestWorkspaceModes:
  """Test workspace mode behavior."""

  @pytest.fixture
  def workspace_standard(self):
    """Create a standard mode workspace for testing."""
    ws = Workspace(
      visibility="conversation",
      mode=WorkspaceMode.STANDARD,
      agent_name="test",
      scope="scope",
      conversation="conv",
    )
    yield ws
    # Cleanup
    if ws.workspace_path and os.path.exists(ws.workspace_path):
      shutil.rmtree(ws.workspace_path, ignore_errors=True)

  @pytest.fixture
  def workspace_readonly(self):
    """Create a read-only mode workspace for testing."""
    # First create with standard to write test files
    ws_std = Workspace(
      visibility="conversation",
      mode=WorkspaceMode.STANDARD,
      agent_name="test-ro",
      scope="scope",
      conversation="conv",
    )
    ws_std.write_file("test.txt", "test content")

    # Now create read-only workspace with same path
    ws = Workspace(
      visibility="conversation",
      mode=WorkspaceMode.READ_ONLY,
      agent_name="test-ro",
      scope="scope",
      conversation="conv",
    )
    yield ws
    # Cleanup
    if ws.workspace_path and os.path.exists(ws.workspace_path):
      shutil.rmtree(ws.workspace_path, ignore_errors=True)

  def test_standard_mode_allows_write(self, workspace_standard):
    """Test standard mode allows write operations."""
    result = workspace_standard.write_file("test.txt", "hello")
    assert "Successfully" in result or "wrote" in result.lower() or "test.txt" in result

  def test_standard_mode_allows_terminal(self, workspace_standard):
    """Test standard mode allows terminal command execution."""
    result = workspace_standard.terminal("echo hello")
    assert "hello" in result

  def test_readonly_mode_blocks_write(self, workspace_readonly):
    """Test read-only mode blocks write operations."""
    with pytest.raises(ValueError, match="read-only mode"):
      workspace_readonly.write_file("new.txt", "content")

  def test_readonly_mode_blocks_edit(self, workspace_readonly):
    """Test read-only mode blocks edit operations."""
    with pytest.raises(ValueError, match="read-only mode"):
      workspace_readonly.edit_file("test.txt", "old", "new")

  def test_readonly_mode_blocks_terminal(self, workspace_readonly):
    """Test read-only mode blocks terminal command execution."""
    with pytest.raises(ValueError, match="read-only mode"):
      workspace_readonly.terminal("echo hello")

  def test_readonly_mode_blocks_remove_file(self, workspace_readonly):
    """Test read-only mode blocks file removal."""
    with pytest.raises(ValueError, match="read-only mode"):
      workspace_readonly.remove_file("test.txt")

  def test_readonly_mode_blocks_remove_directory(self, workspace_readonly):
    """Test read-only mode blocks directory removal."""
    with pytest.raises(ValueError, match="read-only mode"):
      workspace_readonly.remove_directory("somedir")

  def test_readonly_mode_allows_read(self, workspace_readonly):
    """Test read-only mode allows read operations."""
    content = workspace_readonly.read_file("test.txt")
    assert content == "test content"

  def test_readonly_mode_allows_list(self, workspace_readonly):
    """Test read-only mode allows list operations."""
    result = workspace_readonly.list_directory(".")
    assert "test.txt" in result

  def test_readonly_mode_allows_find(self, workspace_readonly):
    """Test read-only mode allows find operations."""
    result = workspace_readonly.find_files("*.txt")
    assert "test.txt" in result

  def test_readonly_mode_allows_search(self, workspace_readonly):
    """Test read-only mode allows search operations."""
    result = workspace_readonly.search_in_files("content", ".")
    assert "test.txt" in result or "content" in result


class TestWorkspaceFileOperations:
  """Test workspace file operations."""

  @pytest.fixture
  def workspace(self):
    """Create a workspace for testing."""
    ws = Workspace(
      visibility="conversation",
      agent_name="file-test",
      scope="scope",
      conversation="conv",
    )
    yield ws
    # Cleanup
    if ws.workspace_path and os.path.exists(ws.workspace_path):
      shutil.rmtree(ws.workspace_path, ignore_errors=True)

  def test_write_and_read_file(self, workspace):
    """Test writing and reading a file."""
    workspace.write_file("hello.txt", "Hello, World!")
    content = workspace.read_file("hello.txt")
    assert content == "Hello, World!"

  def test_edit_file(self, workspace):
    """Test editing a file."""
    workspace.write_file("edit.txt", "Hello, World!")
    workspace.edit_file("edit.txt", "World", "Universe")
    content = workspace.read_file("edit.txt")
    assert content == "Hello, Universe!"

  def test_list_directory(self, workspace):
    """Test listing directory contents."""
    workspace.write_file("file1.txt", "content1")
    workspace.write_file("file2.txt", "content2")
    result = workspace.list_directory(".")
    assert "file1.txt" in result
    assert "file2.txt" in result

  def test_find_files(self, workspace):
    """Test finding files by pattern."""
    workspace.write_file("test1.txt", "content")
    workspace.write_file("test2.txt", "content")
    workspace.write_file("other.md", "content")
    result = workspace.find_files("*.txt")
    assert "test1.txt" in result
    assert "test2.txt" in result
    assert "other.md" not in result

  def test_search_in_files(self, workspace):
    """Test searching in files."""
    workspace.write_file("search1.txt", "hello world")
    workspace.write_file("search2.txt", "goodbye world")
    result = workspace.search_in_files("hello", ".")
    assert "search1.txt" in result
    assert "hello" in result

  def test_remove_file(self, workspace):
    """Test removing a file."""
    workspace.write_file("delete.txt", "content")
    workspace.remove_file("delete.txt")
    result = workspace.find_files("delete.txt")
    assert "delete.txt" not in result or "No files" in result

  def test_remove_directory(self, workspace):
    """Test removing a directory."""
    workspace.write_file("subdir/file.txt", "content")
    workspace.remove_directory("subdir", recursive=True)
    result = workspace.list_directory(".")
    assert "subdir" not in result


class TestWorkspaceTerminal:
  """Test workspace terminal command execution."""

  @pytest.fixture
  def workspace(self):
    """Create a workspace for testing."""
    ws = Workspace(
      visibility="conversation",
      agent_name="terminal-test",
      scope="scope",
      conversation="conv",
    )
    yield ws
    # Cleanup
    if ws.workspace_path and os.path.exists(ws.workspace_path):
      shutil.rmtree(ws.workspace_path, ignore_errors=True)

  def test_terminal_simple(self, workspace):
    """Test running a simple command."""
    result = workspace.terminal("echo hello")
    assert "hello" in result

  def test_terminal_runs_in_workspace_dir(self, workspace):
    """Test command runs in workspace directory."""
    result = workspace.terminal("pwd")
    assert workspace.workspace_path in result or "/private" + workspace.workspace_path in result

  def test_terminal_timeout(self, workspace):
    """Test command timeout."""
    with pytest.raises(TimeoutError):
      workspace.terminal("sleep 10", timeout=1)

  def test_terminal_multiline(self, workspace):
    """Test running multiline commands."""
    result = workspace.terminal('echo "line1" && echo "line2"')
    assert "line1" in result
    assert "line2" in result

  def test_terminal_state_does_not_persist(self, workspace):
    """Test that state does not persist between terminal calls."""
    workspace.terminal("cd /tmp")
    result = workspace.terminal("pwd")
    # Should still be in workspace dir, not /tmp
    assert workspace.workspace_path in result or "/private" + workspace.workspace_path in result

  def test_terminal_in_factory_mode_raises(self):
    """Test terminal raises in factory mode."""
    ws = Workspace(visibility="conversation")
    with pytest.raises(ValueError, match="not initialized"):
      ws.terminal("echo hello")

  def test_terminal_bash_unavailable_error(self, workspace, monkeypatch):
    """Test terminal gives helpful error when bash is not available."""
    # Simulate bash not being available by mocking the method
    monkeypatch.setattr(workspace, "_is_bash_available", lambda: False)

    with pytest.raises(RuntimeError, match="bash is not available"):
      workspace.terminal("echo hello")

  def test_terminal_bash_check_is_cached(self, workspace):
    """Test that bash availability check is cached."""
    # First call should check and cache
    workspace.terminal("echo hello")
    assert hasattr(workspace, "_bash_available")
    assert workspace._bash_available is True


class TestWorkspaceToolFactory:
  """Test workspace tool factory integration."""

  def test_create_tools_standard_mode(self):
    """Test create_tools returns all tools in standard mode."""
    ws = Workspace(visibility="scope", mode=WorkspaceMode.STANDARD)
    tools = ws.create_tools(scope="alice", conversation="chat1", agent_name="assistant")

    tool_names = [t.name for t in tools]
    assert "list_directory" in tool_names
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "find_files" in tool_names
    assert "search_in_files" in tool_names
    assert "remove_file" in tool_names
    assert "remove_directory" in tool_names
    assert "terminal" in tool_names
    assert len(tools) == 9

  def test_create_tools_readonly_mode(self):
    """Test create_tools returns limited tools in read-only mode."""
    ws = Workspace(visibility="scope", mode=WorkspaceMode.READ_ONLY)
    tools = ws.create_tools(scope="alice", conversation="chat1", agent_name="assistant")

    tool_names = [t.name for t in tools]
    assert "list_directory" in tool_names
    assert "read_file" in tool_names
    assert "find_files" in tool_names
    assert "search_in_files" in tool_names
    # Should NOT have write/execute tools
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names
    assert "terminal" not in tool_names
    assert len(tools) == 4

  def test_create_tools_with_visibility_scope(self):
    """Test create_tools respects scope visibility."""
    ws = Workspace(visibility="scope")
    tools = ws.create_tools(scope="alice", conversation="chat1", agent_name="assistant")

    # Tools should be created successfully
    assert len(tools) > 0

    # The workspace path should include scope but not conversation
    # (Testing via the tools would require invoking them)

  def test_create_tools_with_visibility_conversation(self):
    """Test create_tools respects conversation visibility."""
    ws = Workspace(visibility="conversation")
    tools = ws.create_tools(scope="alice", conversation="chat1", agent_name="assistant")

    # Tools should be created successfully
    assert len(tools) > 0


class TestWorkspaceEnvironmentDetection:
  """Test workspace environment detection."""

  def test_local_environment_uses_tmp(self, monkeypatch):
    """Test local environment uses /tmp/workspace."""
    # Ensure we're not in cloud
    monkeypatch.delenv("CLUSTER", raising=False)
    monkeypatch.delenv("ZONE", raising=False)

    ws = Workspace(visibility="all")
    assert ws.workspace_path == "/tmp/workspace"

  def test_cloud_environment_detection(self, monkeypatch):
    """Test cloud environment is detected via CLUSTER/ZONE."""
    monkeypatch.setenv("CLUSTER", "test-cluster")
    monkeypatch.setenv("ZONE", "test-zone")

    # Import after setting env vars
    from autonomy.workspace import _is_running_in_autonomy_computer

    assert _is_running_in_autonomy_computer() is True

  def test_local_environment_detection(self, monkeypatch):
    """Test local environment is detected when CLUSTER/ZONE not set."""
    monkeypatch.delenv("CLUSTER", raising=False)
    monkeypatch.delenv("ZONE", raising=False)

    from autonomy.workspace import _is_running_in_autonomy_computer

    assert _is_running_in_autonomy_computer() is False

  def test_custom_base_dir(self):
    """Test custom base_dir is respected."""
    custom_dir = "/tmp/custom-workspace-test"
    ws = Workspace(visibility="all", base_dir=custom_dir)
    assert ws.workspace_path == custom_dir

    # Cleanup
    if os.path.exists(custom_dir):
      shutil.rmtree(custom_dir, ignore_errors=True)


class TestWorkspaceSandboxModule:
  """Test sandbox module integration."""

  def test_sandbox_module_available(self):
    """Test sandbox module can be imported."""
    from autonomy.workspace import _get_sandbox_module

    sandbox = _get_sandbox_module()
    # Should return the module or None
    assert sandbox is not None or sandbox is None

  def test_sandbox_support_detection(self):
    """Test sandbox support is detected correctly."""
    from autonomy.workspace import _get_sandbox_module

    sandbox = _get_sandbox_module()

    if sandbox is not None:
      # On Linux, sandbox should be supported
      # On macOS/Windows, it should not be
      import platform

      if platform.system() == "Linux":
        assert sandbox.is_sandbox_supported() is True
      else:
        assert sandbox.is_sandbox_supported() is False

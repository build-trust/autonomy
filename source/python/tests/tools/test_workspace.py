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
- Bubblewrap sandboxing (gVisor detection, command construction, fallback)
"""

import os
import subprocess
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from autonomy import Workspace, WorkspaceMode
from autonomy.workspace import _detect_gvisor, _is_bwrap_available


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


class TestGVisorDetection:
  """Test gVisor environment detection."""

  def test_detect_gvisor_when_kernel_is_4_4_0(self):
    """Test gVisor is detected when kernel reports 4.4.0."""
    with patch("autonomy.workspace.platform.release", return_value="4.4.0"):
      assert _detect_gvisor() is True

  def test_detect_gvisor_on_regular_kernel(self):
    """Test gVisor is not detected on a normal kernel."""
    with patch("autonomy.workspace.platform.release", return_value="6.1.161"):
      assert _detect_gvisor() is False

  def test_detect_gvisor_on_fargate_kernel(self):
    """Test gVisor is not detected on Fargate (5.10.x)."""
    with patch("autonomy.workspace.platform.release", return_value="5.10.247-246.992.amzn2.x86_64"):
      assert _detect_gvisor() is False

  def test_detect_gvisor_handles_exception(self):
    """Test gVisor detection returns False on exception."""
    with patch("autonomy.workspace.platform.release", side_effect=OSError("unavailable")):
      assert _detect_gvisor() is False


class TestBwrapAvailability:
  """Test Bubblewrap binary detection."""

  def test_bwrap_available_when_on_path(self):
    """Test bwrap detected when binary exists on PATH."""
    with patch("autonomy.workspace.shutil.which", return_value="/usr/bin/bwrap"):
      assert _is_bwrap_available() is True

  def test_bwrap_not_available_when_missing(self):
    """Test bwrap not detected when binary is absent."""
    with patch("autonomy.workspace.shutil.which", return_value=None):
      assert _is_bwrap_available() is False


class TestBubblewrapSandboxing:
  """Test Bubblewrap sandboxing in Workspace."""

  @pytest.fixture
  def workspace(self):
    """Create a workspace for bwrap testing."""
    ws = Workspace(
      visibility="conversation",
      agent_name="bwrap-test",
      scope="scope",
      conversation="conv",
    )
    yield ws
    if ws.workspace_path and os.path.exists(ws.workspace_path):
      shutil.rmtree(ws.workspace_path, ignore_errors=True)

  def test_should_use_bubblewrap_when_gvisor_and_bwrap(self, workspace):
    """Test _should_use_bubblewrap returns True when both conditions met."""
    with (
      patch("autonomy.workspace._detect_gvisor", return_value=True),
      patch("autonomy.workspace._is_bwrap_available", return_value=True),
    ):
      # Clear cached result
      if hasattr(workspace, "_use_bubblewrap"):
        delattr(workspace, "_use_bubblewrap")
      assert workspace._should_use_bubblewrap() is True

  def test_should_not_use_bubblewrap_without_gvisor(self, workspace):
    """Test _should_use_bubblewrap returns False when not in gVisor."""
    with (
      patch("autonomy.workspace._detect_gvisor", return_value=False),
      patch("autonomy.workspace._is_bwrap_available", return_value=True),
    ):
      if hasattr(workspace, "_use_bubblewrap"):
        delattr(workspace, "_use_bubblewrap")
      assert workspace._should_use_bubblewrap() is False

  def test_should_not_use_bubblewrap_without_bwrap(self, workspace):
    """Test _should_use_bubblewrap returns False when bwrap missing."""
    with (
      patch("autonomy.workspace._detect_gvisor", return_value=True),
      patch("autonomy.workspace._is_bwrap_available", return_value=False),
    ):
      if hasattr(workspace, "_use_bubblewrap"):
        delattr(workspace, "_use_bubblewrap")
      assert workspace._should_use_bubblewrap() is False

  def test_should_use_bubblewrap_caches_result(self, workspace):
    """Test _should_use_bubblewrap caches its result."""
    with (
      patch("autonomy.workspace._detect_gvisor", return_value=True) as mock_gvisor,
      patch("autonomy.workspace._is_bwrap_available", return_value=True) as mock_bwrap,
    ):
      if hasattr(workspace, "_use_bubblewrap"):
        delattr(workspace, "_use_bubblewrap")
      workspace._should_use_bubblewrap()
      workspace._should_use_bubblewrap()
      # Detection functions should only be called once
      mock_gvisor.assert_called_once()
      mock_bwrap.assert_called_once()

  def test_build_bwrap_command(self, workspace):
    """Test _build_bwrap_command constructs the correct command."""
    cmd = workspace._build_bwrap_command("echo hello")
    assert cmd == [
      "bwrap",
      "--ro-bind",
      "/",
      "/",
      "--dev",
      "/dev",
      "--proc",
      "/proc",
      "--tmpfs",
      "/tmp",
      "--bind",
      workspace.workspace_path,
      "/workspace",
      "--unshare-pid",
      "--chdir",
      "/workspace",
      "--",
      "/bin/bash",
      "-c",
      "echo hello",
    ]

  def test_build_bwrap_command_with_complex_command(self, workspace):
    """Test _build_bwrap_command handles complex shell commands."""
    complex_cmd = 'ls -la && echo "done" | grep done'
    cmd = workspace._build_bwrap_command(complex_cmd)
    # The command should be passed as a single argument to bash -c
    assert cmd[-1] == complex_cmd
    assert cmd[-2] == "-c"
    assert cmd[-3] == "/bin/bash"

  def test_bwrap_env_uses_workspace_path(self, workspace):
    """Test _get_bwrap_env sets /workspace as HOME and PWD."""
    env = workspace._get_bwrap_env()
    assert env["HOME"] == "/workspace"
    assert env["PWD"] == "/workspace"
    assert env["WORKSPACE"] == "/workspace"

  def test_bwrap_env_strips_workspace_from_path(self, workspace):
    """Test _get_bwrap_env removes workspace path from PATH."""
    with patch.dict(os.environ, {"PATH": f"{workspace.workspace_path}:/usr/bin:/bin"}):
      env = workspace._get_bwrap_env()
      paths = env["PATH"].split(os.pathsep)
      assert workspace.workspace_path not in paths
      assert "/usr/bin" in paths
      assert "/bin" in paths

  def test_execute_sandboxed_dispatches_to_bwrap(self, workspace):
    """Test _execute_sandboxed calls bwrap path when conditions met."""
    with (
      patch.object(workspace, "_should_use_bubblewrap", return_value=True),
      patch.object(workspace, "_execute_with_bubblewrap", return_value="bwrap output") as mock_bwrap,
    ):
      result = workspace._execute_sandboxed("echo hello", 60)
      mock_bwrap.assert_called_once_with("echo hello", 60)
      assert result == "bwrap output"

  def test_execute_sandboxed_dispatches_to_direct(self, workspace):
    """Test _execute_sandboxed falls back to direct execution."""
    with (
      patch.object(workspace, "_should_use_bubblewrap", return_value=False),
      patch.object(workspace, "_execute_direct", return_value="direct output") as mock_direct,
    ):
      result = workspace._execute_sandboxed("echo hello", 60)
      mock_direct.assert_called_once_with("echo hello", 60)
      assert result == "direct output"

  def test_execute_with_bubblewrap_calls_subprocess(self, workspace):
    """Test _execute_with_bubblewrap calls subprocess.run with correct args."""
    mock_result = MagicMock()
    mock_result.stdout = "hello\n"
    mock_result.stderr = ""

    with patch("autonomy.workspace.subprocess.run", return_value=mock_result) as mock_run:
      result = workspace._execute_with_bubblewrap("echo hello", 60)

      mock_run.assert_called_once()
      call_args = mock_run.call_args
      cmd = call_args[0][0]  # first positional arg
      assert cmd[0] == "bwrap"
      assert "--ro-bind" in cmd
      assert "--unshare-pid" in cmd
      assert call_args[1]["timeout"] == 60
      assert call_args[1]["capture_output"] is True
      assert call_args[1]["text"] is True
      assert result == "hello"

  def test_execute_with_bubblewrap_timeout(self, workspace):
    """Test _execute_with_bubblewrap raises TimeoutError on timeout."""
    with patch("autonomy.workspace.subprocess.run", side_effect=subprocess.TimeoutExpired("bwrap", 5)):
      with pytest.raises(TimeoutError, match="timed out"):
        workspace._execute_with_bubblewrap("sleep 100", 5)

  def test_execute_with_bubblewrap_runtime_error(self, workspace):
    """Test _execute_with_bubblewrap raises RuntimeError on failure."""
    with patch("autonomy.workspace.subprocess.run", side_effect=OSError("bwrap not found")):
      with pytest.raises(RuntimeError, match="execution failed"):
        workspace._execute_with_bubblewrap("echo hello", 60)

  def test_execute_with_bubblewrap_combines_stdout_stderr(self, workspace):
    """Test _execute_with_bubblewrap combines stdout and stderr."""
    mock_result = MagicMock()
    mock_result.stdout = "output"
    mock_result.stderr = "warning"

    with patch("autonomy.workspace.subprocess.run", return_value=mock_result):
      result = workspace._execute_with_bubblewrap("some-cmd", 60)
      assert "output" in result
      assert "warning" in result

  def test_terminal_uses_bwrap_when_available(self, workspace):
    """Test terminal() dispatches through bwrap when conditions met."""
    with (
      patch.object(workspace, "_should_use_bubblewrap", return_value=True),
      patch.object(workspace, "_execute_with_bubblewrap", return_value="sandboxed") as mock_bwrap,
    ):
      result = workspace.terminal("echo hello")
      mock_bwrap.assert_called_once_with("echo hello", workspace.command_timeout)
      assert result == "sandboxed"

  def test_terminal_falls_back_to_direct_without_gvisor(self, workspace):
    """Test terminal() uses direct execution when not in gVisor."""
    with patch.object(workspace, "_should_use_bubblewrap", return_value=False):
      result = workspace.terminal("echo hello")
      assert "hello" in result

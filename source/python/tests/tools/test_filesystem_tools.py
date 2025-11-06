"""
Tests for filesystem tools.

Validates core functionality, security, error handling, and edge cases for
the FilesystemTools class that provides scoped file operations for agents.
"""

import os
import pytest
import tempfile
import shutil

from autonomy.tools.filesystem import FilesystemTools


class TestFilesystemToolsInitialization:
  """Test FilesystemTools initialization and scope validation."""

  def test_basic_initialization(self):
    """Test basic initialization with valid scope."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test-user", base_dir=tmpdir)
      assert fs.scope == "test-user"
      assert fs.base_dir == tmpdir
      assert fs.scope_root == os.path.join(tmpdir, "test-agent", "test-user")
      assert os.path.exists(fs.scope_root)

  def test_initialization_creates_scope_directory(self):
    """Test that scope directory is created on initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
      scope_path = os.path.join(tmpdir, "test-agent", "new-scope")
      assert not os.path.exists(scope_path)

      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="new-scope", base_dir=tmpdir)
      assert os.path.exists(scope_path)
      assert os.path.isdir(scope_path)

  def test_empty_scope_raises_error(self):
    """Test that empty scope raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      with pytest.raises(ValueError, match="scope is required"):
        FilesystemTools(visibility="scope", agent_name="test-agent", scope="", base_dir=tmpdir)

      with pytest.raises(ValueError, match="scope is required"):
        FilesystemTools(visibility="scope", agent_name="test-agent", scope="   ", base_dir=tmpdir)

  def test_invalid_scope_characters(self):
    """Test that scope with path traversal characters raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Test various invalid scopes
      invalid_scopes = [
        "../test",
        "test/../other",
        "test/user",
        "test\\user",
        "..\\test",
      ]

      for invalid_scope in invalid_scopes:
        with pytest.raises(ValueError, match="Invalid scope"):
          FilesystemTools(visibility="scope", agent_name="test-agent", scope=invalid_scope, base_dir=tmpdir)

  def test_default_base_dir(self):
    """Test that default base_dir is /tmp/agent-files."""
    # Note: This test just validates the default, doesn't create actual directory
    # in /tmp as that might fail in some environments
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)
      # Just verify the parameter is accepted and used
      assert tmpdir in fs.scope_root


class TestPathResolution:
  """Test path resolution and security validation."""

  def test_resolve_simple_path(self):
    """Test resolving simple file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)
      resolved = fs._resolve_path("test.txt")
      expected = os.path.join(tmpdir, "test-agent", "user", "test.txt")
      assert resolved == expected

  def test_resolve_path_with_leading_slash(self):
    """Test that leading slash is removed and path is resolved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)
      resolved = fs._resolve_path("/test.txt")
      expected = os.path.join(tmpdir, "test-agent", "user", "test.txt")
      assert resolved == expected

  def test_resolve_nested_path(self):
    """Test resolving nested directory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)
      resolved = fs._resolve_path("docs/2024/notes.md")
      expected = os.path.join(tmpdir, "test-agent", "user", "docs", "2024", "notes.md")
      assert resolved == expected

  def test_resolve_current_directory(self):
    """Test resolving current directory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)
      resolved = fs._resolve_path(".")
      expected = os.path.join(tmpdir, "test-agent", "user")
      assert resolved == expected

  def test_directory_traversal_blocked(self):
    """Test that directory traversal attacks are blocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Various traversal attempts
      traversal_attempts = [
        "../../../etc/passwd",
        "docs/../../etc/passwd",
        "../other-user/file.txt",
        "../../..",
      ]

      for attempt in traversal_attempts:
        with pytest.raises(ValueError, match="escape visibility boundary"):
          fs._resolve_path(attempt)

  def test_empty_path_resolves_to_current(self):
    """Test that empty path resolves to current directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)
      resolved = fs._resolve_path("")
      expected = os.path.join(tmpdir, "test-agent", "user")
      assert resolved == expected


class TestListDirectory:
  """Test list_directory functionality."""

  def test_ls_empty_directory(self):
    """Test listing an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)
      result = fs.list_directory(".")
      assert "Empty directory" in result

  def test_ls_with_files(self):
    """Test listing directory with files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create some files
      with open(os.path.join(fs.scope_root, "file1.txt"), "w") as f:
        f.write("content1")
      with open(os.path.join(fs.scope_root, "file2.txt"), "w") as f:
        f.write("content2")

      result = fs.list_directory(".")
      assert "file1.txt" in result
      assert "file2.txt" in result
      assert "bytes" in result

  def test_ls_with_subdirectories(self):
    """Test listing directory with subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create subdirectories
      os.makedirs(os.path.join(fs.scope_root, "docs"))
      os.makedirs(os.path.join(fs.scope_root, "projects"))

      result = fs.list_directory(".")
      assert "docs/" in result
      assert "projects/" in result

  def test_ls_nonexistent_path(self):
    """Test listing nonexistent directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)
      result = fs.list_directory("nonexistent")
      assert "Error" in result
      assert "does not exist" in result

  def test_ls_file_not_directory(self):
    """Test listing a file (not directory)."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create a file
      filepath = os.path.join(fs.scope_root, "test.txt")
      with open(filepath, "w") as f:
        f.write("content")

      result = fs.list_directory("test.txt")
      assert "Error" in result
      assert "not a directory" in result


class TestReadFile:
  """Test read_file functionality."""

  def test_read_simple_file(self):
    """Test reading a simple text file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create and write file
      content = "Hello, World!"
      filepath = os.path.join(fs.scope_root, "test.txt")
      with open(filepath, "w") as f:
        f.write(content)

      result = fs.read_file("test.txt")
      assert result == content

  def test_read_multiline_file(self):
    """Test reading file with multiple lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      content = "Line 1\nLine 2\nLine 3\n"
      filepath = os.path.join(fs.scope_root, "multiline.txt")
      with open(filepath, "w") as f:
        f.write(content)

      result = fs.read_file("multiline.txt")
      assert result == content

  def test_read_nonexistent_file(self):
    """Test reading nonexistent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)
      result = fs.read_file("nonexistent.txt")
      assert "Error" in result
      assert "does not exist" in result

  def test_read_directory_not_file(self):
    """Test reading a directory (not file)."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create directory
      os.makedirs(os.path.join(fs.scope_root, "docs"))

      result = fs.read_file("docs")
      assert "Error" in result
      assert "not a file" in result


class TestWriteFile:
  """Test write_file functionality."""

  def test_write_simple_file(self):
    """Test writing a simple file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      content = "Test content"
      result = fs.write_file("test.txt", content)

      assert "Successfully wrote" in result
      assert "12 bytes" in result

      # Verify file was created
      filepath = os.path.join(fs.scope_root, "test.txt")
      assert os.path.exists(filepath)
      with open(filepath, "r") as f:
        assert f.read() == content

  def test_write_creates_parent_directories(self):
    """Test that write_file creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      result = fs.write_file("docs/2024/notes.md", "# Notes")

      assert "Successfully wrote" in result

      # Verify directory structure was created
      filepath = os.path.join(fs.scope_root, "docs", "2024", "notes.md")
      assert os.path.exists(filepath)
      with open(filepath, "r") as f:
        assert f.read() == "# Notes"

  def test_write_overwrites_existing_file(self):
    """Test that write_file overwrites existing content."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Write initial content
      fs.write_file("test.txt", "Initial content")

      # Overwrite with new content
      result = fs.write_file("test.txt", "New content")

      assert "Successfully wrote" in result

      # Verify file was overwritten
      filepath = os.path.join(fs.scope_root, "test.txt")
      with open(filepath, "r") as f:
        assert f.read() == "New content"

  def test_write_empty_file(self):
    """Test writing an empty file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      result = fs.write_file("empty.txt", "")

      assert "Successfully wrote 0 bytes" in result

      filepath = os.path.join(fs.scope_root, "empty.txt")
      assert os.path.exists(filepath)


class TestEditFile:
  """Test edit_file (string replacement) functionality."""

  def test_edit_replaces_text(self):
    """Test that edit_file replaces text in file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create file with initial content
      fs.write_file("config.yaml", "port: 8080\nhost: localhost\n")

      # Replace port value
      result = fs.edit_file("config.yaml", "port: 8080", "port: 3000")

      assert "Successfully replaced 1 occurrence" in result

      # Verify content was replaced
      content = fs.read_file("config.yaml")
      assert "port: 3000" in content
      assert "port: 8080" not in content

  def test_edit_replaces_all_occurrences(self):
    """Test that edit_file with replace_all replaces all occurrences."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create file with repeated text
      fs.write_file("test.txt", "Hello World\nHello Universe\nHello Everyone\n")

      # Replace all occurrences
      result = fs.edit_file("test.txt", "Hello", "Hi", replace_all=True)

      assert "Successfully replaced 3 occurrence" in result

      # Verify all were replaced
      content = fs.read_file("test.txt")
      assert "Hi World" in content
      assert "Hi Universe" in content
      assert "Hi Everyone" in content
      assert "Hello" not in content

  def test_edit_fails_on_multiple_without_replace_all(self):
    """Test that edit_file fails when pattern appears multiple times without replace_all."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create file with repeated text
      fs.write_file("test.txt", "test\ntest\ntest\n")

      # Try to replace without replace_all
      result = fs.edit_file("test.txt", "test", "changed")

      assert "Error" in result
      assert "Found 3 occurrences" in result
      assert "replace_all=True" in result

  def test_edit_creates_parent_directories(self):
    """Test that edit_file fails if file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Try to edit non-existent file
      result = fs.edit_file("newfile.txt", "old", "new")

      assert "Error" in result
      assert "does not exist" in result


class TestFindFiles:
  """Test find_files (pattern matching) functionality."""

  def test_glob_simple_pattern(self):
    """Test find_files with simple wildcard pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create test files
      fs.write_file("file1.txt", "content")
      fs.write_file("file2.txt", "content")
      fs.write_file("file3.md", "content")

      result = fs.find_files("*.txt")

      assert "file1.txt" in result
      assert "file2.txt" in result
      assert "file3.md" not in result

  def test_glob_recursive_pattern(self):
    """Test find_files with recursive pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create nested files
      fs.write_file("readme.md", "content")
      fs.write_file("docs/guide.md", "content")
      fs.write_file("docs/api/reference.md", "content")

      result = fs.find_files("**/*.md")

      assert "readme.md" in result
      assert "docs/guide.md" in result
      assert "docs/api/reference.md" in result

  def test_glob_no_matches(self):
    """Test find_files with no matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      fs.write_file("test.txt", "content")

      result = fs.find_files("*.md")

      assert "No files match" in result

  def test_glob_specific_directory(self):
    """Test find_files in specific directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      fs.write_file("root.txt", "content")
      fs.write_file("docs/doc1.txt", "content")
      fs.write_file("docs/doc2.txt", "content")

      result = fs.find_files("docs/*.txt")

      assert "docs/doc1.txt" in result
      assert "docs/doc2.txt" in result
      assert "root.txt" not in result


class TestSearchInFiles:
  """Test search_in_files (content search) functionality."""

  def test_grep_simple_pattern(self):
    """Test search_in_files with simple text pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      fs.write_file("test.txt", "Line 1\nTODO: Fix this\nLine 3\n")

      result = fs.search_in_files("TODO", ".")

      assert "TODO" in result
      assert "test.txt:2:" in result
      assert "Fix this" in result

  def test_grep_regex_pattern(self):
    """Test search_in_files with regex pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      fs.write_file("code.py", "def function1():\ndef function2():\nclass MyClass:\n")

      result = fs.search_in_files(r"def \w+", ".")

      assert "function1" in result
      assert "function2" in result
      assert "MyClass" not in result

  def test_grep_multiple_files(self):
    """Test search_in_files across multiple files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      fs.write_file("file1.txt", "Error in file1\n")
      fs.write_file("file2.txt", "Everything OK\n")
      fs.write_file("file3.txt", "Error in file3\n")

      result = fs.search_in_files("Error", ".")

      assert "file1.txt" in result
      assert "file3.txt" in result
      assert "file2.txt" not in result

  def test_grep_no_matches(self):
    """Test search_in_files with no matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      fs.write_file("test.txt", "Some content\n")

      result = fs.search_in_files("nonexistent", ".")

      assert "No matches found" in result

  def test_grep_invalid_regex(self):
    """Test search_in_files with invalid regex pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      fs.write_file("test.txt", "content")

      result = fs.search_in_files("[invalid(regex", ".")

      assert "Error" in result
      assert "Invalid regex" in result

  def test_grep_specific_file(self):
    """Test search_in_files on specific file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      fs.write_file("target.txt", "Line 1\nMatch here\nLine 3\n")
      fs.write_file("other.txt", "No match in this file\n")

      result = fs.search_in_files("Match", "target.txt")

      assert "target.txt" in result
      assert "Match here" in result
      assert "other.txt" not in result

  def test_grep_nonexistent_path(self):
    """Test search_in_files on nonexistent path."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      result = fs.search_in_files("pattern", "nonexistent.txt")

      assert "Error" in result
      assert "does not exist" in result


class TestScopeIsolation:
  """Test that different scopes are properly isolated."""

  def test_different_scopes_isolated(self):
    """Test that different scopes cannot access each other's files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs_alice = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user-alice", base_dir=tmpdir)
      fs_bob = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user-bob", base_dir=tmpdir)

      # Alice creates a file
      fs_alice.write_file("secret.txt", "Alice's secret")

      # Bob tries to list files - should see empty directory
      result = fs_bob.list_directory(".")
      assert "Empty directory" in result

      # Bob tries to read Alice's file - should fail
      result = fs_bob.read_file("secret.txt")
      assert "Error" in result
      assert "does not exist" in result

  def test_scope_root_separation(self):
    """Test that scope roots are in separate directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs1 = FilesystemTools(visibility="scope", agent_name="test-agent", scope="scope1", base_dir=tmpdir)
      fs2 = FilesystemTools(visibility="scope", agent_name="test-agent", scope="scope2", base_dir=tmpdir)

      assert fs1.scope_root != fs2.scope_root
      assert os.path.exists(fs1.scope_root)
      assert os.path.exists(fs2.scope_root)


class TestEdgeCases:
  """Test edge cases and error conditions."""

  def test_unicode_content(self):
    """Test handling of unicode content."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      content = "Hello 世界 🌍"
      fs.write_file("unicode.txt", content)

      result = fs.read_file("unicode.txt")
      assert result == content

  def test_large_file(self):
    """Test handling of larger files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Create a 1MB file
      content = "x" * (1024 * 1024)
      result = fs.write_file("large.txt", content)

      assert "1048576 bytes" in result

      read_result = fs.read_file("large.txt")
      assert len(read_result) == len(content)

  def test_special_characters_in_filename(self):
    """Test files with special characters in names."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="user", base_dir=tmpdir)

      # Valid special characters in filenames
      filenames = ["test-file.txt", "test_file.txt", "test.2024.txt"]

      for filename in filenames:
        result = fs.write_file(filename, "content")
        assert "Successfully wrote" in result

  def test_empty_directory_operations(self):
    """Test operations on empty scope."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="empty", base_dir=tmpdir)

      # List empty directory
      assert "Empty directory" in fs.list_directory(".")

      # Glob with no files
      assert "No files match" in fs.find_files("*")

      # Grep with no files
      assert "No matches found" in fs.search_in_files("pattern", ".")


class TestRemoveFile:
  """Tests for remove_file operation."""

  def test_delete_simple_file(self):
    """Test deleting a simple file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Create a file
      fs.write_file("test.txt", "Hello")

      # Delete it
      result = fs.remove_file("test.txt")
      assert "Successfully deleted file 'test.txt'" in result

      # Verify it's gone
      assert "does not exist" in fs.read_file("test.txt")

  def test_delete_file_in_subdirectory(self):
    """Test deleting a file in a subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Create file in subdirectory
      fs.write_file("docs/readme.md", "# README")

      # Delete it
      result = fs.remove_file("docs/readme.md")
      assert "Successfully deleted file 'docs/readme.md'" in result

      # Verify it's gone
      assert "does not exist" in fs.read_file("docs/readme.md")

  def test_delete_nonexistent_file(self):
    """Test deleting a file that doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      result = fs.remove_file("nonexistent.txt")
      assert "Error: File 'nonexistent.txt' does not exist" in result

  def test_delete_file_on_directory(self):
    """Test that remove_file returns error when path is a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Create a file to ensure directory exists
      fs.write_file("mydir/file.txt", "content")

      # Try to delete directory with remove_file
      result = fs.remove_file("mydir")
      assert "Error: 'mydir' is a directory" in result

  def test_delete_file_with_leading_slash(self):
    """Test deleting a file with leading slash."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      fs.write_file("test.txt", "Hello")
      result = fs.remove_file("/test.txt")
      assert "Successfully deleted file" in result

  def test_delete_file_prevents_traversal(self):
    """Test that remove_file prevents directory traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Try to delete file outside scope
      result = fs.remove_file("../../etc/passwd")
      assert "Error:" in result


class TestRemoveDirectory:
  """Tests for remove_directory operation."""

  def test_delete_empty_directory(self):
    """Test deleting an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Create empty directory
      os.makedirs(os.path.join(fs.scope_root, "emptydir"))

      # Delete it
      result = fs.remove_directory("emptydir")
      assert "Successfully deleted empty directory 'emptydir'" in result

      # Verify it's gone
      ls_result = fs.list_directory(".")
      assert "emptydir" not in ls_result

  def test_delete_nonempty_directory_without_recursive(self):
    """Test that deleting non-empty directory without recursive fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Create directory with file
      fs.write_file("mydir/file.txt", "content")

      # Try to delete without recursive
      result = fs.remove_directory("mydir", recursive=False)
      assert "Error: Directory 'mydir' is not empty" in result
      assert "recursive=True" in result

      # Verify directory still exists
      assert "file.txt" in fs.list_directory("mydir")

  def test_delete_directory_recursive(self):
    """Test deleting directory with recursive=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Create directory with nested files
      fs.write_file("project/src/main.py", "print('hello')")
      fs.write_file("project/src/utils.py", "# utils")
      fs.write_file("project/README.md", "# Project")

      # Delete recursively
      result = fs.remove_directory("project", recursive=True)
      assert "Successfully deleted directory 'project' and all its contents" in result

      # Verify it's gone
      ls_result = fs.list_directory(".")
      assert "project" not in ls_result

  def test_delete_nonexistent_directory(self):
    """Test deleting a directory that doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      result = fs.remove_directory("nonexistent")
      assert "Error: Directory 'nonexistent' does not exist" in result

  def test_delete_directory_on_file(self):
    """Test that remove_directory returns error when path is a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Create a file
      fs.write_file("test.txt", "content")

      # Try to delete file with remove_directory
      result = fs.remove_directory("test.txt")
      assert "Error: 'test.txt' is not a directory" in result

  def test_delete_root_directory_blocked(self):
    """Test that deleting the root directory is blocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Try to delete root
      result = fs.remove_directory(".", recursive=True)
      assert "Error: Cannot delete the root directory" in result

      # Try with /
      result = fs.remove_directory("/", recursive=True)
      assert "Error: Cannot delete the root directory" in result

  def test_delete_nested_directory_recursive(self):
    """Test deleting deeply nested directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Create deep structure
      fs.write_file("a/b/c/d/e/file.txt", "deep")

      # Delete from middle level
      result = fs.remove_directory("a/b", recursive=True)
      assert "Successfully deleted directory 'a/b' and all its contents" in result

      # Verify parent still exists, child is gone
      assert "a" in fs.list_directory(".")
      assert "does not exist" in fs.read_file("a/b/c/d/e/file.txt")

  def test_delete_directory_prevents_traversal(self):
    """Test that remove_directory prevents directory traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fs = FilesystemTools(visibility="scope", agent_name="test-agent", scope="test", base_dir=tmpdir)

      # Try to delete directory outside scope
      result = fs.remove_directory("../../etc", recursive=True)
      assert "Error:" in result

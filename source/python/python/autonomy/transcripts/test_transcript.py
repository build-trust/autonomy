"""Unit tests for per-conversation transcript logging functionality."""

import json
import os
import tempfile
import shutil
from pathlib import Path

import pytest

from autonomy.transcripts.transcript import (
  get_transcript_config,
  log_raw_request,
  log_raw_response,
  _get_conversation_file_path,
)


@pytest.fixture
def temp_transcript_dir():
  """Create a temporary directory for transcripts."""
  temp_dir = tempfile.mkdtemp(prefix="transcript_test_")
  yield temp_dir
  # Cleanup
  if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)


@pytest.fixture
def reset_config():
  """Reset the global transcript config before and after each test."""
  import autonomy.transcripts.transcript as transcript_module

  # Save original env vars
  original_env = {
    "AUTONOMY_TRANSCRIPTS_DIR": os.environ.get("AUTONOMY_TRANSCRIPTS_DIR"),
  }

  # Clear config
  transcript_module._config = None

  yield

  # Restore config
  transcript_module._config = None

  # Restore env vars
  for key, value in original_env.items():
    if value is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = value


def test_conversation_file_path_generation(reset_config, temp_transcript_dir):
  """Test that conversation file paths are generated correctly."""
  os.environ["AUTONOMY_TRANSCRIPTS_DIR"] = temp_transcript_dir
  config = get_transcript_config()

  # Test normal case
  path = _get_conversation_file_path(
    config,
    agent_name="support_agent",
    scope="user_123",
    conversation="conv_001",
  )

  expected = os.path.join(temp_transcript_dir, "support_agent_user_123_conv_001.jsonl")
  assert path == expected

  # Test with special characters (should be sanitized)
  path = _get_conversation_file_path(
    config,
    agent_name="agent/with/slashes",
    scope="scope\\with\\backslashes",
    conversation="conv_001",
  )

  expected = os.path.join(temp_transcript_dir, "agent_with_slashes_scope_with_backslashes_conv_001.jsonl")
  assert path == expected

  # Test missing scope - should use "none"
  path = _get_conversation_file_path(
    config,
    agent_name="agent",
    scope=None,
    conversation="conv",
  )
  expected = os.path.join(temp_transcript_dir, "agent_none_conv.jsonl")
  assert path == expected

  # Test missing conversation - should use "none"
  path = _get_conversation_file_path(
    config,
    agent_name="agent",
    scope="scope",
    conversation=None,
  )
  expected = os.path.join(temp_transcript_dir, "agent_scope_none.jsonl")
  assert path == expected

  # Test both missing - should use "none" for both
  path = _get_conversation_file_path(
    config,
    agent_name="agent",
    scope=None,
    conversation=None,
  )
  expected = os.path.join(temp_transcript_dir, "agent_none_none.jsonl")
  assert path == expected

  # Test missing agent - should return None
  path = _get_conversation_file_path(
    config,
    agent_name=None,
    scope="scope",
    conversation="conv",
  )
  assert path is None


def test_per_conversation_files_created(reset_config, temp_transcript_dir):
  """Test that separate files are created for each conversation."""
  os.environ["AUTONOMY_TRANSCRIPTS_DIR"] = temp_transcript_dir

  import autonomy.transcripts.transcript as transcript_module
  transcript_module._config = None

  # Log messages for different conversations
  conversations = [
    ("agent1", "scope1", "conv1", "model1"),
    ("agent1", "scope1", "conv2", "model1"),
    ("agent2", "scope2", "conv1", "model2"),
  ]

  for agent, scope, conv, model in conversations:
    log_raw_request(
      payload={"messages": [{"role": "user", "content": f"Hello from {conv}"}]},
      model_name=model,
      agent_name=agent,
      scope=scope,
      conversation=conv,
    )

  # Verify files were created
  files = sorted(os.listdir(temp_transcript_dir))
  expected_files = [
    "agent1_scope1_conv1.jsonl",
    "agent1_scope1_conv2.jsonl",
    "agent2_scope2_conv1.jsonl",
  ]

  assert files == expected_files


def test_message_deduplication(reset_config, temp_transcript_dir):
  """Test that duplicate messages are not written to files."""
  os.environ["AUTONOMY_TRANSCRIPTS_DIR"] = temp_transcript_dir

  import autonomy.transcripts.transcript as transcript_module
  transcript_module._config = None

  agent = "test_agent"
  scope = "test_scope"
  conversation = "test_conv"
  model = "test_model"

  # First request with one message
  log_raw_request(
    payload={"messages": [{"role": "user", "content": "Message 1"}]},
    model_name=model,
    agent_name=agent,
    scope=scope,
    conversation=conversation,
  )

  # Second request with the same message plus a new one
  log_raw_request(
    payload={
      "messages": [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
      ]
    },
    model_name=model,
    agent_name=agent,
    scope=scope,
    conversation=conversation,
  )

  # Third request with all previous messages plus a new one
  log_raw_request(
    payload={
      "messages": [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
      ]
    },
    model_name=model,
    agent_name=agent,
    scope=scope,
    conversation=conversation,
  )

  # Read the file
  file_path = os.path.join(temp_transcript_dir, f"{agent}_{scope}_{conversation}.jsonl")
  with open(file_path, "r") as f:
    lines = f.readlines()

  # Should have 3 unique messages
  assert len(lines) == 3

  # Verify content
  messages = [json.loads(line) for line in lines]
  assert messages[0] == {"role": "user", "content": "Message 1"}
  assert messages[1] == {"role": "assistant", "content": "Response 1"}
  assert messages[2] == {"role": "user", "content": "Message 2"}


def test_response_logging_with_tool_calls(reset_config, temp_transcript_dir):
  """Test that responses with tool calls are logged correctly."""
  os.environ["AUTONOMY_TRANSCRIPTS_DIR"] = temp_transcript_dir

  import autonomy.transcripts.transcript as transcript_module
  transcript_module._config = None

  agent = "test_agent"
  scope = "test_scope"
  conversation = "test_conv"
  model = "test_model"

  # Log response with tool calls
  log_raw_response(
    response={
      "choices": [{
        "message": {
          "role": "assistant",
          "content": "I'll help you with that.",
          "tool_calls": [{
            "id": "call_123",
            "function": {
              "name": "search",
              "arguments": '{"query": "test"}'
            }
          }]
        }
      }]
    },
    model_name=model,
    agent_name=agent,
    scope=scope,
    conversation=conversation,
  )

  # Read the file
  file_path = os.path.join(temp_transcript_dir, f"{agent}_{scope}_{conversation}.jsonl")
  with open(file_path, "r") as f:
    line = f.readline()

  message = json.loads(line)
  assert message["role"] == "assistant"
  assert message["content"] == "I'll help you with that."
  assert len(message["tool_calls"]) == 1
  assert message["tool_calls"][0]["id"] == "call_123"
  assert message["tool_calls"][0]["function"]["name"] == "search"


def test_system_message_extraction(reset_config, temp_transcript_dir):
  """Test that system messages are extracted correctly."""
  os.environ["AUTONOMY_TRANSCRIPTS_DIR"] = temp_transcript_dir

  import autonomy.transcripts.transcript as transcript_module
  transcript_module._config = None

  agent = "test_agent"
  scope = "test_scope"
  conversation = "test_conv"
  model = "test_model"

  # Log request with system message (Anthropic format)
  log_raw_request(
    payload={
      "system": "You are a helpful assistant.",
      "messages": [{"role": "user", "content": "Hello"}]
    },
    model_name=model,
    agent_name=agent,
    scope=scope,
    conversation=conversation,
  )

  # Read the file
  file_path = os.path.join(temp_transcript_dir, f"{agent}_{scope}_{conversation}.jsonl")
  with open(file_path, "r") as f:
    lines = f.readlines()

  # Should have system message and user message
  assert len(lines) == 2

  messages = [json.loads(line) for line in lines]
  assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
  assert messages[1] == {"role": "user", "content": "Hello"}




def test_directory_creation(reset_config):
  """Test that the transcript directory is created if it doesn't exist."""
  temp_dir = tempfile.mkdtemp(prefix="transcript_test_parent_")
  transcript_dir = os.path.join(temp_dir, "nested", "transcripts")

  try:
    os.environ["AUTONOMY_TRANSCRIPTS_DIR"] = transcript_dir

    import autonomy.transcripts.transcript as transcript_module
    transcript_module._config = None

    # Directory should not exist yet
    assert not os.path.exists(transcript_dir)

    # Log a message
    log_raw_request(
      payload={"messages": [{"role": "user", "content": "Test"}]},
      model_name="model1",
      agent_name="agent1",
      scope="scope1",
      conversation="conv1",
    )

    # Directory should now exist
    assert os.path.exists(transcript_dir)

    # File should exist in the directory
    conv_file = os.path.join(transcript_dir, "agent1_scope1_conv1.jsonl")
    assert os.path.exists(conv_file)

  finally:
    if os.path.exists(temp_dir):
      shutil.rmtree(temp_dir)


def test_none_placeholder_for_missing_identifiers(reset_config, temp_transcript_dir):
  """Test that 'none' is used as placeholder when scope or conversation is missing."""
  os.environ["AUTONOMY_TRANSCRIPTS_DIR"] = temp_transcript_dir

  import autonomy.transcripts.transcript as transcript_module
  transcript_module._config = None

  # Log without conversation identifier - should use "none"
  log_raw_request(
    payload={"messages": [{"role": "user", "content": "Test 1"}]},
    model_name="model1",
    agent_name="agent1",
    scope="scope1",
    conversation=None,
  )

  # Log without scope identifier - should use "none"
  log_raw_request(
    payload={"messages": [{"role": "user", "content": "Test 2"}]},
    model_name="model2",
    agent_name="agent2",
    scope=None,
    conversation="conv1",
  )

  # Log without both - should use "none" for both
  log_raw_request(
    payload={"messages": [{"role": "user", "content": "Test 3"}]},
    model_name="model3",
    agent_name="agent3",
    scope=None,
    conversation=None,
  )

  # Should have created 3 files with "none" placeholders
  files = sorted(os.listdir(temp_transcript_dir))
  expected_files = [
    "agent1_scope1_none.jsonl",
    "agent2_none_conv1.jsonl",
    "agent3_none_none.jsonl",
  ]
  assert files == expected_files

  # Verify content
  with open(os.path.join(temp_transcript_dir, "agent1_scope1_none.jsonl")) as f:
    msg = json.loads(f.readline())
    assert msg["content"] == "Test 1"

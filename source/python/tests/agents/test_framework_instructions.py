"""
Tests for framework-level system instructions.

These tests verify that built-in tool descriptions and framework capabilities
are properly added to agent system instructions via the FrameworkInstructionsSection.
"""

import pytest
from autonomy.agents.context import FrameworkInstructionsSection
from autonomy.agents.agent import system_message


@pytest.mark.asyncio
async def test_framework_instructions_without_ask_user():
  """Test framework instructions without ask_user_for_input tool."""
  section = FrameworkInstructionsSection(enable_ask_for_user_input=False)
  messages = await section.get_messages("scope", "conv", {})

  assert len(messages) == 1
  instructions = messages[0]["content"]["text"]

  # Should include framework header
  assert "Autonomy Framework Capabilities" in instructions or "capabilities" in instructions.lower()

  # Should include time tools
  assert "get_current_time_utc" in instructions
  assert "get_current_time" in instructions
  assert "UTC" in instructions
  assert "local" in instructions.lower()

  # Should NOT include ask_user_for_input
  assert "ask_user_for_input" not in instructions
  assert "User Input" not in instructions


async def test_framework_instructions_with_ask_user():
  """Test framework instructions with ask_user_for_input tool enabled."""
  section = FrameworkInstructionsSection(enable_ask_for_user_input=True)
  messages = await section.get_messages("scope", "conv", {})

  assert len(messages) == 1
  instructions = messages[0]["content"]["text"]

  # Should include time tools
  assert "get_current_time_utc" in instructions
  assert "get_current_time" in instructions

  # Should include ask_user_for_input section
  assert "ask_user_for_input" in instructions
  assert "User Input" in instructions or "user input" in instructions.lower()
  assert "pause" in instructions.lower()
  assert "clarification" in instructions.lower() or "additional information" in instructions.lower()


async def test_framework_instructions_format():
  """Test that framework instructions are properly formatted."""
  section = FrameworkInstructionsSection(enable_ask_for_user_input=True)
  messages = await section.get_messages("scope", "conv", {})

  assert len(messages) == 1
  assert messages[0]["role"] == "system"
  assert messages[0]["phase"] == "system"

  instructions = messages[0]["content"]["text"]

  # Should be non-empty string
  assert isinstance(instructions, str)
  assert len(instructions) > 0

  # Should have readable structure
  assert "\n" in instructions  # Multi-line


async def test_framework_instructions_default_parameter():
  """Test FrameworkInstructionsSection with default parameter."""
  section = FrameworkInstructionsSection()
  messages = await section.get_messages("scope", "conv", {})

  assert len(messages) == 1
  instructions = messages[0]["content"]["text"]

  # Default should be without ask_user_for_input
  assert "ask_user_for_input" not in instructions


def test_system_message_format():
  """Test that system_message creates proper message dict."""
  msg = system_message("Test instruction")

  assert isinstance(msg, dict)
  assert msg["role"] == "system"
  assert msg["content"]["text"] == "Test instruction"
  assert msg["content"]["type"] == "text"
  assert msg["phase"] == "system"


async def test_framework_instructions_combined_with_user():
  """Test that framework instructions work alongside user instructions in template."""
  from autonomy.agents.context import ContextTemplate, SystemInstructionsSection

  user_msg = {"role": "system", "content": {"text": "You are a helpful assistant.", "type": "text"}}
  framework_section = FrameworkInstructionsSection(enable_ask_for_user_input=True)

  template = ContextTemplate([SystemInstructionsSection([user_msg]), framework_section])

  messages = await template.build_context("scope", "conv")

  # Should have both messages
  assert len(messages) == 2
  assert "helpful assistant" in messages[0]["content"]["text"]
  assert "get_current_time" in messages[1]["content"]["text"]
  assert "ask_user_for_input" in messages[1]["content"]["text"]


async def test_framework_instructions_content_accuracy():
  """Test that framework instructions accurately describe tools."""
  section = FrameworkInstructionsSection(enable_ask_for_user_input=True)
  messages = await section.get_messages("scope", "conv", {})

  instructions = messages[0]["content"]["text"]

  # Time tools should mention their purpose
  assert "UTC" in instructions or "utc" in instructions
  assert "local" in instructions.lower()

  # ask_user_for_input should explain when to use it
  ask_section = instructions.lower()
  assert any(word in ask_section for word in ["clarification", "information", "decision"])


async def test_framework_instructions_no_duplication():
  """Test that framework instructions don't have obvious duplication."""
  section = FrameworkInstructionsSection(enable_ask_for_user_input=True)
  messages = await section.get_messages("scope", "conv", {})

  instructions = messages[0]["content"]["text"]

  # Split into lines and check for exact duplicates
  lines = [line.strip() for line in instructions.split("\n") if line.strip()]

  # Should have more than one unique line
  assert len(set(lines)) > 1

  # No line should appear more than twice (allowing for formatting)
  from collections import Counter

  line_counts = Counter(lines)
  for line, count in line_counts.items():
    assert count <= 3, f"Line appears {count} times: {line}"


async def test_framework_instructions_with_subagents():
  """Test framework instructions with subagent configurations."""
  subagent_configs = {
    "researcher": {
      "instructions": "You are a research assistant specialized in finding information.",
      "model": {"name": "claude-sonnet-4-v1"},
    },
    "writer": {
      "instructions": "You are a writing assistant specialized in creating content.",
      "model": {"name": "claude-sonnet-4-v1"},
    },
  }

  section = FrameworkInstructionsSection(enable_ask_for_user_input=False, subagent_configs=subagent_configs)
  messages = await section.get_messages("scope", "conv", {})

  instructions = messages[0]["content"]["text"]

  # Should include subagent section
  assert "Subagent" in instructions or "subagent" in instructions.lower()
  assert "researcher" in instructions
  assert "writer" in instructions

  # Should include subagent tools
  assert "start_subagent" in instructions
  assert "delegate_to_subagent" in instructions
  assert "list_subagents" in instructions or "stop_subagent" in instructions


async def test_framework_instructions_without_subagents():
  """Test framework instructions without subagent configurations."""
  section = FrameworkInstructionsSection(enable_ask_for_user_input=False, subagent_configs=None)
  messages = await section.get_messages("scope", "conv", {})

  instructions = messages[0]["content"]["text"]

  # Should NOT include subagent section
  assert "start_subagent" not in instructions
  assert "delegate_to_subagent" not in instructions


async def test_framework_instructions_all_features():
  """Test framework instructions with all features enabled."""
  subagent_configs = {
    "helper": {
      "instructions": "You are a helpful helper.",
      "model": {"name": "claude-sonnet-4-v1"},
    }
  }

  section = FrameworkInstructionsSection(enable_ask_for_user_input=True, subagent_configs=subagent_configs)
  messages = await section.get_messages("scope", "conv", {})

  instructions = messages[0]["content"]["text"]

  # Should include all sections
  assert "get_current_time" in instructions  # Time tools
  assert "ask_user_for_input" in instructions  # User input
  assert "helper" in instructions  # Subagents
  assert "delegate" in instructions.lower()  # Subagent delegation

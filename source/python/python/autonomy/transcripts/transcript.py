"""
Agent transcript logging utilities.

This module provides utilities for logging agent/model interactions (transcripts).
Transcripts show what's sent to the model (input) and what's returned (output),
with one file per conversation in JSONL format.

Environment Variables:
  AUTONOMY_TRANSCRIPTS_DIR=/path  - Directory for per-conversation transcript files (required)
    • Files are named: {agent}_{scope}_{conversation}.jsonl
    • Uses "none" as placeholder for missing scope or conversation
    • Each file contains JSONL format (one message per line)
    • Directory is automatically created if it doesn't exist (including parent directories)
    • Each message is a complete JSON object on its own line

JSONL Record Format:
  One message per line:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [...]}
    {"role": "tool", "name": "tool_name", "content": "..."}

Examples:
  # Enable per-conversation transcripts
  AUTONOMY_TRANSCRIPTS_DIR=/tmp/transcripts python script.py

  # View a specific conversation
  cat /tmp/transcripts/agent_scope_conv.jsonl | jq '.'

  # Filter by role
  cat /tmp/transcripts/*.jsonl | jq 'select(.role=="assistant")'

  # Get all tool calls
  cat /tmp/transcripts/*.jsonl | jq 'select(.tool_calls) | .tool_calls[]'

  # Count messages by role
  cat /tmp/transcripts/*.jsonl | jq -r '.role' | sort | uniq -c

  # Watch conversations in real-time
  tail -f /tmp/transcripts/*.jsonl | jq '.'
"""

import json
import os
from typing import Dict, List, Optional, Any


class TranscriptConfig:
  """Configuration for transcript logging."""

  def __init__(self):
    """Initialize transcript configuration from environment variables."""
    self.directory = os.environ.get("AUTONOMY_TRANSCRIPTS_DIR")

    # Track output messages to avoid duplicates (per conversation)
    # Key: f"{agent}_{scope}_{conversation}_{model}", Value: count of messages output
    self.message_counts = {}

  def should_log(self) -> bool:
    """Should we log anything at all?"""
    return self.directory is not None


# Global config instance
_config: Optional[TranscriptConfig] = None


def get_transcript_config() -> TranscriptConfig:
  """Get the global transcript configuration."""
  global _config
  if _config is None:
    _config = TranscriptConfig()
  return _config


def _get_conversation_file_path(
  config: TranscriptConfig,
  agent_name: Optional[str] = None,
  scope: Optional[str] = None,
  conversation: Optional[str] = None,
) -> Optional[str]:
  """
  Get the file path for a specific conversation.

  Args:
    config: Transcript configuration
    agent_name: Optional agent name
    scope: Optional scope identifier
    conversation: Optional conversation identifier

  Returns:
    File path string if directory is configured and agent_name is provided, else None
    Uses "none" as placeholder for missing scope or conversation.
  """
  if not config.directory:
    return None

  if not agent_name:
    return None

  # Sanitize identifiers for file system, use "none" for missing values
  safe_agent = agent_name.replace("/", "_").replace("\\", "_")
  safe_scope = (scope or "none").replace("/", "_").replace("\\", "_")
  safe_conversation = (conversation or "none").replace("/", "_").replace("\\", "_")

  # Create file name
  filename = f"{safe_agent}_{safe_scope}_{safe_conversation}.jsonl"
  return os.path.join(config.directory, filename)


def _output(
  text: str,
  config: TranscriptConfig,
  agent_name: Optional[str] = None,
  scope: Optional[str] = None,
  conversation: Optional[str] = None,
) -> None:
  """
  Output text to per-conversation file.

  Args:
    text: Text to output
    config: Transcript configuration
    agent_name: Agent name (required for file output)
    scope: Scope identifier (optional, uses "none" if not provided)
    conversation: Conversation identifier (optional, uses "none" if not provided)
  """
  # Get conversation file path
  conversation_file = _get_conversation_file_path(config, agent_name, scope, conversation)

  # Write to per-conversation file if we have all required identifiers
  if conversation_file:
    try:
      os.makedirs(config.directory, exist_ok=True)
      with open(conversation_file, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    except Exception as e:
      from ..logs import get_logger

      logger = get_logger("transcripts")
      logger.error(f"Failed to write to conversation transcript file {conversation_file}: {e}")




def detect_provider(model_name: str) -> str:
  """
  Detect the provider from the model name.

  Args:
    model_name: Full model name (e.g., "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

  Returns:
    Provider name (e.g., "bedrock", "openai", "anthropic")
  """
  model_name_lower = model_name.lower()

  if "bedrock/" in model_name_lower:
    return "bedrock"
  elif "openai/" in model_name_lower or "gpt-" in model_name_lower:
    return "openai"
  elif "anthropic/" in model_name_lower or "claude" in model_name_lower:
    return "anthropic"
  elif "ollama" in model_name_lower:
    return "ollama"
  elif "litellm_proxy/" in model_name_lower:
    return "litellm_proxy"
  else:
    return "unknown"


def format_provider_payload(payload: Dict[str, Any], provider: str) -> tuple[Dict[str, Any], List[str]]:
  """
  Format payload in provider-specific format and track transformations.

  Args:
    payload: Generic payload
    provider: Provider name

  Returns:
    Tuple of (formatted_payload, transformations_list)
  """
  transformations = []

  if provider == "anthropic" or provider == "bedrock":
    # Anthropic/Bedrock uses separate "system" field
    formatted = payload.copy()

    # Extract system message if present
    messages = formatted.get("messages", [])
    if messages and messages[0].get("role") == "system":
      system_msg = messages[0]
      formatted["system"] = system_msg.get("content", "")
      formatted["messages"] = messages[1:]
      transformations.append("Extracted system message → 'system' field")

    # Transform tool format: parameters → input_schema
    if "tools" in formatted:
      new_tools = []
      for tool in formatted["tools"]:
        if "function" in tool:
          new_tool = {
            "name": tool["function"].get("name"),
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"].get("parameters", {}),
          }
          new_tools.append(new_tool)
        else:
          new_tools.append(tool)

      if new_tools != formatted["tools"]:
        formatted["tools"] = new_tools
        transformations.append("Tool format: 'parameters' → 'input_schema'")

    # Add bedrock-specific wrapper
    if provider == "bedrock":
      model_id = formatted.get("model", "")
      body_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": formatted.get("max_tokens", 4096),
      }

      # Copy relevant fields to body
      for key in ["system", "messages", "tools", "temperature"]:
        if key in formatted:
          body_payload[key] = formatted[key]

      formatted = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": body_payload,
      }
      transformations.append("Wrapped in Bedrock-specific structure")

    return formatted, transformations

  elif provider == "openai":
    # OpenAI uses standard format
    return payload, ["No transformations (OpenAI native format)"]

  else:
    # Unknown provider, return as-is
    return payload, ["No transformations (unknown provider)"]


def log_raw_request(
  payload: Dict[str, Any],
  model_name: str,
  agent_name: Optional[str] = None,
  scope: Optional[str] = None,
  conversation: Optional[str] = None,
) -> None:
  """
  Log the raw API request payload.

  Outputs each message as a separate JSONL line to the per-conversation file.

  Args:
    payload: The raw API payload (dict)
    model_name: Full model name
    agent_name: Optional agent name
    scope: Optional scope identifier
    conversation: Optional conversation identifier
  """
  config = get_transcript_config()

  if not config.should_log():
    return

  # Detect provider from model name for transformations
  provider = detect_provider(model_name)

  # Format for the specific provider and get transformations
  formatted_payload, transformations = format_provider_payload(payload, provider)

  # Track which messages we've already output to avoid duplicates
  conv_key = f"{agent_name}_{scope}_{conversation}_{model_name}"
  messages_already_output = config.message_counts.get(conv_key, 0)

  # Collect all messages in order
  all_messages = []

  # Add system message if present (Anthropic/Bedrock format)
  if "system" in formatted_payload:
    all_messages.append({"role": "system", "content": formatted_payload["system"]})

  # Add all messages from request
  if "messages" in formatted_payload:
    all_messages.extend(formatted_payload["messages"])

  # Only output messages we haven't seen before
  new_messages = all_messages[messages_already_output:]
  for msg in new_messages:
    _output(
      json.dumps(msg, ensure_ascii=False),
      config,
      agent_name=agent_name,
      scope=scope,
      conversation=conversation,
    )

  # Update count
  config.message_counts[conv_key] = len(all_messages)


def log_raw_response(
  response: Dict[str, Any],
  model_name: str,
  agent_name: Optional[str] = None,
  scope: Optional[str] = None,
  conversation: Optional[str] = None,
) -> None:
  """
  Log the raw API response.

  Outputs the assistant message as a JSONL line to the per-conversation file.

  Args:
    response: The raw API response (dict)
    model_name: Full model name
    agent_name: Optional agent name
    scope: Optional scope identifier
    conversation: Optional conversation identifier
  """
  config = get_transcript_config()

  if not config.should_log():
    return

  # Track conversation key to update message count
  conv_key = f"{agent_name}_{scope}_{conversation}_{model_name}"

  # Extract assistant message from response
  assistant_msg = None

  if "choices" in response and len(response["choices"]) > 0:
    choice = response["choices"][0]
    if "message" in choice:
      message = choice["message"]
      assistant_msg = {"role": "assistant"}

      if "content" in message and message["content"]:
        assistant_msg["content"] = message["content"]

      if "tool_calls" in message and message["tool_calls"]:
        assistant_msg["tool_calls"] = []

        # Add tool calls to assistant message
        for tc in message["tool_calls"]:
          if "function" in tc:
            assistant_msg["tool_calls"].append({
              "id": tc.get("id", "unknown"),
              "type": "function",
              "function": {
                "name": tc["function"].get("name", "unknown"),
                "arguments": tc["function"].get("arguments", "{}")
              }
            })

  # For Anthropic format
  elif "content" in response:
    assistant_msg = {"role": "assistant"}
    content_parts = []
    tool_calls = []

    for block in response.get("content", []):
      if block.get("type") == "text":
        content_parts.append(block.get("text", ""))
      elif block.get("type") == "tool_use":
        tool_calls.append({
          "id": block.get("id", "unknown"),
          "type": "function",
          "function": {
            "name": block.get("name", "unknown"),
            "arguments": json.dumps(block.get("input", {}))
          }
        })

    if content_parts:
      assistant_msg["content"] = "\n".join(content_parts)

    if tool_calls:
      assistant_msg["tool_calls"] = tool_calls

  # Output the assistant message and update count
  if assistant_msg:
    _output(
      json.dumps(assistant_msg, ensure_ascii=False),
      config,
      agent_name=agent_name,
      scope=scope,
      conversation=conversation,
    )
    config.message_counts[conv_key] = config.message_counts.get(conv_key, 0) + 1

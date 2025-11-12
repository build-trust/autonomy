"""
Agent transcript logging utilities.

This module provides utilities for logging agent/model interactions (transcripts).
Transcripts show what's sent to the model (input) and what's returned (output),
in both human-readable and raw API format.

Environment Variables:
  AUTONOMY_TRANSCRIPTS=1          - Enable transcript logging (human-readable)
  AUTONOMY_TRANSCRIPTS_RAW=1      - Also show raw API payloads/responses
  AUTONOMY_TRANSCRIPTS_RAW_ONLY=1 - Show ONLY raw JSON (no human-readable)
  AUTONOMY_TRANSCRIPTS_FILE=/path - Write output to file (appends)

Examples:
  # Human-readable context and responses
  AUTONOMY_TRANSCRIPTS=1 python script.py

  # Human-readable + raw API payloads
  AUTONOMY_TRANSCRIPTS=1 AUTONOMY_TRANSCRIPTS_RAW=1 python script.py

  # Only raw JSON (for piping to jq, etc.)
  AUTONOMY_TRANSCRIPTS_RAW_ONLY=1 python script.py 2>/dev/null | jq

  # Save to file
  AUTONOMY_TRANSCRIPTS=1 AUTONOMY_TRANSCRIPTS_FILE=/tmp/transcript.log python script.py
"""

import json
import os
from typing import Dict, List, Optional, Any


class TranscriptConfig:
  """Configuration for transcript logging."""

  def __init__(self):
    """Initialize transcript configuration from environment variables."""
    self.enabled = os.environ.get("AUTONOMY_TRANSCRIPTS") == "1"
    self.raw = os.environ.get("AUTONOMY_TRANSCRIPTS_RAW") == "1"
    self.raw_only = os.environ.get("AUTONOMY_TRANSCRIPTS_RAW_ONLY") == "1"
    self.file = os.environ.get("AUTONOMY_TRANSCRIPTS_FILE")

    # If raw_only is set, it implies raw is enabled but human-readable is disabled
    if self.raw_only:
      self.raw = True
      self.enabled = False

  def should_log(self) -> bool:
    """Should we log anything at all?"""
    return self.enabled or self.raw or self.raw_only


# Global config instance
_config: Optional[TranscriptConfig] = None


def get_transcript_config() -> TranscriptConfig:
  """Get the global transcript configuration."""
  global _config
  if _config is None:
    _config = TranscriptConfig()
  return _config


def _output(text: str, config: TranscriptConfig) -> None:
  """
  Output text to stdout and/or file based on config.

  Args:
    text: Text to output
    config: Transcript configuration
  """
  # Print to stdout (unless raw_only mode with file output)
  if not (config.raw_only and config.file):
    print(text)

  # Write to file if configured
  if config.file:
    try:
      os.makedirs(os.path.dirname(config.file) if os.path.dirname(config.file) else ".", exist_ok=True)
      with open(config.file, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    except Exception as e:
      from ..logs import get_logger

      logger = get_logger("transcripts")
      logger.error(f"Failed to write to transcript file {config.file}: {e}")


def log_context(
  messages: List[dict],
  tools: Optional[List[dict]] = None,
  title: str = "CONTEXT",
  agent_name: Optional[str] = None,
  scope: Optional[str] = None,
  conversation: Optional[str] = None,
) -> None:
  """
  Log the context (messages + tools) in human-readable format.

  Args:
    messages: List of message dicts
    tools: Optional list of tool specifications
    title: Title for the output
    agent_name: Optional agent name
    scope: Optional scope identifier
    conversation: Optional conversation identifier
  """
  config = get_transcript_config()

  if not config.enabled:
    return

  lines = []
  separator = "=" * 80
  msg_separator = "-" * 80

  # Header
  lines.append("")
  lines.append(separator)
  header = f"{title} ({len(messages)} messages"
  if tools:
    header += f", {len(tools)} tools"
  header += ")"
  if agent_name:
    header += f" [agent={agent_name}]"
  if scope:
    header += f" [scope={scope}]"
  if conversation:
    header += f" [conversation={conversation}]"
  lines.append(header)
  lines.append(separator)
  lines.append("")

  # Messages section
  lines.append("MESSAGES:")
  lines.append("")

  for idx, msg in enumerate(messages, 1):
    role = msg.get("role", "UNKNOWN").upper()
    lines.append(f"[{idx}] {role}")

    # Handle content (string or dict)
    content = msg.get("content", "")
    if isinstance(content, dict):
      if "text" in content:
        for line in content["text"].splitlines():
          lines.append(f"  {line}")
      else:
        for line in json.dumps(content, indent=2).splitlines():
          lines.append(f"  {line}")
    elif isinstance(content, str):
      for line in content.splitlines():
        lines.append(f"  {line}")
    else:
      for line in json.dumps(content, indent=2).splitlines():
        lines.append(f"  {line}")

    # Handle tool calls
    if "tool_calls" in msg:
      lines.append("")
      lines.append("  Tool Calls:")
      for tool_call in msg["tool_calls"]:
        if isinstance(tool_call, dict):
          func_name = tool_call.get("function", {}).get("name", "unknown")
          func_args = tool_call.get("function", {}).get("arguments", "{}")
          lines.append(f"    - {func_name}:")

          # Parse and format arguments
          try:
            args_obj = json.loads(func_args) if isinstance(func_args, str) else func_args
            for line in json.dumps(args_obj, indent=2).splitlines():
              lines.append(f"      {line}")
          except:
            lines.append(f"      {func_args}")

    # Handle tool results
    if "tool_call_id" in msg:
      lines.append(f"  Tool Call ID: {msg.get('tool_call_id')}")
      if "name" in msg:
        lines.append(f"  Tool Name: {msg.get('name')}")

    lines.append(msg_separator)
    lines.append("")

  # Tools section
  if tools:
    lines.append("")
    lines.append("AVAILABLE TOOLS:")
    lines.append("")

    for tool_idx, tool in enumerate(tools, 1):
      tool_name = tool.get("function", {}).get("name", "unknown")
      tool_desc = tool.get("function", {}).get("description", "")
      tool_params = tool.get("function", {}).get("parameters", {})

      lines.append(f"[TOOL {tool_idx}] {tool_name}")
      if tool_desc:
        for line in tool_desc.splitlines():
          lines.append(f"  {line}")
      lines.append("")
      lines.append("  Parameters:")
      for line in json.dumps(tool_params, indent=2).splitlines():
        lines.append(f"    {line}")
      lines.append(msg_separator)
      lines.append("")

  # Footer
  lines.append(separator)
  lines.append("")

  _output("\n".join(lines), config)


def log_model_response(
  content: str,
  tool_calls: Optional[List[Dict]] = None,
  agent_name: Optional[str] = None,
  model_name: Optional[str] = None,
) -> None:
  """
  Log the model's response in human-readable format.

  Args:
    content: The response text from the model
    tool_calls: Optional list of tool calls made by the model
    agent_name: Optional agent name
    model_name: Optional model name
  """
  config = get_transcript_config()

  if not config.enabled:
    return

  lines = []
  separator = "=" * 80

  lines.append("")
  lines.append(separator)

  title = "MODEL RESPONSE"
  if agent_name:
    title += f" [agent={agent_name}]"
  if model_name:
    title += f" [model={model_name}]"

  lines.append(title)
  lines.append(separator)
  lines.append("")

  # Content
  if content:
    lines.append("Content:")
    for line in content.splitlines():
      lines.append(f"  {line}")
    lines.append("")

  # Tool calls
  if tool_calls:
    lines.append("Tool Calls:")
    lines.append("")
    for idx, tool_call in enumerate(tool_calls, 1):
      tool_name = tool_call.get("function", {}).get("name", "unknown")
      tool_args = tool_call.get("function", {}).get("arguments", "{}")

      lines.append(f"[{idx}] {tool_name}")

      # Parse and format arguments
      try:
        args_obj = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
        for line in json.dumps(args_obj, indent=2).splitlines():
          lines.append(f"  {line}")
      except:
        lines.append(f"  {tool_args}")

      lines.append("")

  lines.append(separator)
  lines.append("")

  _output("\n".join(lines), config)


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


def log_raw_request(payload: Dict[str, Any], provider: str, model_name: str, agent_name: Optional[str] = None) -> None:
  """
  Log the raw API request payload.

  Args:
    payload: The raw API payload (dict)
    provider: Provider name (e.g., "anthropic", "openai", "bedrock")
    model_name: Full model name
    agent_name: Optional agent name
  """
  config = get_transcript_config()

  if not config.raw:
    return

  # Format for the specific provider and get transformations
  formatted_payload, transformations = format_provider_payload(payload, provider)

  # In raw_only mode, output pure JSON
  if config.raw_only:
    _output(json.dumps(formatted_payload, indent=2, ensure_ascii=False), config)
  else:
    # Format with headers and transformations
    lines = []
    separator = "=" * 80

    lines.append("")
    lines.append(separator)

    if agent_name:
      lines.append(f"RAW API REQUEST → {provider.upper()} ({model_name}) [agent={agent_name}]")
    else:
      lines.append(f"RAW API REQUEST → {provider.upper()} ({model_name})")

    lines.append(separator)
    lines.append("")

    # Add transformations info
    if transformations:
      lines.append("Transformations applied:")
      for transform in transformations:
        lines.append(f"  • {transform}")
      lines.append("")
      lines.append("-" * 80)
      lines.append("")

    # Add JSON payload
    lines.extend(json.dumps(formatted_payload, indent=2, ensure_ascii=False).splitlines())

    lines.append("")
    lines.append(separator)
    lines.append("")

    _output("\n".join(lines), config)


def log_raw_response(
  response: Dict[str, Any], provider: str, model_name: str, agent_name: Optional[str] = None
) -> None:
  """
  Log the raw API response.

  Args:
    response: The raw API response (dict)
    provider: Provider name (e.g., "anthropic", "openai", "bedrock")
    model_name: Full model name
    agent_name: Optional agent name
  """
  config = get_transcript_config()

  if not config.raw:
    return

  # In raw_only mode, output pure JSON
  if config.raw_only:
    _output(json.dumps(response, indent=2, ensure_ascii=False), config)
  else:
    # Format with headers
    lines = []
    separator = "=" * 80

    lines.append("")
    lines.append(separator)

    if agent_name:
      lines.append(f"RAW API RESPONSE ← {provider.upper()} ({model_name}) [agent={agent_name}]")
    else:
      lines.append(f"RAW API RESPONSE ← {provider.upper()} ({model_name})")

    lines.append(separator)
    lines.append("")

    # Add JSON response
    lines.extend(json.dumps(response, indent=2, ensure_ascii=False).splitlines())

    lines.append("")
    lines.append(separator)
    lines.append("")

    _output("\n".join(lines), config)

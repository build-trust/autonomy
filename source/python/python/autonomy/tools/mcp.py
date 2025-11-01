import json
import re
from typing import Optional

from .protocol import InvokableTool


class McpTool(InvokableTool):
  def __init__(self, server_name, tool_name):
    self.server_name = server_name
    self.tool_name = tool_name
    self.name = f"{server_name}_{tool_name}"
    self.node = None

    if re.match(r"^[a-z0-9_-]+$", self.name) is None:
      raise ValueError("Server and tool names may only contain [a-z0-9_-] characters")

  async def spec(self):
    mcp_spec = await self.node.mcp_tool_spec(self.server_name, self.tool_name)
    if "not found" in mcp_spec:
      raise ValueError(f"Tool '{self.tool_name}' not found on server '{self.server_name}'")

    # Parse JSON with proper error handling
    try:
      parsed_spec = json.loads(mcp_spec)
    except json.JSONDecodeError as e:
      raise ValueError(
        f"MCP server returned invalid JSON for tool '{self.tool_name}' on server '{self.server_name}': {e}"
      )

    # Validate that parsed result is a dictionary
    if not isinstance(parsed_spec, dict):
      raise ValueError(
        f"MCP server returned invalid tool specification for '{self.tool_name}' on server '{self.server_name}': expected object, got {type(parsed_spec).__name__}"
      )

    # Validate required fields with descriptive errors
    if "description" not in parsed_spec:
      raise ValueError(
        f"MCP server returned invalid tool specification for '{self.tool_name}' on server '{self.server_name}': missing required field 'description'"
      )

    if "inputSchema" not in parsed_spec:
      raise ValueError(
        f"MCP server returned invalid tool specification for '{self.tool_name}' on server '{self.server_name}': missing required field 'inputSchema'"
      )

    # Validate field types
    description = parsed_spec["description"]
    input_schema = parsed_spec["inputSchema"]

    if not isinstance(description, str):
      raise ValueError(
        f"MCP server returned invalid tool specification for '{self.tool_name}' on server '{self.server_name}': 'description' must be a string, got {type(description).__name__}"
      )

    if input_schema is not None and not isinstance(input_schema, dict):
      raise ValueError(
        f"MCP server returned invalid tool specification for '{self.tool_name}' on server '{self.server_name}': 'inputSchema' must be an object or null, got {type(input_schema).__name__}"
      )

    spec = {
      "type": "function",
      "function": {
        "name": self.name,
        "description": description,
        "parameters": input_schema,
        "strict": False,
      },
    }

    return spec

  async def invoke(self, json_argument: Optional[str]) -> str:
    return await self.node.call_mcp_tool(self.server_name, self.tool_name, json_argument)

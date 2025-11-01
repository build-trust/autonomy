import json
import pytest

from autonomy.tools.mcp import McpTool


class MockNode:
  """Mock node for testing MCP tool behavior."""

  def __init__(self, mcp_spec_response=None, tool_result=None):
    self.mcp_spec_response = mcp_spec_response
    self.tool_result = tool_result

  async def mcp_tool_spec(self, server_name, tool_name):
    return self.mcp_spec_response

  async def call_mcp_tool(self, server_name, tool_name, json_argument):
    return self.tool_result


@pytest.mark.asyncio
async def test_mcp_tool_spec_missing_description_key():
  """Test that McpTool provides descriptive error when MCP server returns spec without 'description' key."""

  # Create malformed MCP spec missing 'description' key
  malformed_spec = json.dumps(
    {
      "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}
      # Missing 'description' key - should give descriptive error
    }
  )

  mock_node = MockNode(mcp_spec_response=malformed_spec)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should raise descriptive ValueError instead of generic KeyError
  with pytest.raises(ValueError, match="missing required field 'description'"):
    await tool.spec()


@pytest.mark.asyncio
async def test_mcp_tool_spec_missing_input_schema_key():
  """Test that McpTool provides descriptive error when MCP server returns spec without 'inputSchema' key."""

  # Create malformed MCP spec missing 'inputSchema' key
  malformed_spec = json.dumps(
    {
      "description": "A test tool description"
      # Missing 'inputSchema' key - should give descriptive error
    }
  )

  mock_node = MockNode(mcp_spec_response=malformed_spec)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should raise descriptive ValueError instead of generic KeyError
  with pytest.raises(ValueError, match="missing required field 'inputSchema'"):
    await tool.spec()


@pytest.mark.asyncio
async def test_mcp_tool_spec_missing_both_keys():
  """Test that McpTool provides descriptive error when MCP server returns spec missing both required keys."""

  # Create malformed MCP spec missing both required keys
  malformed_spec = json.dumps(
    {
      "some_other_field": "irrelevant data"
      # Missing both 'description' and 'inputSchema' keys
    }
  )

  mock_node = MockNode(mcp_spec_response=malformed_spec)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should raise descriptive ValueError (first one encountered will be 'description')
  with pytest.raises(ValueError, match="missing required field 'description'"):
    await tool.spec()


@pytest.mark.asyncio
async def test_mcp_tool_spec_invalid_json():
  """Test that McpTool provides descriptive error when MCP server returns invalid JSON."""

  # Return completely invalid JSON
  invalid_json = '{"description": "incomplete json"'  # Missing closing brace

  mock_node = MockNode(mcp_spec_response=invalid_json)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should raise descriptive ValueError instead of raw JSONDecodeError
  with pytest.raises(ValueError, match="returned invalid JSON"):
    await tool.spec()


@pytest.mark.asyncio
async def test_mcp_tool_spec_empty_response():
  """Test that McpTool provides descriptive error when MCP server returns empty response."""

  # Return empty string (invalid JSON)
  empty_response = ""

  mock_node = MockNode(mcp_spec_response=empty_response)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should raise descriptive ValueError instead of raw JSONDecodeError
  with pytest.raises(ValueError, match="returned invalid JSON"):
    await tool.spec()


@pytest.mark.asyncio
async def test_mcp_tool_spec_null_values():
  """Test that McpTool validates null values in required fields."""

  # Create spec with null values for required fields
  spec_with_nulls = json.dumps(
    {
      "description": None,  # null description
      "inputSchema": None,  # null inputSchema
    }
  )

  mock_node = MockNode(mcp_spec_response=spec_with_nulls)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should now raise descriptive ValueError for invalid type
  with pytest.raises(ValueError, match="'description' must be a string, got NoneType"):
    await tool.spec()


@pytest.mark.asyncio
async def test_mcp_tool_spec_wrong_types():
  """Test that McpTool validates types of required fields."""

  # Create spec with wrong types for required fields
  spec_with_wrong_types = json.dumps(
    {
      "description": 123,  # Should be string, but is number
      "inputSchema": "not an object",  # Should be object, but is string
    }
  )

  mock_node = MockNode(mcp_spec_response=spec_with_wrong_types)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should now raise descriptive ValueError for invalid type
  with pytest.raises(ValueError, match="'description' must be a string, got int"):
    await tool.spec()


@pytest.mark.asyncio
async def test_mcp_tool_spec_successful_case():
  """Test that McpTool works correctly with well-formed MCP spec."""

  # Create properly formed MCP spec
  valid_spec = json.dumps(
    {
      "description": "A valid test tool",
      "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
      },
    }
  )

  mock_node = MockNode(mcp_spec_response=valid_spec)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  spec = await tool.spec()

  # Verify correct parsing and structure
  expected_spec = {
    "type": "function",
    "function": {
      "name": "test_server_test_tool",
      "description": "A valid test tool",
      "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
      },
      "strict": False,
    },
  }

  assert spec == expected_spec


@pytest.mark.asyncio
async def test_mcp_tool_spec_nested_json_errors():
  """Test complex nested JSON parsing scenarios that could cause crashes."""

  # Test deeply nested invalid JSON
  complex_invalid = (
    '{"description": "test", "inputSchema": {"type": "object", "properties": {"nested": {"invalid": json}}}}'
  )

  mock_node = MockNode(mcp_spec_response=complex_invalid)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should raise descriptive ValueError instead of raw JSONDecodeError
  with pytest.raises(ValueError, match="returned invalid JSON"):
    await tool.spec()


# Additional edge cases that demonstrate the fragility of the current implementation


@pytest.mark.asyncio
async def test_mcp_tool_spec_array_instead_of_object():
  """Test that McpTool provides descriptive error when MCP server returns array instead of object."""

  # Return JSON array instead of object
  array_response = json.dumps(["description", "inputSchema"])

  mock_node = MockNode(mcp_spec_response=array_response)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should raise descriptive ValueError instead of generic TypeError
  with pytest.raises(ValueError, match="expected object, got list"):
    await tool.spec()


@pytest.mark.asyncio
async def test_mcp_tool_spec_unicode_handling():
  """Test that McpTool handles unicode characters properly."""

  # Create spec with unicode characters
  unicode_spec = json.dumps(
    {
      "description": "Tool with unicode: 🚀 émojis and àccénts",
      "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Query with ünïcödé"}},
      },
    }
  )

  mock_node = MockNode(mcp_spec_response=unicode_spec)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # This should work correctly if JSON parsing handles unicode properly
  spec = await tool.spec()
  assert "🚀" in spec["function"]["description"]
  assert "ünïcödé" in spec["function"]["parameters"]["properties"]["query"]["description"]


@pytest.mark.asyncio
async def test_mcp_tool_production_crash_scenario():
  """Test that demonstrates how MCP tool crashes in production scenarios."""

  # Simulate a real production scenario where an MCP server returns malformed response
  # This could happen due to server bugs, network corruption, or version mismatches

  class ProductionMockNode:
    """Simulate various failure modes that could occur in production."""

    def __init__(self, failure_mode="missing_description"):
      self.failure_mode = failure_mode

    async def mcp_tool_spec(self, server_name, tool_name):
      if self.failure_mode == "missing_description":
        # Server bug: forgot to include description
        return '{"inputSchema": {"type": "object"}}'
      elif self.failure_mode == "missing_input_schema":
        # Server bug: forgot to include inputSchema
        return '{"description": "A tool without schema"}'
      elif self.failure_mode == "corrupted_json":
        # Network corruption or server error
        return '{"description": "test", "inputSchema":'  # Truncated JSON
      elif self.failure_mode == "wrong_structure":
        # Server returns different API version or structure
        return '{"name": "tool", "schema": {}, "desc": "wrong keys"}'
      elif self.failure_mode == "server_error":
        # Server returns error message as JSON string
        return '"Server error occurred"'

    async def call_mcp_tool(self, server_name, tool_name, json_argument):
      return "mock result"

  # Test missing description - this will cause KeyError crash
  production_node = ProductionMockNode("missing_description")
  tool = McpTool("production_server", "search_tool")
  tool.node = production_node

  try:
    spec = await tool.spec()
    pytest.fail("Expected ValueError but tool.spec() succeeded")
  except ValueError as e:
    # This is the fix - production code now gives descriptive error instead of crashing
    assert "missing required field 'description'" in str(e)
    print(f"PRODUCTION FIX: Agent handled missing 'description' gracefully: {e}")

  # Test missing inputSchema - this will cause KeyError crash
  production_node = ProductionMockNode("missing_input_schema")
  tool = McpTool("production_server", "search_tool")
  tool.node = production_node

  try:
    spec = await tool.spec()
    pytest.fail("Expected ValueError but tool.spec() succeeded")
  except ValueError as e:
    # This is the fix - production code now gives descriptive error instead of crashing
    assert "missing required field 'inputSchema'" in str(e)
    print(f"PRODUCTION FIX: Agent handled missing 'inputSchema' gracefully: {e}")

  # Test corrupted JSON - this will cause JSONDecodeError crash
  production_node = ProductionMockNode("corrupted_json")
  tool = McpTool("production_server", "search_tool")
  tool.node = production_node

  try:
    spec = await tool.spec()
    pytest.fail("Expected ValueError but tool.spec() succeeded")
  except ValueError as e:
    # This is the fix - production code now wraps JSON errors with descriptive messages
    assert "returned invalid JSON" in str(e)
    print(f"PRODUCTION FIX: Agent handled corrupted JSON gracefully: {e}")

  # Test wrong structure - this will cause KeyError crash
  production_node = ProductionMockNode("wrong_structure")
  tool = McpTool("production_server", "search_tool")
  tool.node = production_node

  try:
    spec = await tool.spec()
    pytest.fail("Expected ValueError but tool.spec() succeeded")
  except ValueError as e:
    # This is the fix - production code now gives descriptive error for missing fields
    assert "missing required field 'description'" in str(e)
    print(f"PRODUCTION FIX: Agent handled wrong API structure gracefully: {e}")

  # Test server error response - this will cause TypeError crash
  production_node = ProductionMockNode("server_error")
  tool = McpTool("production_server", "search_tool")
  tool.node = production_node

  try:
    spec = await tool.spec()
    pytest.fail("Expected ValueError but tool.spec() succeeded")
  except ValueError as e:
    # This is the fix - production code now validates response type
    assert "expected object, got str" in str(e)
    print(f"PRODUCTION FIX: Agent handled string response gracefully: {e}")


@pytest.mark.asyncio
async def test_what_should_happen_instead():
  """Test demonstrating how MCP tools should handle errors gracefully."""

  # This test shows the behavior after we fixed the bug
  # Now we have robust error handling with descriptive messages

  malformed_spec = '{"wrong": "structure"}'

  mock_node = MockNode(mcp_spec_response=malformed_spec)
  tool = McpTool("test_server", "test_tool")
  tool.node = mock_node

  # After the fix, this should return a descriptive error instead of crashing
  try:
    spec = await tool.spec()
    # This should not succeed with malformed input, but should fail gracefully
    pytest.fail("Tool should have failed gracefully, not succeeded")
  except ValueError as e:
    # After fix, we get a descriptive ValueError instead of a generic KeyError
    assert "missing required field 'description'" in str(e)
    assert "test_server" in str(e) and "test_tool" in str(e)
    print(f"SUCCESS: Graceful error handling with descriptive message: {e}")

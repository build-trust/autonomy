import inspect
import json

from autonomy.tools.tool import function_spec, parameter_info_from_docstring, Tool


def test_function_spec():
  def sample_function(param1: int, param2: str) -> None:
    """A sample function for testing.

    Args:
        param1 (int): An integer parameter.
        param2 (str): A string parameter.
    """
    pass

  name, spec = function_spec(sample_function)
  assert name == "sample_function"

  expected_spec = {
    "type": "function",
    "function": {
      "name": "sample_function",
      "description": inspect.cleandoc("""A sample function for testing.

            Args:
                param1 (int): An integer parameter.
                param2 (str): A string parameter.
            """),
      "parameters": {
        "type": "object",
        "properties": {
          "param1": {"type": "integer", "description": "parameter param1"},
          "param2": {"type": "string", "description": "parameter param2"},
        },
        "required": ["param1", "param2"],
      },
    },
  }
  assert spec == expected_spec


def test_function_spec_no_docstring():
  def function_without_docstring(param1: int, param2: str) -> None:
    pass

  name, spec = function_spec(function_without_docstring)
  assert name == "function_without_docstring"

  expected_spec = {
    "type": "function",
    "function": {
      "name": "function_without_docstring",
      "description": "Function function_without_docstring",
      "parameters": {
        "type": "object",
        "properties": {
          "param1": {"type": "integer", "description": "parameter param1"},
          "param2": {"type": "string", "description": "parameter param2"},
        },
        "required": ["param1", "param2"],
      },
    },
  }
  assert spec == expected_spec


def test_function_spec_no_type_hints():
  def function_with_docstring_no_type_hints(param1, param2):
    """A sample function for testing.

    Args:
        param1 (int): An integer parameter.
        param2 (str): A string parameter.
    """
    pass

  name, spec = function_spec(function_with_docstring_no_type_hints)
  assert name == "function_with_docstring_no_type_hints"

  expected_spec = {
    "type": "function",
    "function": {
      "name": "function_with_docstring_no_type_hints",
      "description": inspect.cleandoc("""A sample function for testing.

            Args:
                param1 (int): An integer parameter.
                param2 (str): A string parameter.
            """),
      "parameters": {
        "type": "object",
        "properties": {
          "param1": {"type": "integer", "description": "parameter param1"},
          "param2": {"type": "string", "description": "parameter param2"},
        },
        "required": ["param1", "param2"],
      },
    },
  }
  assert spec == expected_spec


def test_function_spec_no_docstring_no_type_hints():
  def function_no_docstring_no_type_hints(param1, param2):
    pass

  name, spec = function_spec(function_no_docstring_no_type_hints)
  assert name == "function_no_docstring_no_type_hints"

  expected_spec = {
    "type": "function",
    "function": {
      "name": "function_no_docstring_no_type_hints",
      "description": "Function function_no_docstring_no_type_hints",
      "parameters": {
        "type": "object",
        "properties": {
          "param1": {"type": "string", "description": "parameter param1"},
          "param2": {"type": "string", "description": "parameter param2"},
        },
        "required": ["param1", "param2"],
      },
    },
  }
  assert spec == expected_spec


def test_parameter_info_from_docstring():
  docstring = """A sample function for testing.

    Args:
        param1 (int): An integer parameter.
        param2 (str): A string parameter.
    """
  p_info = parameter_info_from_docstring(docstring)

  expected = {"param1": "int", "param2": "str"}
  assert p_info == expected


def test_parameter_info_from_docstring_no_arguments():
  docstring = """A sample function with no parameters."""
  p_info = parameter_info_from_docstring(docstring)

  expected = {}
  assert p_info == expected


def test_function_spec_with_optional_parameters():
  def function_with_optional_params(required_param: str, optional_param: str = "default"):
    """A function with both required and optional parameters.

    Args:
        required_param (str): A required parameter.
        optional_param (str): An optional parameter with default value.
    """
    pass

  name, spec = function_spec(function_with_optional_params)
  assert name == "function_with_optional_params"

  expected_spec = {
    "type": "function",
    "function": {
      "name": "function_with_optional_params",
      "description": inspect.cleandoc("""A function with both required and optional parameters.

            Args:
                required_param (str): A required parameter.
                optional_param (str): An optional parameter with default value.
            """),
      "parameters": {
        "type": "object",
        "properties": {
          "required_param": {"type": "string", "description": "parameter required_param"},
          "optional_param": {"type": "string", "description": "parameter optional_param"},
        },
        "required": ["required_param"],  # Only required_param should be required
      },
    },
  }
  assert spec == expected_spec


def test_function_spec_with_only_optional_parameters():
  def function_with_only_optional_params(param1: str = "default1", param2: int = 42):
    """A function with only optional parameters."""
    pass

  name, spec = function_spec(function_with_only_optional_params)
  assert name == "function_with_only_optional_params"

  # Should have no required parameters
  assert spec["function"]["parameters"]["required"] == []

  # Should have both parameters in properties
  properties = spec["function"]["parameters"]["properties"]
  assert "param1" in properties
  assert "param2" in properties
  assert properties["param1"]["type"] == "string"
  assert properties["param2"]["type"] == "integer"


def test_function_spec_mixed_required_optional():
  def mixed_function(req1: str, req2: int, opt1: str = "default", opt2: bool = True):
    """Function with mixed required and optional parameters."""
    pass

  name, spec = function_spec(mixed_function)

  # Should have exactly 2 required parameters
  required_params = spec["function"]["parameters"]["required"]
  assert len(required_params) == 2
  assert "req1" in required_params
  assert "req2" in required_params

  # Optional parameters should not be in required list
  assert "opt1" not in required_params
  assert "opt2" not in required_params

  # All parameters should be in properties
  properties = spec["function"]["parameters"]["properties"]
  assert len(properties) == 4
  assert all(param in properties for param in ["req1", "req2", "opt1", "opt2"])


def test_error_context_preservation():
  """Test that exception context is preserved in error responses."""

  def tool_that_raises_value_error():
    """A tool that raises a ValueError."""
    raise ValueError("Test error with context")

  def tool_that_raises_key_error():
    """A tool that raises a KeyError."""
    data = {"key": "value"}
    return data["missing_key"]

  def tool_that_raises_type_error():
    """A tool that raises a TypeError."""
    return "string" + 123

  import asyncio

  # Test ValueError context preservation
  tool1 = Tool(tool_that_raises_value_error)
  result1 = asyncio.run(tool1.invoke(""))

  # Should preserve error type and message
  assert "Tool execution failed: ValueError: Test error with context" == result1
  assert "ValueError" in result1
  assert "Test error with context" in result1

  # Test KeyError context preservation
  tool2 = Tool(tool_that_raises_key_error)
  result2 = asyncio.run(tool2.invoke(""))

  assert "Tool execution failed: KeyError: 'missing_key'" == result2
  assert "KeyError" in result2
  assert "missing_key" in result2

  # Test TypeError context preservation
  tool3 = Tool(tool_that_raises_type_error)
  result3 = asyncio.run(tool3.invoke(""))

  assert "Tool execution failed: TypeError:" in result3
  assert "concatenate str" in result3


def test_json_parsing_error_context():
  """Test that JSON parsing errors preserve context."""

  def simple_tool(param: str):
    return f"Got: {param}"

  import asyncio

  tool = Tool(simple_tool)
  result = asyncio.run(tool.invoke('{"invalid": json}'))

  # Should show ValueError with JSON format error context
  assert "Tool execution failed: ValueError: Invalid JSON format:" in result
  assert "Expecting value" in result


def test_successful_execution_unchanged():
  """Test that successful tool execution is not affected by error handling improvements."""

  def successful_tool(message: str = "default"):
    return f"Success: {message}"

  import asyncio

  tool = Tool(successful_tool)

  # Test with no arguments (uses default)
  result1 = asyncio.run(tool.invoke(""))
  assert result1 == "Success: default"

  # Test with arguments
  result2 = asyncio.run(tool.invoke('{"message": "custom"}'))
  assert result2 == "Success: custom"


def test_json_payload_size_protection():
  """Test that JSON payload size limits protect against DoS attacks."""

  def data_tool(data: str = ""):
    return f"Length: {len(data)}"

  import asyncio

  tool = Tool(data_tool)

  # Test normal size payload (should work)
  normal_data = "x" * 1000  # 1KB
  normal_json = json.dumps({"data": normal_data})
  result1 = asyncio.run(tool.invoke(normal_json))
  assert "Length: 1000" in result1

  # Test oversized payload (should be rejected)
  oversized_data = "x" * 2000000  # 2MB
  oversized_json = json.dumps({"data": oversized_data})
  result2 = asyncio.run(tool.invoke(oversized_json))

  assert "Tool execution failed: ValueError: JSON argument too large:" in result2
  assert "bytes (max: 1,048,576)" in result2


def test_json_argument_schema_validation():
  """Test that JSON arguments are validated against function signature."""

  def typed_tool(name: str, age: int, active: bool = True):
    return f"{name}-{age}-{active}"

  import asyncio

  tool = Tool(typed_tool)

  # Test valid arguments
  result1 = asyncio.run(tool.invoke('{"name": "Alice", "age": 30}'))
  assert result1 == "Alice-30-True"

  # Test extra unexpected parameters
  result2 = asyncio.run(tool.invoke('{"name": "Bob", "age": 25, "extra": "bad"}'))
  assert "Tool execution failed: ValueError: Unexpected arguments: extra" in result2

  # Test missing required parameters
  result3 = asyncio.run(tool.invoke('{"name": "Charlie"}'))
  assert "Tool execution failed: ValueError: Missing required arguments: age" in result3

  # Test multiple missing parameters
  result4 = asyncio.run(tool.invoke("{}"))
  assert "Tool execution failed: ValueError: Missing required arguments:" in result4
  assert "age" in result4 and "name" in result4


def test_json_edge_cases_and_security():
  """Test JSON parsing edge cases and security improvements."""

  def simple_tool(param: str = "default"):
    return f"Param: {param}"

  import asyncio

  tool = Tool(simple_tool)

  # Test empty JSON object
  result1 = asyncio.run(tool.invoke("{}"))
  assert result1 == "Param: default"

  # Test non-object JSON (should be rejected)
  result2 = asyncio.run(tool.invoke('["array", "not", "object"]'))
  assert "Tool execution failed: ValueError: JSON argument must be an object, got list" in result2

  # Test whitespace-only argument
  result3 = asyncio.run(tool.invoke("   \t\n  "))
  assert result3 == "Param: default"

  # Test JSON with null values
  result4 = asyncio.run(tool.invoke('{"param": null}'))
  assert "Param: None" in result4


def test_improved_json_error_messages():
  """Test that JSON parsing provides helpful error messages."""

  def multi_param_tool(name: str, age: int, email: str, active: bool = True):
    return f"User: {name}"

  import asyncio

  tool = Tool(multi_param_tool)

  # Test multiple missing required parameters
  result1 = asyncio.run(tool.invoke('{"name": "Alice"}'))
  assert "Missing required arguments:" in result1
  assert "age" in result1 and "email" in result1

  # Test multiple unexpected parameters
  result2 = asyncio.run(tool.invoke('{"name": "Bob", "age": 30, "email": "bob@test.com", "bad1": "x", "bad2": "y"}'))
  assert "Unexpected arguments:" in result2
  assert "bad1" in result2 and "bad2" in result2

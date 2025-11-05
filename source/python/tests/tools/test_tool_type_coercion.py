import pytest
from autonomy.tools.tool import Tool


# Test functions with different type signatures
async def func_with_int(count: int) -> str:
  """Function that expects an integer."""
  return f"count={count}, type={type(count).__name__}"


async def func_with_float(value: float) -> str:
  """Function that expects a float."""
  return f"value={value}, type={type(value).__name__}"


async def func_with_bool(enabled: bool) -> str:
  """Function that expects a boolean."""
  return f"enabled={enabled}, type={type(enabled).__name__}"


async def func_with_str(name: str) -> str:
  """Function that expects a string."""
  return f"name={name}, type={type(name).__name__}"


async def func_with_multiple_types(count: int, ratio: float, enabled: bool, name: str) -> str:
  """Function with multiple typed parameters."""
  return f"count={count}({type(count).__name__}), ratio={ratio}({type(ratio).__name__}), enabled={enabled}({type(enabled).__name__}), name={name}({type(name).__name__})"


async def func_with_optional(value: int = 10) -> str:
  """Function with optional parameter."""
  return f"value={value}, type={type(value).__name__}"


@pytest.mark.asyncio
async def test_coerce_string_to_int():
  """Test coercing string to int."""
  tool = Tool(func_with_int)
  result = await tool.invoke('{"count": "42"}')
  assert "count=42" in result
  assert "type=int" in result


@pytest.mark.asyncio
async def test_coerce_string_to_float():
  """Test coercing string to float."""
  tool = Tool(func_with_float)
  result = await tool.invoke('{"value": "3.14"}')
  assert "value=3.14" in result
  assert "type=float" in result


@pytest.mark.asyncio
async def test_coerce_int_to_float():
  """Test coercing int to float."""
  tool = Tool(func_with_float)
  result = await tool.invoke('{"value": 42}')
  assert "value=42" in result
  assert "type=float" in result


@pytest.mark.asyncio
async def test_coerce_string_to_bool_true():
  """Test coercing various string representations to True."""
  tool = Tool(func_with_bool)

  for true_value in ["true", "True", "TRUE", "1", "yes", "Yes", "on", "On"]:
    result = await tool.invoke(f'{{"enabled": "{true_value}"}}')
    assert "enabled=True" in result
    assert "type=bool" in result


@pytest.mark.asyncio
async def test_coerce_string_to_bool_false():
  """Test coercing various string representations to False."""
  tool = Tool(func_with_bool)

  for false_value in ["false", "False", "FALSE", "0", "no", "No", "off", "Off"]:
    result = await tool.invoke(f'{{"enabled": "{false_value}"}}')
    assert "enabled=False" in result
    assert "type=bool" in result


@pytest.mark.asyncio
async def test_coerce_int_to_bool():
  """Test coercing int to bool."""
  tool = Tool(func_with_bool)

  # Non-zero int should be True
  result = await tool.invoke('{"enabled": 1}')
  assert "enabled=True" in result

  # Zero int should be False
  result = await tool.invoke('{"enabled": 0}')
  assert "enabled=False" in result


@pytest.mark.asyncio
async def test_coerce_number_to_string():
  """Test coercing number to string."""
  tool = Tool(func_with_str)
  result = await tool.invoke('{"name": 123}')
  assert "name=123" in result
  assert "type=str" in result


@pytest.mark.asyncio
async def test_coerce_multiple_types():
  """Test coercing multiple arguments with different types."""
  tool = Tool(func_with_multiple_types)
  result = await tool.invoke('{"count": "5", "ratio": "3.14", "enabled": "true", "name": "test"}')
  assert "count=5(int)" in result
  assert "ratio=3.14(float)" in result
  assert "enabled=True(bool)" in result
  assert "name=test(str)" in result


@pytest.mark.asyncio
async def test_no_coercion_when_types_match():
  """Test that no coercion happens when types already match."""
  tool = Tool(func_with_int)
  # JSON naturally deserializes numbers as int/float
  result = await tool.invoke('{"count": 42}')
  assert "count=42" in result
  assert "type=int" in result


@pytest.mark.asyncio
async def test_invalid_int_coercion():
  """Test that invalid int coercion raises appropriate error."""
  tool = Tool(func_with_int)
  result = await tool.invoke('{"count": "not_a_number"}')
  assert "Tool execution failed" in result
  assert "has invalid type" in result.lower()


@pytest.mark.asyncio
async def test_invalid_float_coercion():
  """Test that invalid float coercion raises appropriate error."""
  tool = Tool(func_with_float)
  result = await tool.invoke('{"value": "not_a_number"}')
  assert "Tool execution failed" in result
  assert "has invalid type" in result.lower()


@pytest.mark.asyncio
async def test_invalid_bool_coercion():
  """Test that invalid bool coercion raises appropriate error."""
  tool = Tool(func_with_bool)
  result = await tool.invoke('{"enabled": "maybe"}')
  assert "Tool execution failed" in result
  assert "has invalid type" in result.lower()


@pytest.mark.asyncio
async def test_optional_parameter_with_default():
  """Test optional parameters with defaults work correctly."""
  tool = Tool(func_with_optional)

  # Call without parameter - should use default
  result = await tool.invoke('{}')
  assert "value=10" in result
  assert "type=int" in result

  # Call with parameter as string - should coerce
  result = await tool.invoke('{"value": "25"}')
  assert "value=25" in result
  assert "type=int" in result


@pytest.mark.asyncio
async def test_empty_arguments():
  """Test that empty arguments work correctly."""
  tool = Tool(func_with_optional)
  result = await tool.invoke('')
  assert "value=10" in result  # Should use default


@pytest.mark.asyncio
async def test_null_arguments():
  """Test that null arguments work correctly."""
  tool = Tool(func_with_optional)
  result = await tool.invoke(None)
  assert "value=10" in result  # Should use default

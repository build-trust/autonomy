import json
import inspect
import re
import traceback

from typing import get_type_hints, Optional, get_origin, get_args, List, Dict, Set, Tuple, Union
from functools import wraps
from docstring_parser import parse

from .protocol import InvokableTool
from ..logs.logs import InfoContext


class Tool(InvokableTool, InfoContext):
  _logger = None

  @classmethod
  def class_logger(cls):
    if cls._logger:
      return cls._logger
    else:
      from ..logs.logs import get_logger

      cls._logger = get_logger("tool")
      return cls._logger

  def __init__(self, func):
    self.logger = Tool.class_logger()
    name, spec = function_spec(func)
    self.func = wrap(func)
    self.name = name
    self._spec = spec
    self.node = None
    if re.match(r"^[a-z0-9_-]+$", spec["function"]["name"]) is None:
      raise ValueError("Tool name may only contain [a-z0-9_-] characters")

  async def spec(self) -> dict:
    return self._spec

  async def invoke(self, json_argument: Optional[str]) -> str:
    with self.info(f"Invoke tool: '{self.name}'", f"invoked tool: '{self.name}'"):
      self.logger.debug(f"The tool arguments are: {json_argument}")
      try:
        args = self._parse_and_validate_arguments(json_argument)
        tool_response = str(await self.func(**args))
        self.logger.debug(f"The tool call succeeded: {tool_response}")
      except Exception as e:
        # Preserve full exception context for debugging
        error_details = {
          "error_type": type(e).__name__,
          "error_message": str(e),
          "tool_name": self.name,
          "arguments": json_argument,
          "traceback": traceback.format_exc(),
        }

        # Log detailed error information for debugging
        self.logger.error(
          f"Tool '{self.name}' execution failed: {error_details['error_type']}: {error_details['error_message']}"
        )
        self.logger.debug(f"Full error context: {error_details}")

        # Return user-friendly error message while preserving key details
        tool_response = f"Tool execution failed: {error_details['error_type']}: {error_details['error_message']}"
      return tool_response

  def _parse_and_validate_arguments(self, json_argument: Optional[str]) -> dict:
    """Parse and validate JSON arguments with proper error handling and size limits."""

    # Handle empty/null arguments
    if json_argument is None or json_argument.strip() == "":
      return {}

    # Check payload size limit (default 1MB)
    MAX_JSON_SIZE = 1024 * 1024  # 1MB
    if len(json_argument) > MAX_JSON_SIZE:
      raise ValueError(f"JSON argument too large: {len(json_argument):,} bytes (max: {MAX_JSON_SIZE:,})")

    # Parse JSON with proper error handling
    try:
      args = json.loads(json_argument)
    except json.JSONDecodeError as e:
      raise ValueError(f"Invalid JSON format: {str(e)}")

    # Validate that args is a dictionary
    if not isinstance(args, dict):
      raise ValueError(f"JSON argument must be an object, got {type(args).__name__}")

    # Get function signature for validation
    signature = inspect.signature(self.func)

    # Validate arguments against function signature
    self._validate_arguments_against_signature(args, signature)

    # Coerce argument types to match function type hints
    args = self._coerce_argument_types(args)

    return args

  def _validate_arguments_against_signature(self, args: dict, signature: inspect.Signature) -> None:
    """Validate arguments against the function signature."""

    param_names = set(signature.parameters.keys())
    provided_args = set(args.keys())

    # Check for unexpected arguments
    extra_args = provided_args - param_names
    if extra_args:
      raise ValueError(f"Unexpected arguments: {', '.join(sorted(extra_args))}")

    # Check for missing required arguments
    missing_required = set()
    for param_name, param in signature.parameters.items():
      if param.default == inspect.Parameter.empty and param_name not in args:
        missing_required.add(param_name)

    if missing_required:
      raise ValueError(f"Missing required arguments: {', '.join(sorted(missing_required))}")

  def _coerce_argument_types(self, args: dict) -> dict:
    """Coerce argument types to match function type hints.

    JSON deserialization may produce strings where integers/floats/bools are expected.
    This method attempts to coerce arguments to their expected types based on type hints.
    """
    try:
      type_hints = get_type_hints(self.func)
    except Exception:
      # If we can't get type hints, return args unchanged
      return args

    coerced_args = {}
    for arg_name, arg_value in args.items():
      if arg_name not in type_hints:
        # No type hint available, use as-is
        coerced_args[arg_name] = arg_value
        continue

      expected_type = type_hints[arg_name]

      # Handle Optional types (Optional[X] is Union[X, None])
      origin = get_origin(expected_type)
      if origin is Union or (
        hasattr(expected_type, "__class__") and expected_type.__class__.__name__ == "UnionType"
      ):
        # For Optional/Union types, try to get the non-None type
        type_args = get_args(expected_type)
        if type_args:
          # Get first non-None type
          expected_type = next((t for t in type_args if t is not type(None)), expected_type)

      # Try to coerce the value to the expected type
      try:
        coerced_value = self._coerce_value(arg_value, expected_type)
        # Log type coercion if it occurred
        if type(arg_value) != type(coerced_value):
          self.logger.debug(
            f"Coerced argument '{arg_name}': {type(arg_value).__name__}({arg_value!r}) -> "
            f"{type(coerced_value).__name__}({coerced_value!r})"
          )
        coerced_args[arg_name] = coerced_value
      except (ValueError, TypeError) as e:
        # If coercion fails, log and raise a clear error
        self.logger.debug(
          f"Failed to coerce argument '{arg_name}' from {type(arg_value).__name__} to {expected_type.__name__}: {e}"
        )
        raise ValueError(
          f"Argument '{arg_name}' has invalid type: expected {expected_type.__name__}, "
          f"got {type(arg_value).__name__} (value: {arg_value!r})"
        )

    return coerced_args

  def _coerce_value(self, value, expected_type):
    """Coerce a single value to the expected type.

    Args:
      value: The value to coerce
      expected_type: The target type

    Returns:
      The coerced value

    Raises:
      ValueError: If coercion is not possible
      TypeError: If types are incompatible
    """
    # Handle None values
    if value is None:
      return None

    # Get the origin type for generic types (List, Dict, etc.)
    origin = get_origin(expected_type)

    # For generic types like List[int], Dict[str, Any], check against the origin
    if origin is not None:
      # List, Dict, Set, Tuple, etc.
      if origin is list:
        if isinstance(value, list):
          return value
      elif origin is dict:
        if isinstance(value, dict):
          return value
      elif origin is set:
        if isinstance(value, set):
          return value
      elif origin is tuple:
        if isinstance(value, tuple):
          return value
      # For other generic types, try to check against the origin
      try:
        if isinstance(value, origin):
          return value
      except TypeError:
        # origin might not be a valid type for isinstance
        pass

    # Try isinstance check for non-generic types
    try:
      if isinstance(value, expected_type):
        return value
    except TypeError:
      # expected_type might be a generic type like List[int] which can't be used with isinstance
      pass

    # Coerce to int
    if expected_type is int:
      if isinstance(value, str):
        return int(value)
      elif isinstance(value, (float, bool)):
        return int(value)
      else:
        raise TypeError(f"Cannot coerce {type(value).__name__} to int")

    # Coerce to float
    elif expected_type is float:
      if isinstance(value, (str, int)):
        return float(value)
      else:
        raise TypeError(f"Cannot coerce {type(value).__name__} to float")

    # Coerce to bool
    elif expected_type is bool:
      if isinstance(value, str):
        # Handle common string representations of booleans
        lower_value = value.lower()
        if lower_value in ("true", "1", "yes", "on"):
          return True
        elif lower_value in ("false", "0", "no", "off"):
          return False
        else:
          raise ValueError(f"Cannot coerce string '{value}' to bool")
      elif isinstance(value, (int, float)):
        return bool(value)
      else:
        raise TypeError(f"Cannot coerce {type(value).__name__} to bool")

    # Coerce to str
    elif expected_type is str:
      return str(value)

    # For complex types (list, dict, etc.), check basic type compatibility
    elif expected_type is list or expected_type is List:
      if isinstance(value, list):
        return value
    elif expected_type is dict or expected_type is Dict:
      if isinstance(value, dict):
        return value

    # For other types, return as-is and let Python handle it
    return value


def wrap(f) -> callable:
  @wraps(f)
  async def wrapper(**kwargs):
    r = f(**kwargs)
    if inspect.iscoroutine(r):
      return await r
    return r

  return wrapper


def function_spec(f) -> (str, dict):
  f_name = f.__name__
  f_description = inspect.cleandoc(f.__doc__) if f.__doc__ else f"Function {f_name}"
  f_parameters = parameters_spec(f)
  return f_name, {
    "type": "function",
    "function": {"name": f_name, "description": f_description, "parameters": f_parameters},
  }


def parameters_spec(f):
  f_parameters = {"type": "object", "properties": {}, "required": []}

  signature = inspect.signature(f)
  type_hints = get_type_hints(f)

  p_info_from_docstring = parameter_info_from_docstring(f.__doc__)
  for p_name, p in signature.parameters.items():
    # Get parameter type from typehint and docstring
    # Prefer the value from the docstring if available.
    # Convert from python type to json schema type
    p_type = str(type_hints.get(p_name, "Any")).replace("<class '", "").replace("'>", "")
    p_type = p_info_from_docstring.get(p_name, p_type)
    p_type = to_json_schema_type(p_type)

    f_parameters["properties"][p_name] = {"type": p_type, "description": f"parameter {p_name}"}

    # Only add to required list if parameter has no default value
    if p.default == inspect.Parameter.empty:
      f_parameters["required"].append(p_name)

  return f_parameters


def to_json_schema_type(p_type):
  return {
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "str": "string",
    "list": "array",
    "dict": "object",
  }.get(p_type, "string")


def parameter_info_from_docstring(docstring):
  p_info = {}
  if not docstring:
    return p_info

  parsed = parse(docstring)
  for parameter in parsed.params:
    p_name = parameter.arg_name
    p_type = parameter.type_name
    p_info[p_name] = p_type

  return p_info

import json
import inspect
import re
import traceback

from typing import get_type_hints, Optional
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

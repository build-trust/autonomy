import re


def validate_name(name: str) -> None:
  """
  Validate that a name contains only alphanumeric characters, hyphens, and underscores.

  Args:
      name: The name to validate

  Raises:
      ValueError: If the name contains invalid characters
  """
  if not name:
    raise ValueError("Name cannot be empty")

  # Allow alphanumeric characters, hyphens, and underscores
  if not re.match(r"^[a-zA-Z0-9_-]+$", name):
    raise ValueError(f"Invalid name '{name}'. Name must contain only alphanumeric characters, hyphens, and underscores")

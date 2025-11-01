import re

from .validate_socket_address import validate_socket_address


def parse_socket_address(address: str | int, default_host: str = "127.0.0.1") -> tuple[str, int]:
  """
  Parse a socket address and return the host and port.
  If the address is just a port number, return the default host with that port.

  Args:
    address: Socket address as either:
      - int: port number (1-65535)
      - str: "host:port", ":port", or "port" format
    default_host: Default host to use when only port is provided

  Returns:
    tuple[str, int]: (host, port)

  Raises:
    ValueError: If address format is invalid or values are out of range
  """
  if address is None:
    raise ValueError("Address cannot be None")

  host = default_host
  port = None

  if isinstance(address, int):
    port = address
  elif isinstance(address, str):
    address = address.strip()
    if not address:
      raise ValueError("Address string cannot be empty")

    # Handle IPv6 addresses with port (e.g., "[::1]:8080")
    ipv6_match = re.match(r"^\[([^\]]+)\]:(.+)$", address)
    if ipv6_match:
      host = ipv6_match.group(1)
      port_str = ipv6_match.group(2)
      try:
        port = int(port_str)
      except ValueError:
        raise ValueError(f"Invalid port number in IPv6 address: {address}")
    elif ":" in address:
      # Handle IPv4 or hostname with port
      if address.count(":") > 1:
        # This might be an IPv6 address without brackets - not supported for port parsing
        raise ValueError(f"Ambiguous address format. For IPv6 addresses with port, use [host]:port format: {address}")

      parts = address.split(":", 1)
      if len(parts) != 2:
        raise ValueError(f"Invalid address format: {address}")

      _host, port_str = parts
      if _host:
        host = _host.strip()

      if not port_str:
        raise ValueError(f"Port cannot be empty in address: {address}")

      try:
        port = int(port_str.strip())
      except ValueError:
        raise ValueError(f"Invalid port number '{port_str}' in address: {address}")
    else:
      # Try to parse as just a port number
      try:
        port = int(address)
      except ValueError:
        raise ValueError(
          f"Invalid address format. Expected 'host:port', ':port', port number, or '[IPv6]:port': {address}"
        )
  else:
    raise ValueError(f"Address must be int or str, got {type(address).__name__}: {address}")

  if port is None:
    raise ValueError(f"Could not extract port from address: {address}")

  validate_socket_address(host, port)
  return host, port

import socket


def validate_socket_address(host: str, port: int) -> None:
  """
  Validates host and port values.

  Args:
    host: Hostname or IP address
    port: Port number

  Raises:
    ValueError: If host or port is invalid
  """
  if not isinstance(host, str):
    raise ValueError(f"Host must be a string, got {type(host).__name__}: {host}")

  if not isinstance(port, int):
    raise ValueError(f"Port must be an integer, got {type(port).__name__}: {port}")

  if port < 0 or port > 65535:
    raise ValueError(f"Port {port} is out of range (0-65535)")

  host = host.strip()
  if not host:
    raise ValueError("Host cannot be empty")

  try:
    # Check if host resolves - this validates both IPv4, IPv6, and hostnames
    socket.getaddrinfo(host, port, family=socket.AF_UNSPEC)
  except socket.gaierror as e:
    raise ValueError(f"Host '{host}' is not a valid address: {e}")

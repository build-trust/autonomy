from socket import socket, AF_INET, SOCK_STREAM


def is_port_free(port: int) -> bool:
  with socket(AF_INET, SOCK_STREAM) as s:
    try:
      s.bind(("", port))
      return True
    except OSError:
      return False

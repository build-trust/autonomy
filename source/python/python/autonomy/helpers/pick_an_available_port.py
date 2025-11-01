from socket import socket, AF_INET, SOCK_STREAM


def pick_an_available_port() -> int:
  with socket(AF_INET, SOCK_STREAM) as s:
    s.bind(("", 0))
    return s.getsockname()[1]

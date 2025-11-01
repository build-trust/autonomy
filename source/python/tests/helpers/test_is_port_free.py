import socket

from autonomy.helpers.is_port_free import is_port_free


class TestIsPortFree:
  def test_free_port(self):
    with socket.socket() as s:
      s.bind(("", 0))
      port = s.getsockname()[1]

    # assert that the port is free once the socket is closed
    assert is_port_free(port) is True

  def test_occupied_port(self):
    with socket.socket() as s:
      s.bind(("", 0))
      port = s.getsockname()[1]

      # assert that the port is not free while the socket is open
      assert is_port_free(port) is False

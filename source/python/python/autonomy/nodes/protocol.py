from __future__ import annotations
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Mailbox(Protocol):
  async def send(
    self,
    address: str,
    message: str,
    node: Optional[str] = None,
    policy: Optional[str] = None,
  ) -> None: ...

  async def receive(
    self,
    policy: Optional[str] = None,
    timeout: Optional[int] = None,
  ) -> str: ...


@runtime_checkable
class Context(Protocol):
  async def reply(self, message: str): ...


@runtime_checkable
class Worker(Protocol):
  async def handle_message(self, context: Context, message: str): ...


@runtime_checkable
class NodeProtocol(Protocol):
  @property
  def name(self) -> str: ...

  async def identifier(self) -> str: ...

  async def send_and_receive(
    self,
    address: str,
    message: str,
    node: Optional[str] = None,
    policy: Optional[str] = None,
    timeout: Optional[int] = None,
  ) -> str: ...

  async def start_worker(
    self,
    name: str,
    worker: Worker,
    policy: Optional[str] = None,
    exposed_as: Optional[str] = None,
  ): ...

  async def stop_worker(self, name: str): ...

  async def list_agents(self): ...

  async def list_workers(self): ...

  async def list_nodes_priv(self): ...

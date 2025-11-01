import asyncio
import atexit
from os import getenv, environ
from re import fullmatch
from secrets import token_hex
from typing import Awaitable, Callable, Optional, List

from .protocol import Mailbox, Worker
from .remote import NodeController

from ..autonomy_in_rust_for_python import Node as RustNode, Mailbox as RustMailbox
from ..logs import get_logger

logger = get_logger("node.local")

# Global cleanup registry for resource management
_cleanup_functions = []
_cleanup_registered = False


def register_cleanup_function(cleanup_fn):
  """Register a cleanup function to be called when the process exits."""
  global _cleanup_functions, _cleanup_registered
  _cleanup_functions.append(cleanup_fn)

  # Register atexit handler only once
  if not _cleanup_registered:
    atexit.register(_run_cleanup_functions)
    _cleanup_registered = True


def _run_cleanup_functions():
  """Run all registered cleanup functions."""
  for cleanup_fn in _cleanup_functions:
    try:
      cleanup_fn()
    except Exception:
      # Ignore exceptions during cleanup to avoid disrupting shutdown
      pass


class Node:
  @staticmethod
  def start(
    main: Optional[Callable[["Node"], Awaitable[None]]] = None,
    name: Optional[str] = None,
    ticket: Optional[str] = None,
    allow: Optional[str] = None,
    http_server=None,
    wait_until_interrupted: bool = True,
    cache_secure_channels: bool = False,
    use_local_db: bool = False,
    **kwargs,
  ):
    try:
      cluster_name = _pick_cluster_name()
      zone_name = _pick_zone_name()
      node_name = _pick_node_name(cluster_name, zone_name)
      ticket = _pick_ticket(ticket)
      allow = _pick_access_policy(allow, cluster_name)

      if use_local_db or getenv("AUTONOMY_USE_IN_MEMORY_DATABASE"):
        environ.pop("OCKAM_DATABASE_INSTANCE", None)
        environ["OCKAM_SQLITE_IN_MEMORY"] = "true"
        environ["OCKAM_TELEMETRY_EXPORT"] = "false"

      # Check environment variable for wait_until_interrupted
      env_wait = getenv("AUTONOMY_WAIT_UNTIL_INTERRUPTED")
      logger.debug(f"Got AUTONOMY_WAIT_UNTIL_INTERRUPTED={env_wait} from the environment")
      should_wait = wait_until_interrupted
      if env_wait is not None:
        should_wait = env_wait.lower() in ("true", "1", "yes", "on")
        logger.debug(f"Environment variable overriding wait_until_interrupted: {should_wait}")

      if should_wait or main is None:
        main = _wait_until_interrupted_decorator(main)

      async def start_node(rust_node):
        node = Node(rust_node)
        await NodeController(node).start()

        if http_server is None:
          from .. import agents

          asyncio.create_task(agents.HttpServer().start(node))
        else:
          asyncio.create_task(http_server.start(node))

        logger.info(f"Started node '{node_name}'")
        await main(node)

      try:
        logger.info(f"Starting node '{node_name}'")
        RustNode.start(
          start_node,
          name=node_name,
          ticket=ticket,
          allow=allow,
          cache_secure_channels=cache_secure_channels,
          **kwargs,
        )
      except KeyboardInterrupt:
        logger.debug("Detected a KeyboardInterrupt")
        logger.debug(f"Stopping node '{node_name}'")

    except Exception as e:
      logger.error(f"Exception: {e}")
      raise e
    finally:
      # Run cleanup functions before node stops
      _run_cleanup_functions()
      logger.debug(f"Stopped node '{node_name}'")

  def __init__(self, rust_node):
    self.rust = rust_node

  @property
  def name(self) -> str:
    return self.rust.name

  async def identifier(self) -> str:
    return await self.rust.identifier()

  async def create_mailbox(self, address: str, policy: Optional[str] = None) -> RustMailbox:
    return await self.rust.create_mailbox(address, policy)

  async def send(
    self,
    address: str,
    message: str,
    node: Optional[str] = None,
    policy: Optional[str] = None,
  ) -> Mailbox:
    mailbox = await self.create_mailbox(token_hex(12), policy)
    await mailbox.send(address, message, node, policy)
    return mailbox

  async def send_and_receive(
    self,
    address: str,
    message: str,
    node: Optional[str] = None,
    policy: Optional[str] = None,
    timeout: Optional[int] = None,
  ) -> str:
    return await self.rust.send_and_receive(address, message, node, policy, timeout)

  async def start_worker(
    self,
    name: str,
    worker: Worker,
    policy: Optional[str] = None,
    exposed_as: Optional[str] = None,
  ):
    _check_worker_name(name)
    return await self.rust.start_worker(name, worker, policy, exposed_as)

  async def start_internal_worker(
    self,
    name: str,
    worker: Worker,
    policy: Optional[str] = None,
    exposed_as: Optional[str] = None,
  ):
    return await self.rust.start_internal_worker(name, worker, policy, exposed_as)

  async def stop_worker(self, name: str):
    return await self.rust.stop_worker(name)

  async def list_workers(self):
    return await self.rust.list_workers()

  async def list_agents(self):
    return await self.rust.list_agents()

  async def list_nodes_priv(self):
    return await self.rust.list_nodes_priv()

  async def call_mcp_tool(
    self,
    server_name: str,
    tool_name: str,
    tool_args_as_json: Optional[str],
  ) -> str:
    return await self.rust.call_mcp_tool(server_name, tool_name, tool_args_as_json)

  async def list_tools(self) -> List[dict]:
    return await self.rust.list_tools()

  async def mcp_tool_spec(self, server_name: str, tool_name: str) -> str:
    return await self.rust.mcp_tool_spec(server_name, tool_name)

  async def mcp_tools(self) -> str:
    return await self.rust.mcp_tools()

  async def start_spawner(
    self,
    name: str,
    agent_factory: Callable[[], Worker],
    key_extractor: Callable[[str], str],
    policy: Optional[str] = None,
    exposed_as: Optional[str] = None,
  ):
    return await self.rust.start_spawner(name, agent_factory, key_extractor, policy, exposed_as)

  async def stop(self) -> None:
    return await self.rust.stop()

  async def interrupted(self) -> None:
    return await self.rust.interrupted()


def _wait_until_interrupted_decorator(func=None):
  async def wrapper(*args, **kwargs):
    node = args[0]
    if func is not None:
      await func(*args, **kwargs)

    logger.debug("Waiting until interrupted")
    await node.interrupted()

  return wrapper


def _pick_cluster_name() -> str:
  cluster = getenv("CLUSTER")
  logger.debug(f"Got CLUSTER={cluster} from the environment")
  return cluster


def _pick_zone_name() -> str:
  zone = getenv("ZONE")
  logger.debug(f"Got ZONE={zone} from the environment")
  return zone


def _pick_node_name(cluster_name: str, zone_name: str) -> str:
  node_name = getenv("NODE")
  logger.debug(f"Got NODE={node_name} from the environment")

  if node_name:
    name = node_name
  elif cluster_name and zone_name and node_name:
    name = f"{cluster_name}-{zone_name}-{node_name}"
  else:
    name = token_hex(12)

  logger.debug(f"Picked node_name={name}")
  return name


def _pick_ticket(ticket: str | None) -> str:
  env_ticket = getenv("ENROLLMENT_TICKET")
  logger.debug(f"Got ENROLLMENT_TICKET={env_ticket} from the environment")

  if ticket is not None and ticket.strip():
    t = ticket
  elif env_ticket is not None and env_ticket.strip():
    t = env_ticket
  else:
    t = None

  logger.debug(f"Picked ticket={t}")
  return t


def _pick_access_policy(allow, cluster) -> str:
  if allow is None and cluster:
    return f"message.is_local or cluster={cluster}"


def _check_worker_name(name: str):
  if not fullmatch(r"[a-zA-Z0-9_-]+", name):
    raise ValueError(f"Invalid name '{name}'. Only alphanumeric characters, '-' and '_' are allowed")

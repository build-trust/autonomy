import dill
import base64
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
  from .node import Node
  from .protocol import Worker


@dataclass
class NodeRequest:
  method: str
  args: Tuple = ()
  kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeResponse:
  success: bool
  result: Any = None
  error: Optional[str] = None


NODE_CONTROLLER_ADDRESS = "node_controller"


async def list_nodes(node, prefix, filter=None):
  nodes = await node.list_nodes_priv()
  # Strip the relay prefix from the node names
  nodes = [n.removeprefix("forward_to_") for n in nodes]
  # Filter the nodes to only include those in our cluster
  if prefix is not None:
    nodes = [n for n in nodes if n.startswith(prefix)]
  else:
    nodes = []  # Return empty list if no prefix specified
  if filter is not None:
    nodes = [n for n in nodes if filter in n]
  return [RemoteNode(node, n) for n in nodes]


class NodeController:
  def __init__(self, node: "Node"):
    self.node = node

  async def start(self):
    await self.node.start_internal_worker(NODE_CONTROLLER_ADDRESS, self)

  async def handle_message(self, context, message):
    try:
      request = decode_message(message)

      if isinstance(request, NodeRequest):
        handler_name = f"_handle_{request.method}"
        handler = getattr(self, handler_name, None)

        if handler is not None:
          result = await handler(*request.args, **request.kwargs)
          response = NodeResponse(success=True, result=result)
        else:
          response = NodeResponse(success=False, error=f"Unknown method: {request.method}")
      else:
        response = NodeResponse(success=False, error=f"Unknown request type: {type(request)}")

    except Exception as e:
      response = NodeResponse(success=False, error=str(e))

    response = encode_message(response)
    await context.reply(response)

  async def _handle_identifier(self) -> str:
    return await self.node.identifier()

  async def _handle_start_agent(
    self,
    instructions,
    name,
    model,
    memory_model,
    memory_embeddings_model,
    tools,
    planner,
    exposed_as,
    knowledge,
    max_iterations,
  ):
    from ..agents import Agent

    await Agent.start(
      self.node,
      instructions,
      name,
      model,
      memory_model,
      memory_embeddings_model,
      tools,
      planner,
      exposed_as,
      knowledge,
      max_iterations,
    )
    return "ok"

  async def _handle_start_agents(
    self,
    instructions,
    number_of_agents,
    model,
    memory_model,
    memory_embeddings_model,
    tools,
    planner,
    knowledge,
    max_iterations,
  ):
    from ..agents import Agent

    agents = await Agent.start_many(
      self.node,
      instructions,
      number_of_agents,
      model,
      memory_model,
      memory_embeddings_model,
      tools,
      planner,
      knowledge,
      max_iterations,
    )
    return [agent.name for agent in agents]

  async def _handle_start_worker(self, name, worker, policy=None, exposed_as=None):
    worker.node = self.node
    await self.node.start_worker(name, worker, policy, exposed_as)
    return "ok"

  async def _handle_stop_worker(self, name):
    await self.node.stop_worker(name)
    return "ok"

  async def _handle_list_agents(self):
    return await self.node.list_agents()

  async def _handle_list_workers(self):
    return await self.node.list_workers()

  async def _handle_list_nodes_priv(self):
    return await self.node.list_nodes_priv()


def encode_message(message):
  message = dill.dumps(message)
  message = base64.b64encode(message).decode("utf-8")

  return message


def decode_message(message):
  message = base64.b64decode(message)
  message = dill.loads(message)

  return message


class RemoteNode:
  def __init__(self, local_node, remote_name):
    self.local_node = local_node
    self.remote_name = remote_name

  @cached_property
  def logger(self) -> logging.Logger:
    from ..logs.logs import get_logger

    return get_logger("node.remote")

  @property
  def name(self) -> str:
    return self.remote_name

  async def send_and_receive(
    self,
    address: str,
    message: str,
    node: Optional[str] = None,
    policy: Optional[str] = None,
    timeout: Optional[int] = 600,
  ) -> str:
    # Ignore the 'node' parameter, a remote node (self) already represents a specific node
    return await self.local_node.send_and_receive(
      address,
      message,
      node=self.remote_name,
      policy=policy,
      timeout=timeout,
    )

  async def send_request(self, request):
    request = encode_message(request)

    # TODO: Policies
    response = await self.send_and_receive(NODE_CONTROLLER_ADDRESS, request)
    response = decode_message(response)

    if isinstance(response, NodeResponse):
      if not response.success:
        raise RuntimeError(f"Remote node error: {response.error}")
      return response
    else:
      raise RuntimeError(f"Unexpected response type: {type(response)}")

  async def send_generic_request(self, method: str, *args, **kwargs) -> NodeResponse:
    request = NodeRequest(method, args, kwargs)
    return await self.send_request(request)

  async def identifier(self) -> str:
    response = await self.send_generic_request("identifier")
    return response.result

  async def start_worker(
    self,
    name: str,
    worker: "Worker",
    policy: Optional[str] = None,
    exposed_as: Optional[str] = None,
  ):
    from ..helpers.validate_address import validate_address

    validate_address(name)
    await self.send_generic_request("start_worker", name, worker, policy=policy, exposed_as=exposed_as)

  async def stop_worker(self, name: str):
    await self.send_generic_request("stop_worker", name)

  async def start_agent(
    self,
    instructions: str,
    name: str,
    model,
    memory_model,
    memory_embeddings_model,
    tools,
    planner,
    exposed_as,
    knowledge,
    max_iterations,
  ):
    await self.send_generic_request(
      "start_agent",
      instructions=instructions,
      name=name,
      model=model,
      memory_model=memory_model,
      memory_embeddings_model=memory_embeddings_model,
      tools=tools,
      planner=planner,
      exposed_as=exposed_as,
      knowledge=knowledge,
      max_iterations=max_iterations,
    )

  async def start_agents(
    self,
    instructions,
    number_of_agents,
    model,
    memory_model,
    memory_embeddings_model,
    tools,
    planner,
    knowledge,
    max_iterations,
  ):
    response = await self.send_generic_request(
      "start_agents",
      instructions=instructions,
      number_of_agents=number_of_agents,
      model=model,
      memory_model=memory_model,
      memory_embeddings_model=memory_embeddings_model,
      tools=tools,
      planner=planner,
      knowledge=knowledge,
      max_iterations=max_iterations,
    )
    return response.result

  async def list_agents(self):
    response = await self.send_generic_request("list_agents")
    return response.result

  async def list_workers(self):
    response = await self.send_generic_request("list_workers")
    return response.result

  async def list_nodes_priv(self):
    response = await self.send_generic_request("list_nodes_priv")
    return response.result

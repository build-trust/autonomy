import asyncio
import json
import os
import uvicorn
from dataclasses import asdict, is_dataclass
from enum import Enum
from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security.api_key import APIKeyQuery
from fastapi import Security
from typing import Annotated, Optional
from starlette.types import ASGIApp, Receive, Scope, Send

from ..helpers.parse_socket_address import parse_socket_address
from ..nodes.message import (
  AgentReference,
  GetConversationsRequest,
  StreamedConversationSnippet,
  Phase,
  Reference,
  FlowReference,
)
from ..nodes.node import Node
from ..logs.logs import InfoContext, get_logging_config, get_log_levels, set_log_level, set_log_levels, LEVELS
from .agent import get_agent_voice_config, AgentReference
from .voice import VoiceAgent, VoiceConfig

"""
    This class starts an HTTP server allowing a user to interact with a node and its agents.
"""

DEFAULT_HOST = os.environ.get("DEFAULT_HOST_HTTP", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("DEFAULT_PORT_HTTP", "8000"))


class NodeInjectionMiddleware:
  """Raw ASGI middleware to inject node into request state for both HTTP and WebSocket"""

  def __init__(self, app: ASGIApp, http_server: "HttpServer"):
    self.app = app
    self.http_server = http_server

  async def __call__(self, scope: Scope, receive: Receive, send: Send):
    if scope["type"] in ("http", "websocket"):
      scope["state"] = scope.get("state", {})
      scope["state"]["node"] = self.http_server.node
    await self.app(scope, receive, send)


class HttpServer(InfoContext):
  def __init__(self, listen_address=f"{DEFAULT_HOST}:{DEFAULT_PORT}", app: FastAPI = None, api=None):
    from ..logs.logs import get_logger

    self.logger = get_logger("http")
    self.node = None
    self.host = DEFAULT_HOST
    self.port = DEFAULT_PORT
    self.set_host_and_port(listen_address)
    self.api = api

    if not app:
      # Only apply API key validation if API_KEY environment variable is set
      dependencies = [Depends(validate_api_key)] if os.environ.get("API_KEY") else []
      self.app = FastAPI(dependencies=dependencies)
    else:
      self.app = app

    # Add middleware that works for both HTTP and WebSocket
    self.app.add_middleware(NodeInjectionMiddleware, http_server=self)
    self._setup_routes()

  def get_node_from_websocket(self, websocket: WebSocket):
    """Helper method to get node from WebSocket connection"""
    return self.node

  def _setup_routes(self):
    if os.path.exists("index.html"):

      @self.app.get("/")
      async def index():
        return FileResponse("index.html")

    @self.app.get("/agents")
    async def get_agents(node=Depends(self.get_node)):
      with self.info("Retrieving agents", "Retrieved agents"):
        return {"agents": await node.list_agents()}

    @self.app.get("/workers")
    async def get_workers(node=Depends(self.get_node)):
      with self.info("Retrieving workers", "Retrieved workers"):
        return {"workers": await node.list_workers()}

    @self.app.get("/agents/{name}")
    async def get_agent_by_name(name: str, node=Depends(self.get_node)):
      with self.info(f"Retrieving agent with name '{name}'", f"Retrieved agent with name '{name}'"):
        return await find_agent(node, name)

    @self.app.get("/agents/{name}/conversations")
    async def get_conversations_by_agent_name(name: str, node=Depends(self.get_node)):
      with self.info(f"Retrieving conversations for agent '{name}'", f"Retrieved conversations for agent '{name}'"):
        return await get_agent_by_name_and_scope_and_conversation_impl(name, None, None, node)

    @self.app.get("/agents/{name}/scopes/{scope}/conversations")
    async def get_conversations_by_agent_name_and_scope(name: str, scope: str, node=Depends(self.get_node)):
      with self.info(
        f"Retrieving conversations for agent '{name}' and scope '{scope}'",
        f"Retrieved conversations for agent '{name}' and scope '{scope}'",
      ):
        return await get_agent_by_name_and_scope_and_conversation_impl(name, scope, None, node)

    @self.app.get("/agents/{name}/scopes/{scope}/conversations/{conversation}")
    async def get_agent_by_name_and_scope_and_conversation(
      name: str, scope: str, conversation: str, node=Depends(self.get_node)
    ):
      with self.info(
        f"Retrieving conversations for agent '{name}', scope '{scope}' and conversation '{conversation}'",
        f"Retrieved conversations for agent '{name}', scope '{scope}' and conversation '{conversation}'",
      ):
        return await get_agent_by_name_and_scope_and_conversation_impl(name, scope, conversation, node)

    async def get_agent_by_name_and_scope_and_conversation_impl(
      name: str, scope: None | str, conversation: None | str, node
    ):
      # Check if the agent exists
      await find_agent(node, name)

      try:
        agent = AgentReference(name, node)
        query = GetConversationsRequest(scope, conversation)
        response = await agent.send_and_receive_request(query)
        return response
      except Exception as e:
        self.logger.error(f"Failed to get the conversations for agent '{name}': {e}")
        raise HTTPException(status_code=500, detail="Failed to get the conversations for agent '{name}'")

    @self.app.post("/agents/{name}")
    async def send_message_to_agent(
      name: str,
      message: Request,
      stream: bool = False,
      content_size: int = 50,
      timeout: int = 180,
      node=Depends(self.get_node),
    ):
      with self.info(f"Sending a message to agent '{name}'", f"Sent a message to agent '{name}'"):
        await find_agent(node, name)
        return await send_message_to_reference(AgentReference(name, node), message, stream, content_size, timeout)

    @self.app.post("/flows/{name}")
    async def send_message_to_flow(
      name: str,
      message: Request,
      stream: bool = False,
      content_size: int = 50,
      timeout: int = 180,
      node=Depends(self.get_node),
    ):
      with self.info(f"Sending a message to flow '{name}'", f"Sent a message to flow '{name}'"):
        await find_worker(node, name)
        return await send_message_to_reference(FlowReference(name, node), message, stream, content_size, timeout)

    async def find_agent(node, name):
      agents = await node.list_agents()
      agent = next((a for a in agents if a.get("name") == name), None)
      if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
      return agent

    async def find_worker(node, name):
      workers = await node.list_workers()
      worker = next((w for w in workers if w.get("name") == name), None)
      if worker is None:
        raise HTTPException(status_code=404, detail=f"Worker '{name}' not found")
      return worker

    async def find_tool(node, name):
      tools = await node.list_tools()
      tool = next((t for t in tools if t.get("name") == name), None)
      if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
      return tool

    async def send_message_to_reference(
      reference: Reference, message: Request, stream: bool = False, content_size: int = 50, timeout: int = 180
    ):
      try:
        message_json = await message.json()
        if "message" in message_json:
          msg = message_json["message"]
        else:
          msg = message_json
        scope = message_json.get("scope", None)
        conversation = message_json.get("conversation", None)

        if stream:

          async def stream_response():
            received_snippet = None

            async for response in reference.send_stream(msg, scope, conversation, timeout=timeout):
              received = response.snippet
              # even if we make a streaming request, the response might not be streaming if a downstream
              # service does not support streaming
              if type(response) is StreamedConversationSnippet:
                # if phase is missing or None, set "executing" phase
                if not hasattr(received, "phase") or received.phase is None:
                  received.phase = Phase.EXECUTING
                if not received_snippet:
                  received_snippet = received
                else:
                  yield json.dumps(received_snippet, default=default) + "\n"
                  received_snippet = received
                  if response.finished:
                    break
              else:
                yield json.dumps(received, default=default) + "\n"
                break

            # Always yield the final snippet when stream completes
            # This handles both normal completion (finished=True) and agent pause (stream ends early)
            if received_snippet:
              yield json.dumps(received_snippet, default=default) + "\n"

          return StreamingResponse(stream_response(), media_type="application/json")
        else:
          return await reference.send(msg, scope, conversation, timeout=timeout)
      except Exception as e:
        self.logger.error(f"Failed to send message to '{reference.name}': {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")

    @self.app.get("/tools")
    async def get_tools(node=Depends(self.get_node)):
      with self.info("Get the exposed tools", "Retrieved the exposed tools"):
        return node.list_tools()

    @self.app.get("/tools/{name}")
    async def get_tool_by_name(name: str, node=Depends(self.get_node)):
      with self.info(f"Get tool: {name}", f"Retrieved tool: {name}"):
        return find_tool(node, name)

    @self.app.get("/logs/levels")
    async def get_logging_levels():
      """
      Get current log levels for all modules.

      Returns a dictionary of module names to their log levels.
      """
      return {
        "levels": get_log_levels(),
        "available_levels": list(LEVELS.keys()),
      }

    @self.app.post("/logs/levels")
    async def set_logging_levels(request: Request):
      """
      Dynamically change log levels at runtime.

      Request body:
        - level: The log level to set (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - module: Optional module name. If not provided, sets the default level.

      Examples:
        {"level": "DEBUG"} - Set default level to DEBUG
        {"level": "DEBUG", "module": "agent"} - Set agent module to DEBUG
      """
      data = await request.json()
      level = data.get("level")
      module = data.get("module")

      if not level:
        raise HTTPException(status_code=400, detail="'level' is required")

      level_upper = level.upper()
      if level_upper not in [l.upper() for l in LEVELS.keys()]:
        raise HTTPException(
          status_code=400,
          detail=f"Invalid level '{level}'. Must be one of: {list(LEVELS.keys())}",
        )

      if module:
        set_log_level(module, level_upper)
        self.logger.info(f"Log level for '{module}' set to {level_upper}")
      else:
        set_log_levels(level_upper)
        self.logger.info(f"Default log level set to {level_upper}")

      return {
        "status": "ok",
        "module": module or "default",
        "level": level_upper,
        "levels": get_log_levels(),
      }

    @self.app.websocket("/agents/{name}/voice")
    async def voice_endpoint(
      websocket: WebSocket,
      name: str,
      scope: Optional[str] = None,
      conversation: Optional[str] = None,
    ):
      """
      WebSocket endpoint for real-time voice conversations with an agent.

      Uses the Voice Interface pattern:
      - VoiceAgent handles audio I/O via OpenAI Realtime API (fast, low latency)
      - Delegates complex tasks to primary agent (the main agent with tools)

      Query Parameters:
          scope: Optional scope for memory isolation (e.g., user ID)
          conversation: Optional conversation ID for memory isolation
      """
      self.logger.info(f"🔌 Voice connection for agent '{name}' (scope={scope}, conversation={conversation})")

      node = self.get_node_from_websocket(websocket)
      voice_session = None

      try:
        await websocket.accept()
      except Exception as e:
        self.logger.error(f"❌ Failed to accept WebSocket: {e}")
        raise

      try:
        # Verify agent exists
        await find_agent(node, name)

        # Get voice config
        voice_config = get_agent_voice_config(name)
        if voice_config is None:
          await websocket.send_json(
            {
              "type": "error",
              "error": f"Agent '{name}' does not have voice enabled. Add voice=VoiceConfig() when starting the agent.",
            }
          )
          return

        # Create primary agent reference and voice agent
        primary = AgentReference(name, node)
        voice_agent = VoiceAgent(
          name=f"{name}-voice",
          primary=primary,
          config=voice_config,
        )

        # Create session and run bidirectional communication
        voice_session = await voice_agent.create_session(scope=scope, conversation=conversation)
        self.logger.info(f"✅ VoiceSession created for agent '{name}'")

        await voice_session.run_with_client(
          send_json=websocket.send_json,
          receive_text=websocket.receive_text,
          voice_config=voice_config,
          agent_name=name,
        )

      except Exception as e:
        self.logger.error(f"❌ Voice error for '{name}': {e}", exc_info=True)
        try:
          await websocket.send_json({"type": "error", "error": str(e)})
        except:
          pass

      finally:
        if voice_session:
          await voice_session.close()
          self.logger.info(f"🔌 Voice session closed for agent '{name}'")

        try:
          await websocket.close()
        except:
          pass

  def get_node(self):
    """Get node instance. Can be used as a dependency in routes."""
    return self.node

  async def serve(self):
    self.logger.info(f"Starting http server at {self.host}:{self.port}")
    config = uvicorn.Config(self.app, host=self.host, port=self.port, log_config=get_logging_config(), access_log=True)
    server = uvicorn.Server(config)
    self.logger.info(f"Started http server at {self.host}:{self.port}")
    await server.serve()

  def set_host_and_port(self, listen_address):
    try:
      host, port = parse_socket_address(listen_address)
      self.host = host
      self.port = port
    except ValueError:
      raise ValueError(f"Invalid listen_address: {listen_address}")

  async def start(self, node):
    self.node = node
    # mount some additional if provided custom routes
    if self.api:
      self.api.routes(self.node)
      self.app.mount("/", self.api.api)

    asyncio.create_task(self.serve())


# Retrieve the node from the HTTP request state
def get_node(request: Request) -> Node:
  # The state is populated by our middleware
  return request.state.node


# Retrieve the node from WebSocket state
def get_node_ws(websocket: WebSocket) -> Node:
  # The state is populated by our middleware
  return websocket.state.node


# Node as a dependency for HTTP endpoints
NodeDep = Annotated[Node, Depends(get_node)]

# Node as a dependency for WebSocket endpoints
WebSocketNodeDep = Annotated[Node, Depends(get_node_ws)]


def default(obj):
  if is_dataclass(obj):
    return asdict(obj)
  if isinstance(obj, Enum):
    return obj.value
  raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def validate_api_key(api_key: str = Security(APIKeyQuery(name="api_key", auto_error=False))) -> str:
  """
  Validate API key for HTTP requests.

  Note: This is only applied when the API_KEY environment variable is set.
  WebSocket routes are not affected by this global dependency.
  """
  expected = os.environ.get("API_KEY")
  if not expected:
    # No API_KEY configured, allow access
    return None
  if api_key != expected:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
  return expected

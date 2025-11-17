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
from typing import Annotated
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
from ..logs.logs import InfoContext, get_logging_config
from ..models.voice_model import VoiceModel

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
      self.app = FastAPI(dependencies=[Depends(validate_api_key)])
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
      timeout: int = 60,
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
      timeout: int = 60,
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
      reference: Reference, message: Request, stream: bool = False, content_size: int = 50, timeout: int = 60
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

    @self.app.websocket("/agents/{name}/voice")
    async def voice_endpoint(websocket: WebSocket, name: str):
      """WebSocket endpoint for real-time voice conversations with an agent"""
      self.logger.info(f"🔌 WebSocket voice connection attempt for agent '{name}'")
      self.logger.debug(f"   Client: {websocket.client}")

      # Get node from WebSocket state (injected by middleware)
      node = self.get_node_from_websocket(websocket)

      try:
        # Accept WebSocket connection
        await websocket.accept()
        self.logger.info(f"✅ WebSocket connection accepted for agent '{name}'")
      except Exception as e:
        self.logger.error(f"❌ Failed to accept WebSocket connection: {e}")
        raise

      voice_session = None

      try:
        # Verify agent exists
        await find_agent(node, name)

        # TODO: Get agent's VoiceModel configuration
        # For now, use default configuration
        voice_model = VoiceModel(
          "gpt-4o-realtime-preview",
          voice="echo",
          instructions="You are a helpful AI assistant having a voice conversation."
        )

        # Create voice session
        voice_session = await voice_model.create_session()
        self.logger.info(f"✅ Voice session created for agent '{name}'")

        # Send connection confirmation to client
        await websocket.send_json({
          "type": "connected",
          "message": f"Connected to agent '{name}' voice interface"
        })

        # Task to receive audio from client and forward to voice model
        async def receive_from_client():
          try:
            while True:
              message = await websocket.receive_text()
              data = json.loads(message)

              if data.get("type") == "audio":
                # Client sent audio chunk - forward to voice model
                audio_base64 = data.get("audio", "")
                if audio_base64:
                  import base64
                  audio_bytes = base64.b64decode(audio_base64)
                  await voice_session.send_audio(audio_bytes)

              elif data.get("type") == "event":
                # Client sent custom event - forward to voice model
                event = data.get("event", {})
                if event:
                  await voice_session.send_event(event)

              elif data.get("type") == "close":
                self.logger.info(f"📪 Client requested close for agent '{name}'")
                break

          except WebSocketDisconnect:
            self.logger.info(f"📪 Client disconnected from agent '{name}'")
          except Exception as e:
            self.logger.error(f"❌ Error receiving from client: {e}")

        # Task to receive events from voice model and forward to client
        async def receive_from_voice_model():
          try:
            async for event in voice_session.receive_events():
              event_type = event.get("type")

              # Log non-audio events for debugging
              if event_type not in ["response.audio.delta", "response.audio_transcript.delta"]:
                self.logger.debug(f"📨 Voice model event: {event_type}")

              # Forward relevant events to client
              if event_type == "response.audio.delta":
                # Audio chunk from assistant
                audio_base64 = event.get("delta", "")
                if audio_base64:
                  await websocket.send_json({
                    "type": "audio_chunk",
                    "audio": audio_base64
                  })

              elif event_type == "response.audio_transcript.delta":
                # Transcript of what assistant is saying
                transcript_delta = event.get("delta", "")
                if transcript_delta:
                  await websocket.send_json({
                    "type": "transcript",
                    "text": transcript_delta,
                    "role": "assistant"
                  })

              elif event_type == "conversation.item.input_audio_transcription.completed":
                # User's speech transcription
                user_transcript = event.get("transcript", "")
                if user_transcript:
                  await websocket.send_json({
                    "type": "user_transcript",
                    "text": user_transcript
                  })
                  self.logger.info(f"🗣️  User said: {user_transcript}")

              elif event_type == "input_audio_buffer.speech_started":
                await websocket.send_json({
                  "type": "speech_started"
                })
                self.logger.debug("🎤 Speech started")

              elif event_type == "input_audio_buffer.speech_stopped":
                await websocket.send_json({
                  "type": "speech_stopped"
                })
                self.logger.debug("⏸️  Speech stopped (VAD detected)")

              elif event_type == "response.done":
                await websocket.send_json({
                  "type": "response_complete"
                })
                self.logger.debug("✅ Response complete")

              elif event_type == "error":
                error = event.get("error", {})
                await websocket.send_json({
                  "type": "error",
                  "error": error.get("message", "Unknown error")
                })
                self.logger.error(f"❌ Voice model error: {error.get('message', 'Unknown')}")

              elif event_type == "session.created":
                self.logger.debug("✅ Voice model session created")

              elif event_type == "session.updated":
                self.logger.debug("✅ Voice model session updated")

          except Exception as e:
            self.logger.error(f"❌ Error receiving from voice model: {e}")

        # Run both tasks concurrently
        await asyncio.gather(
          receive_from_client(),
          receive_from_voice_model(),
          return_exceptions=True
        )

      except Exception as e:
        self.logger.error(f"❌ Voice WebSocket error for agent '{name}': {e}")
        try:
          await websocket.send_json({
            "type": "error",
            "error": str(e)
          })
        except:
          pass

      finally:
        # Clean up voice session
        if voice_session:
          await voice_session.close()
          self.logger.info(f"🔌 Voice session closed for agent '{name}'")

        # Close client WebSocket
        try:
          await websocket.close()
        except:
          pass

        self.logger.info(f"🔌 Voice WebSocket connection closed for agent '{name}'")

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
  expected = os.environ.get("API_KEY")
  if api_key != expected:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key header")
  return expected

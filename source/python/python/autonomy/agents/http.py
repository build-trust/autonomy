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
from ..models.voice import Voice
from ..models.voice_model import VoiceModel
from .agent import get_agent_voice, get_agent_voice_model, AgentReference
from ..voice.stt import create_stt_provider
from ..voice.tts import create_tts_provider
from ..voice.vad import create_vad

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
      """
      WebSocket endpoint for real-time voice conversations with an agent.

      Architecture B: Voice as Agent I/O
      Audio Input → STT → Agent (main model) → TTS → Audio Output

      This endpoint shares conversation state with text endpoints,
      enabling seamless switching between voice and text modalities.
      """
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

      # Architecture B components
      stt_provider = None
      tts_provider = None
      vad = None
      voice_session = None  # Legacy VoiceModel session

      try:
        # Verify agent exists
        await find_agent(node, name)

        # Get agent's Voice configuration
        voice_config = get_agent_voice(name)
        voice_model = get_agent_voice_model(name)

        # Determine which architecture to use
        use_architecture_b = voice_config is not None

        if use_architecture_b:
          # ============================================================
          # ARCHITECTURE B: STT → Agent → TTS
          # ============================================================
          self.logger.info(f"🎯 Using Architecture B (Voice as Agent I/O)")

          # Create STT provider
          stt_provider = create_stt_provider(
            provider=voice_config.stt_provider, model=voice_config.stt_model, sample_rate=voice_config.sample_rate
          )
          self.logger.info(f"✅ STT provider created: {voice_config.stt_provider}")

          # Create TTS provider
          tts_provider = create_tts_provider(provider=voice_config.tts_provider, model=voice_config.tts_model)
          self.logger.info(f"✅ TTS provider created: {voice_config.tts_provider}")

          # Create VAD
          if voice_config.vad_enabled:
            vad = create_vad(
              method=voice_config.vad_method,
              threshold=voice_config.vad_threshold,
              silence_duration_ms=voice_config.vad_silence_duration_ms,
              sample_rate=voice_config.sample_rate,
            )
            self.logger.info(f"✅ VAD created: method={voice_config.vad_method}")

          # Create agent reference
          agent = AgentReference(name, node)

          # Track conversation scope (shared with text endpoint)
          scope = f"voice-{websocket.client.host}" if websocket.client else "voice-default"
          conversation = None  # Will be set by client or auto-generated

          # Send connection confirmation
          await websocket.send_json(
            {
              "type": "connected",
              "message": f"Connected to agent '{name}' voice interface (Architecture B)",
              "config": {
                "stt_provider": voice_config.stt_provider,
                "tts_provider": voice_config.tts_provider,
                "sample_rate": voice_config.sample_rate,
                "audio_format": voice_config.audio_format,
              },
            }
          )

          # Task to receive audio from client, transcribe, and send to agent
          async def receive_from_client():
            nonlocal conversation, scope
            try:
              while True:
                message = await websocket.receive_text()
                data = json.loads(message)

                if data.get("type") == "config":
                  # Client sending configuration
                  scope_override = data.get("scope")
                  if scope_override:
                    scope = scope_override
                  conversation_override = data.get("conversation")
                  if conversation_override:
                    conversation = conversation_override
                  self.logger.info(f"📝 Config: scope={scope}, conversation={conversation}")

                elif data.get("type") == "audio":
                  # Client sent audio chunk
                  audio_base64 = data.get("audio", "")
                  if audio_base64:
                    import base64

                    audio_bytes = base64.b64decode(audio_base64)
                    self.logger.debug(f"📥 Received audio chunk: {len(audio_bytes)} bytes")

                    if vad:
                      # Process with VAD
                      is_complete, complete_audio = await vad.process_audio(audio_bytes)
                      self.logger.debug(
                        f"🔊 VAD: is_complete={is_complete}, audio_len={len(complete_audio) if complete_audio else 0}"
                      )

                      if is_complete and complete_audio:
                        # Utterance complete - transcribe and send to agent
                        self.logger.info(f"🎤 Voice activity detected (utterance complete)")

                        # Transcribe
                        self.logger.debug(f"🎯 Starting STT transcription...")
                        stt_result = await stt_provider.transcribe(complete_audio, voice_config.language)
                        self.logger.debug(f"📝 STT raw result: {stt_result if stt_result else '(empty)'}")

                        # Extract text from result (handle both string and dict responses)
                        if isinstance(stt_result, dict):
                          text = stt_result.get("text", "")
                        elif isinstance(stt_result, str):
                          text = stt_result
                        else:
                          text = str(stt_result) if stt_result else ""

                        self.logger.debug(f"📝 Extracted text: {text if text else '(empty)'}")

                        if text:
                          # Send transcript to client
                          await websocket.send_json({"type": "transcript", "text": text, "role": "user"})
                          self.logger.info(f"🗣️  User said: {text}")

                          # Send to agent
                          await websocket.send_json({"type": "agent_thinking"})

                          # Stream response from agent
                          # Accumulate all text chunks
                          response_text = ""
                          chunk_count = 0
                          async for response_chunk in agent.send_stream(
                            message=text, scope=scope, conversation=conversation
                          ):
                            chunk_count += 1

                            # Extract response text from chunk
                            if hasattr(response_chunk, "snippet"):
                              for msg in response_chunk.snippet.messages:
                                if msg.role.value == "assistant" and hasattr(msg.content, "text"):
                                  chunk_text = msg.content.text
                                  self.logger.debug(
                                    f"📦 Chunk #{chunk_count}: text='{chunk_text}' (len={len(chunk_text)})"
                                  )
                                  # Accumulate text from each chunk
                                  if chunk_text:
                                    response_text += chunk_text

                          self.logger.info(
                            f"📊 Stream complete: {chunk_count} chunks, final response length={len(response_text)}"
                          )

                          if not response_text:
                            self.logger.warning("No assistant response found in stream")
                            continue

                          self.logger.info(f"🤖 Agent response: {response_text[:100]}...")

                          # Convert to speech
                          self.logger.debug(f"🔊 Starting TTS synthesis...")
                          audio_chunk_count = 0
                          async for audio_chunk in tts_provider.synthesize_stream(
                            text=response_text, voice=voice_config.voice
                          ):
                            # Send audio to client
                            audio_b64 = base64.b64encode(audio_chunk).decode()
                            await websocket.send_json({"type": "audio", "audio": audio_b64})
                            audio_chunk_count += 1
                            self.logger.debug(f"📤 Sent audio chunk #{audio_chunk_count} ({len(audio_chunk)} bytes)")

                          # Mark response complete
                          self.logger.info(f"✅ Response complete ({audio_chunk_count} audio chunks sent)")
                          await websocket.send_json({"type": "response_complete"})
                    else:
                      # No VAD - transcribe immediately (not recommended)
                      text = await stt_provider.transcribe(audio_bytes, voice_config.language)
                      if text:
                        await websocket.send_json({"type": "transcript", "text": text, "role": "user"})

                elif data.get("type") == "close":
                  self.logger.info(f"📪 Client requested close for agent '{name}'")
                  break

            except WebSocketDisconnect:
              self.logger.info(f"📪 Client disconnected from agent '{name}'")
            except Exception as e:
              self.logger.error(f"❌ Error in Architecture B receive: {e}", exc_info=True)

          # Run the receive task
          await receive_from_client()

        else:
          # ============================================================
          # ARCHITECTURE A (LEGACY): OpenAI Realtime API
          # ============================================================
          self.logger.info(f"🔄 Using Architecture A (Legacy VoiceModel)")

          # Use VoiceModel or create default
          if voice_model is None:
            voice_model = VoiceModel(
              "gpt-4o-realtime-preview",
              voice="echo",
              instructions="You are a helpful AI assistant having a voice conversation.",
            )

          # Create voice session
          voice_session = await voice_model.create_session()
          self.logger.info(f"✅ Voice session created for agent '{name}'")

          # Send connection confirmation to client
          await websocket.send_json(
            {"type": "connected", "message": f"Connected to agent '{name}' voice interface (Architecture A)"}
          )

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
                    await websocket.send_json({"type": "audio_chunk", "audio": audio_base64})

                elif event_type == "response.audio_transcript.delta":
                  # Transcript of what assistant is saying
                  transcript_delta = event.get("delta", "")
                  if transcript_delta:
                    await websocket.send_json({"type": "transcript", "text": transcript_delta, "role": "assistant"})

                elif event_type == "conversation.item.input_audio_transcription.completed":
                  # User's speech transcription
                  user_transcript = event.get("transcript", "")
                  if user_transcript:
                    await websocket.send_json({"type": "user_transcript", "text": user_transcript})
                    self.logger.info(f"🗣️  User said: {user_transcript}")

                elif event_type == "input_audio_buffer.speech_started":
                  await websocket.send_json({"type": "speech_started"})
                  self.logger.debug("🎤 Speech started")

                elif event_type == "input_audio_buffer.speech_stopped":
                  await websocket.send_json({"type": "speech_stopped"})
                  self.logger.debug("⏸️  Speech stopped (VAD detected)")

                elif event_type == "response.done":
                  await websocket.send_json({"type": "response_complete"})
                  self.logger.debug("✅ Response complete")

                elif event_type == "error":
                  error = event.get("error", {})
                  await websocket.send_json({"type": "error", "error": error.get("message", "Unknown error")})
                  self.logger.error(f"❌ Voice model error: {error.get('message', 'Unknown')}")

                elif event_type == "session.created":
                  self.logger.debug("✅ Voice model session created")

                elif event_type == "session.updated":
                  self.logger.debug("✅ Voice model session updated")

            except Exception as e:
              self.logger.error(f"❌ Error receiving from voice model: {e}")

          # Run both tasks concurrently
          await asyncio.gather(receive_from_client(), receive_from_voice_model(), return_exceptions=True)

      except Exception as e:
        self.logger.error(f"❌ Voice WebSocket error for agent '{name}': {e}", exc_info=True)
        try:
          await websocket.send_json({"type": "error", "error": str(e)})
        except:
          pass

      finally:
        # Clean up voice session (Architecture A)
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

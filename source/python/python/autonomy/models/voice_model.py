from typing import Optional, AsyncGenerator, Dict, Any
import websockets
import json
import base64
import os
from ..logs import get_logger

logger = get_logger("voice_model")


class VoiceModel:
  """
  VoiceModel for real-time bidirectional voice communication.

  Supports:
  - OpenAI Realtime API (gpt-4o-realtime-preview)
  - Server-side Voice Activity Detection (VAD)
  - Streaming audio input/output
  - Real-time transcription
  - Low-latency bidirectional conversations

  This class connects to the LiteLLM proxy's WebSocket endpoint for
  realtime voice models, enabling load balancing, failover, and
  centralized logging.

  Example:
      voice_model = VoiceModel(
          "gpt-4o-realtime-preview",
          voice="echo",
          instructions="You are a helpful assistant"
      )
      session = await voice_model.create_session()
      await session.send_audio(audio_bytes)
      async for event in session.receive_events():
          if event["type"] == "response.audio.delta":
              audio_chunk = base64.b64decode(event["delta"])
              # Play audio chunk
  """

  def __init__(
    self,
    name: str,
    voice: str = "echo",
    instructions: Optional[str] = None,
    vad_config: Optional[Dict[str, Any]] = None,
    input_audio_format: str = "pcm16",
    output_audio_format: str = "pcm16",
    temperature: float = 0.8,
    max_response_output_tokens: int = 4096,
    **kwargs
  ):
    """
    Initialize a VoiceModel instance.

    :param name: Model name (e.g., 'gpt-4o-realtime-preview')
    :param voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
    :param instructions: System instructions for the model
    :param vad_config: Voice Activity Detection configuration
    :param input_audio_format: Audio format for input (pcm16, g711_ulaw, g711_alaw)
    :param output_audio_format: Audio format for output (pcm16, g711_ulaw, g711_alaw)
    :param temperature: Sampling temperature (0.6 to 1.2)
    :param max_response_output_tokens: Maximum tokens in response
    :param kwargs: Additional parameters
    """
    self.name = name
    self.voice = voice
    self.instructions = instructions or "You are a helpful and friendly AI assistant."
    self.input_audio_format = input_audio_format
    self.output_audio_format = output_audio_format
    self.temperature = temperature
    self.max_response_output_tokens = max_response_output_tokens
    self.kwargs = kwargs

    # Default VAD configuration
    self.vad_config = vad_config or {
      "type": "server_vad",
      "threshold": 0.5,
      "prefix_padding_ms": 300,
      "silence_duration_ms": 500
    }

    # Session configuration
    self.session_config = {
      "modalities": ["text", "audio"],
      "instructions": self.instructions,
      "voice": self.voice,
      "input_audio_format": self.input_audio_format,
      "output_audio_format": self.output_audio_format,
      "input_audio_transcription": {"model": "whisper-1"},
      "turn_detection": self.vad_config,
      "temperature": self.temperature,
      "max_response_output_tokens": self.max_response_output_tokens,
      **kwargs
    }

    logger.info(
      f"VoiceModel initialized: model={name}, voice={voice}, "
      f"vad_type={self.vad_config.get('type')}"
    )

  async def create_session(self) -> "VoiceSession":
    """
    Create a new voice session with WebSocket connection.

    Returns a VoiceSession object for bidirectional communication.
    The session must be connected before use.

    :return: VoiceSession instance
    """
    session = VoiceSession(self)
    await session.connect()
    return session

  def _get_websocket_url(self) -> str:
    """
    Get the WebSocket URL for the realtime API.

    Checks for LiteLLM proxy configuration first, falls back to direct OpenAI.

    :return: WebSocket URL
    """
    # Check for LiteLLM proxy
    proxy_base = os.environ.get("LITELLM_PROXY_API_BASE")
    if proxy_base:
      # Convert HTTP(S) to WS(S)
      ws_base = proxy_base.replace("https://", "wss://").replace("http://", "ws://")
      url = f"{ws_base}/v1/realtime?model={self.name}"
      logger.debug(f"Using LiteLLM proxy WebSocket: {url}")
      return url

    # Fallback to direct OpenAI
    url = f"wss://api.openai.com/v1/realtime?model={self.name}"
    logger.debug(f"Using direct OpenAI WebSocket: {url}")
    return url

  def _get_auth_headers(self) -> Dict[str, str]:
    """
    Get authentication headers for WebSocket connection.

    :return: Dictionary of headers
    """
    headers = {}

    # Check for LiteLLM proxy API key
    proxy_key = os.environ.get("LITELLM_PROXY_API_KEY")
    if proxy_key:
      # LiteLLM proxy expects both api-key and Authorization headers
      headers["api-key"] = proxy_key
      headers["Authorization"] = f"Bearer {proxy_key}"
      logger.debug("Using LiteLLM proxy API key")
      return headers

    # Fallback to OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
      headers["Authorization"] = f"Bearer {openai_key}"
      headers["OpenAI-Beta"] = "realtime=v1"
      logger.debug("Using OpenAI API key")
      return headers

    logger.warning("No API key found in environment")
    return headers


class VoiceSession:
  """
  Manages a real-time voice conversation session.

  Handles WebSocket connection lifecycle, audio streaming,
  and event forwarding between client and model.
  """

  def __init__(self, model: VoiceModel):
    """
    Initialize a VoiceSession.

    :param model: VoiceModel instance
    """
    self.model = model
    self.websocket: Optional[websockets.WebSocketClientProtocol] = None
    self.connected = False
    logger.debug(f"VoiceSession created for model: {model.name}")

  async def connect(self):
    """
    Establish WebSocket connection to the realtime API.

    Sends session configuration after connection is established.

    :raises RuntimeError: If connection fails
    """
    url = self.model._get_websocket_url()
    headers = self.model._get_auth_headers()

    try:
      logger.info(f"Connecting to realtime API: {url}")

      self.websocket = await websockets.connect(
        url,
        additional_headers=headers
      )

      self.connected = True
      logger.info("✅ WebSocket connection established")

      # Send session configuration
      await self.websocket.send(json.dumps({
        "type": "session.update",
        "session": self.model.session_config
      }))

      logger.info("✅ Session configuration sent")

    except Exception as e:
      self.connected = False
      logger.error(f"❌ Failed to connect to realtime API: {e}")
      raise RuntimeError(f"Failed to establish WebSocket connection: {e}")

  async def send_audio(self, audio_chunk: bytes):
    """
    Send audio chunk to the model.

    Audio is base64-encoded before sending.

    :param audio_chunk: Raw audio bytes (PCM16 format)
    :raises RuntimeError: If session is not connected
    """
    if not self.connected or not self.websocket:
      raise RuntimeError("Session not connected. Call connect() first.")

    audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')

    await self.websocket.send(json.dumps({
      "type": "input_audio_buffer.append",
      "audio": audio_base64
    }))

    logger.debug(f"Sent audio chunk: {len(audio_chunk)} bytes")

  async def send_event(self, event: Dict[str, Any]):
    """
    Send a custom event to the model.

    :param event: Event dictionary (must have "type" field)
    :raises RuntimeError: If session is not connected
    """
    if not self.connected or not self.websocket:
      raise RuntimeError("Session not connected. Call connect() first.")

    await self.websocket.send(json.dumps(event))
    logger.debug(f"Sent event: {event.get('type')}")

  async def receive_events(self) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Receive events from the model as they arrive.

    Yields events as dictionaries. Events include:
    - response.audio.delta: Audio chunk from model
    - response.audio_transcript.delta: Transcript of model response
    - input_audio_buffer.speech_started: User started speaking
    - input_audio_buffer.speech_stopped: User stopped speaking
    - conversation.item.input_audio_transcription.completed: User transcript ready
    - And many more...

    :yield: Event dictionaries
    :raises RuntimeError: If session is not connected
    """
    if not self.connected or not self.websocket:
      raise RuntimeError("Session not connected. Call connect() first.")

    try:
      async for message in self.websocket:
        event = json.loads(message)
        event_type = event.get("type", "unknown")

        # Log non-audio events for debugging
        if event_type not in ["response.audio.delta", "response.audio_transcript.delta"]:
          logger.debug(f"📨 Received event: {event_type}")

        yield event

    except websockets.exceptions.ConnectionClosed:
      logger.info("WebSocket connection closed")
      self.connected = False
    except Exception as e:
      logger.error(f"Error receiving events: {e}")
      self.connected = False
      raise

  async def close(self):
    """
    Close the WebSocket connection and clean up resources.
    """
    if self.websocket:
      try:
        await self.websocket.close()
        logger.info("WebSocket connection closed")
      except Exception as e:
        logger.warning(f"Error closing WebSocket: {e}")
      finally:
        self.websocket = None
        self.connected = False

  async def __aenter__(self):
    """Context manager entry - connection already established in create_session"""
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit - close connection"""
    await self.close()

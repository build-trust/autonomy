"""
Voice Agent for the Voice Interface pattern.

This module provides VoiceConfig, VoiceAgent, and VoiceSession classes for
real-time voice interactions using the Voice Interface pattern. The VoiceAgent
handles user interaction via OpenAI Realtime API with low latency, delegating
complex tasks to the primary agent.

Architecture:
    WebSocket Client ←→ VoiceSession ←→ OpenAI Realtime API
                              ↓
                        VoiceAgent
                              ↓ (delegate_to_primary)
                       Primary Agent (main agent with tools)

Example:
    # VoiceAgent is created automatically when an Agent has voice config
    agent = await Agent.start(
        name="customer-support",
        instructions="You are a customer service agent...",
        model=Model("gpt-4.1"),
        tools=[lookup_account, find_store],
        voice=VoiceConfig(
            realtime_model="gpt-4o-realtime-preview",
            voice="nova",
        ),
    )

    # Voice sessions are created by the HTTP server for each WebSocket connection
    voice_agent = VoiceAgent(name="support-voice", primary=agent, config=voice_config)
    session = await voice_agent.create_session()
"""

import asyncio
import json
import base64
import os
import traceback
from typing import Optional, Dict, Any, AsyncGenerator, List, TYPE_CHECKING, TypedDict, NotRequired, Callable, Awaitable

try:
  import websockets
  from websockets.client import WebSocketClientProtocol
except ImportError:
  websockets = None
  WebSocketClientProtocol = None

from ..logs import get_logger
from ..tools.protocol import InvokableTool

if TYPE_CHECKING:
  from .agent import Agent
  from ..nodes.message import AgentReference

logger = get_logger("voice_agent")


# =============================================================================
# VOICE CONFIGURATION
# =============================================================================


class VoiceConfig(TypedDict, total=False):
  """
  Configuration for Voice Interface pattern.

  The voice agent handles user interaction via OpenAI Realtime API,
  delegating complex tasks to the primary agent (the main agent).

  All fields are optional and have sensible defaults.

  Architecture:
      User ←→ Voice Agent (fast, low latency)
                    ↓ (complex tasks)
             Primary Agent (powerful, has tools)

  Example:
      # Minimal - all defaults
      agent = await Agent.start(
          name="customer-support",
          instructions="You are a customer service agent...",
          model=Model("claude-sonnet-4-v1"),
          voice={},  # Uses all defaults
      )

      # Or with custom settings
      agent = await Agent.start(
          name="customer-support",
          instructions="You are a customer service agent...",
          model=Model("claude-sonnet-4-v1"),
          voice={
              "realtime_model": "gpt-4o-realtime-preview",
              "voice": "nova",
              "allowed_actions": ["greetings", "collect_phone_number"],
          },
      )

  Attributes:
      realtime_model: Model for the voice agent (must support realtime API)
      voice: TTS voice ID (alloy, echo, fable, onyx, nova, shimmer)
      allowed_actions: Actions the voice agent can handle without primary agent
      instructions: Custom voice agent instructions (auto-generated if None)
      filler_phrases: Phrases to say before delegating to primary agent
      input_audio_format: Audio format for input (pcm16, g711_ulaw, g711_alaw)
      output_audio_format: Audio format for output (pcm16, g711_ulaw, g711_alaw)
      vad_threshold: VAD sensitivity (0.0-1.0)
      vad_prefix_padding_ms: Audio to include before speech detection
      vad_silence_duration_ms: Silence duration to detect end of speech
  """

  realtime_model: str
  voice: str
  allowed_actions: List[str]
  instructions: Optional[str]
  filler_phrases: List[str]
  input_audio_format: str
  output_audio_format: str
  vad_threshold: float
  vad_prefix_padding_ms: int
  vad_silence_duration_ms: int


# Default values for VoiceConfig
VOICE_CONFIG_DEFAULTS: VoiceConfig = {
  "realtime_model": "gpt-4o-realtime-preview",
  "voice": "echo",
  "allowed_actions": [
    "greetings",
    "chitchat",
    "collect_information",
    "clarifications",
  ],
  "instructions": None,
  "filler_phrases": [
    "Just a second.",
    "Let me check.",
    "One moment.",
    "Let me look into that.",
    "Give me a moment.",
    "Let me see.",
  ],
  "input_audio_format": "pcm16",
  "output_audio_format": "pcm16",
  "vad_threshold": 0.5,
  "vad_prefix_padding_ms": 300,
  "vad_silence_duration_ms": 500,
}


def get_voice_config_value(config: VoiceConfig, key: str) -> Any:
  """Get a value from VoiceConfig with default fallback."""
  return config.get(key, VOICE_CONFIG_DEFAULTS.get(key))


# =============================================================================
# DELEGATE TO PRIMARY TOOL
# =============================================================================


class DelegateToPrimaryTool(InvokableTool):
  """
  Tool for Voice Agent to delegate to the Primary Agent.

  The Voice Agent calls this tool when it needs:
  - To answer factual questions
  - To call tools (database lookups, APIs, etc.)
  - To handle complex requests beyond basic chitchat

  The primary agent receives the full conversation history
  and relevant context, processes the request (potentially
  calling its own tools), and returns a high-quality response
  that the voice agent speaks verbatim.

  Example usage by voice agent:
    User: "What's my account balance?"
    Voice Agent: "Let me check on that."
    Tool call: delegate_to_primary(context="User wants account balance")
    → Primary agent processes request, calls tools
    → Returns: "Your current balance is $142.50"
    Voice Agent speaks: "Your current balance is $142.50"

  Note: This tool works with AgentReference (message passing) rather than
  direct Agent instances, making it compatible with the spawner pattern.
  """

  def __init__(
    self,
    primary: "AgentReference",
    conversation_history: List[Dict],
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
  ):
    """
    Initialize DelegateToPrimaryTool.

    Args:
        primary: AgentReference to the primary agent
        conversation_history: Shared list tracking conversation for context
        scope: Optional scope for memory isolation (e.g., user ID)
        conversation: Optional conversation ID for memory isolation
    """
    super().__init__()
    self.name = "delegate_to_primary"
    self.primary = primary
    self.conversation_history = conversation_history
    self.scope = scope
    self.conversation = conversation
    self.description = (
      "Use this tool to get a response from the primary agent. "
      "The primary agent is more intelligent and has access to tools. "
      "Use for ANY request beyond basic greetings or chitchat. "
      "ALWAYS say a filler phrase to the user before calling this tool."
    )

  async def invoke(self, json_argument: Optional[str]) -> Dict[str, Any]:
    """
    Delegate to primary agent and return response.

    Args:
        json_argument: JSON with 'context' key containing relevant info from last user message

    Returns:
        Dict with primary agent's response and speak_verbatim flag
    """
    # Parse arguments
    if json_argument is None or json_argument.strip() == "":
      params = {}
    else:
      try:
        params = json.loads(json_argument)
        if not isinstance(params, dict):
          params = {}
      except json.JSONDecodeError:
        params = {}

    context = params.get("context", "")

    logger.info(f"[DELEGATE] Delegating to primary agent with context: {context[:100]}...")

    try:
      # Build primary agent prompt with conversation history
      response = await self._call_primary(context)

      logger.info(f"[DELEGATE] Primary agent response: {response[:100]}...")

      return {
        "response": response,
        "speak_verbatim": True,
      }

    except Exception as e:
      logger.error(f"[DELEGATE] Error calling primary agent: {e}")
      logger.debug(f"[DELEGATE] Traceback: {traceback.format_exc()}")
      return {
        "response": "I apologize, but I encountered an issue processing your request. Could you please try again?",
        "speak_verbatim": True,
      }

  async def _call_primary(self, context: str) -> str:
    """
    Call the primary agent and get response via message passing.

    Uses AgentReference.send() to communicate with the primary agent,
    making this compatible with the spawner pattern where agents are
    created on-demand.

    Args:
        context: Key context from the user's last message

    Returns:
        Primary agent's response text
    """
    # Build the message for primary agent with conversation history
    history_text = ""
    if self.conversation_history:
      history_lines = []
      for msg in self.conversation_history[-10:]:  # Last 10 messages for context
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        history_lines.append(f"{role}: {content}")
      history_text = "\n".join(history_lines)

    # Create the primary agent query message
    primary_message = f"""[Voice Agent Delegation Request]

Conversation History:
{history_text}

Latest Context from User: {context}

Please provide a response that will be spoken aloud by the voice agent.
Be concise and natural-sounding for voice. Avoid lists and complex formatting.
Keep responses brief (1-3 sentences max)."""

    # Use provided scope/conversation for memory isolation
    # Falls back to defaults if not provided
    scope = self.scope if self.scope is not None else "voice-delegation"
    conversation = self.conversation if self.conversation is not None else "delegation"

    try:
      # Send message to primary agent via AgentReference and get response
      # The primary agent will process the request, potentially calling tools
      response_messages = await self.primary.send(
        primary_message,
        scope=scope,
        conversation=conversation,
        timeout=30,  # 30 second timeout for voice interactions
      )

      # Extract the assistant's response from the returned messages
      response_text = ""
      for msg in response_messages:
        # Look for assistant messages with text content
        if hasattr(msg, "content") and msg.content:
          # Handle different content types
          content = msg.content
          if isinstance(content, str):
            response_text = content
          elif hasattr(content, "text"):
            response_text = content.text
          elif isinstance(content, list):
            # Handle list of content parts
            for part in content:
              if hasattr(part, "text"):
                response_text = part.text
                break
              elif isinstance(part, str):
                response_text = part
                break

      return response_text if response_text else "I'm sorry, I couldn't generate a response."

    except Exception as e:
      logger.error(f"[DELEGATE] Error sending to primary agent: {e}")
      logger.debug(f"[DELEGATE] Traceback: {traceback.format_exc()}")
      return "I'm sorry, I couldn't process that request."

  async def spec(self) -> dict:
    """Return OpenAI-compatible tool specification."""
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": {
          "type": "object",
          "properties": {
            "context": {
              "type": "string",
              "description": (
                "Key context from the user's most recent message. "
                "Include any specific information the primary agent needs to answer, "
                "such as account numbers, questions, or requests."
              ),
            }
          },
          "required": ["context"],
        },
      },
    }


# =============================================================================
# VOICE AGENT
# =============================================================================


class VoiceAgent:
  """
  Lightweight voice agent for real-time user interaction.

  Uses OpenAI Realtime API for low-latency voice communication.
  Delegates complex tasks to a primary agent.

  The VoiceAgent:
  - Handles basic tasks directly (greetings, chitchat, collecting info)
  - Uses filler phrases before delegating complex tasks
  - Delegates to primary agent for tool calls and complex reasoning
  - Speaks primary agent responses verbatim

  Attributes:
      name: Unique name for this voice agent
      primary: The primary agent to delegate to
      config: Voice configuration settings
      instructions: Generated or custom instructions for the voice agent
  """

  def __init__(
    self,
    name: str,
    primary: "Agent | AgentReference",
    config: VoiceConfig,
  ):
    """
    Initialize VoiceAgent.

    Args:
        name: Unique name for this voice agent
        primary: The primary agent (Agent instance or AgentReference) to delegate to
        config: Voice configuration settings
    """
    self.name = name
    self.primary = primary
    self.config = config

    # Generate voice agent instructions
    self.instructions = self._generate_instructions()

    logger.info(f"VoiceAgent '{name}' initialized with primary agent '{primary.name}'")

  def _get_config(self, key: str) -> Any:
    """Get a config value with default fallback."""
    return get_voice_config_value(self.config, key)

  def _generate_instructions(self) -> str:
    """
    Generate instructions for the voice agent.

    If custom instructions are provided in config, uses those.
    Otherwise, auto-generates instructions based on allowed actions
    and filler phrases.

    Returns:
        Instructions string for the voice agent
    """
    custom_instructions = self._get_config("instructions")
    if custom_instructions:
      return custom_instructions

    # Auto-generate based on config
    allowed_actions = self._get_config("allowed_actions")
    filler_phrases = self._get_config("filler_phrases")
    allowed = "\n".join([f"- {action}" for action in allowed_actions])
    fillers = ", ".join([f'"{p}"' for p in filler_phrases[:4]])

    return f"""You are a helpful voice assistant. You handle basic interactions directly and delegate complex tasks to the primary agent via the delegate_to_primary tool.

# What You Can Handle Directly
{allowed}

# When to Delegate
For ANY request not listed above, you MUST use the delegate_to_primary tool.
This includes:
- Factual questions about accounts, policies, or products
- Actions that require looking up information
- Complex requests that need tools or deeper reasoning
- Anything beyond basic chitchat

# Before Delegating
ALWAYS say a filler phrase first, such as: {fillers}
Never call delegate_to_primary without first saying something to the user.

# After Receiving Primary Agent Response
Read the primary agent's response VERBATIM. Do not modify, summarize, or add to it.

# Speaking Style
- Be concise and natural
- Speak clearly for voice
- Don't repeat yourself
- Vary your responses to sound natural
- Keep responses brief (1-2 sentences when possible)

# Example Flow
User: "What's my account balance?"
You: "Let me check on that." [then call delegate_to_primary]
[Primary agent returns: "Your balance is $142.50"]
You: "Your balance is $142.50"
"""

  async def create_session(
    self,
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
  ) -> "VoiceSession":
    """
    Create a new voice session.

    Args:
        scope: Optional scope for memory isolation (e.g., user ID)
        conversation: Optional conversation ID for memory isolation

    Returns:
        Connected VoiceSession instance

    Raises:
        RuntimeError: If websockets package is not installed
    """
    if websockets is None:
      raise RuntimeError("websockets package is required for voice support. Install it with: pip install websockets")

    session = VoiceSession(self, scope=scope, conversation=conversation)
    await session.connect()
    return session


class VoiceSession:
  """
  Manages a single voice conversation session.

  Handles WebSocket communication with the OpenAI Realtime API
  and coordinates delegation to the primary agent.

  The session:
  - Connects to OpenAI Realtime API
  - Configures the session with voice agent settings
  - Tracks conversation history for primary agent context
  - Handles tool calls (delegation) internally
  - Forwards audio and events to/from the client

  Attributes:
      agent: The VoiceAgent this session belongs to
      websocket: WebSocket connection to Realtime API
      connected: Whether the session is currently connected
      conversation_history: List tracking conversation messages
      scope: Optional scope for memory isolation (e.g., user ID)
      conversation: Optional conversation ID for memory isolation
  """

  def __init__(
    self,
    agent: VoiceAgent,
    scope: Optional[str] = None,
    conversation: Optional[str] = None,
  ):
    """
    Initialize VoiceSession.

    Args:
        agent: The VoiceAgent this session belongs to
        scope: Optional scope for memory isolation (e.g., user ID)
        conversation: Optional conversation ID for memory isolation
    """
    self.agent = agent
    self.websocket: Optional[WebSocketClientProtocol] = None
    self.connected = False
    self.conversation_history: List[Dict[str, str]] = []
    self.scope = scope
    self.conversation = conversation

    # Create delegation tool with access to conversation history and scope/conversation
    self.delegate_tool = DelegateToPrimaryTool(
      agent.primary,
      self.conversation_history,
      scope=scope,
      conversation=conversation,
    )

    # Interruption handling state
    self.is_responding = False  # True when a response is in progress
    self.is_output_audio_active = False  # True when audio is being output

    # Transcript ordering: buffer assistant transcript events until user transcript arrives
    # The user transcript comes from Whisper async and may arrive after the assistant
    # has already started responding with a filler phrase
    self.pending_assistant_events: List[Dict[str, Any]] = []
    self.user_transcript_received = False  # Track if we've received user transcript for current turn

    logger.debug(f"VoiceSession created for agent '{agent.name}' (scope={scope}, conversation={conversation})")

  def _get_websocket_url(self) -> str:
    """
    Get WebSocket URL for realtime API.

    Checks for LiteLLM proxy configuration first, falls back to direct OpenAI.

    Returns:
        WebSocket URL string
    """
    proxy_base = os.environ.get("LITELLM_PROXY_API_BASE")
    if proxy_base:
      # Convert HTTP(S) to WS(S)
      ws_base = proxy_base.replace("https://", "wss://").replace("http://", "ws://")
      url = f"{ws_base}/v1/realtime?model={self.agent._get_config('realtime_model')}"
      logger.debug(f"Using LiteLLM proxy WebSocket: {url}")
      return url

    # Fallback to direct OpenAI
    url = f"wss://api.openai.com/v1/realtime?model={self.agent._get_config('realtime_model')}"
    logger.debug(f"Using direct OpenAI WebSocket: {url}")
    return url

  def _get_auth_headers(self) -> Dict[str, str]:
    """
    Get authentication headers for WebSocket connection.

    Returns:
        Dictionary of headers
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

    logger.warning("No API key found in environment for realtime API")
    return headers

  async def connect(self):
    """
    Connect to the realtime API.

    Establishes WebSocket connection and sends session configuration.

    Raises:
        RuntimeError: If connection fails
    """
    url = self._get_websocket_url()
    headers = self._get_auth_headers()

    try:
      logger.info(f"Connecting to realtime API: {url}")

      self.websocket = await websockets.connect(
        url,
        additional_headers=headers,
        ping_interval=20,
        ping_timeout=20,
      )
      self.connected = True

      logger.info("✅ WebSocket connection established")

      # Configure session
      await self._configure_session()

      logger.info("✅ Session configuration sent")

    except Exception as e:
      self.connected = False
      logger.error(f"❌ Failed to connect to realtime API: {e}")
      raise RuntimeError(f"Failed to establish WebSocket connection: {e}")

  async def _configure_session(self):
    """
    Send session configuration to realtime API.

    Configures modalities, voice, VAD settings, and tools.
    """
    # Get tool spec for delegation
    delegate_tool_spec = await self.delegate_tool.spec()

    # Convert to realtime API tool format
    realtime_tool = {
      "type": "function",
      "name": delegate_tool_spec["function"]["name"],
      "description": delegate_tool_spec["function"]["description"],
      "parameters": delegate_tool_spec["function"]["parameters"],
    }

    config = {
      "type": "session.update",
      "session": {
        "modalities": ["text", "audio"],
        "instructions": self.agent.instructions,
        "voice": self.agent._get_config("voice"),
        "input_audio_format": self.agent._get_config("input_audio_format"),
        "output_audio_format": self.agent._get_config("output_audio_format"),
        "input_audio_transcription": {
          "model": "whisper-1",
        },
        "turn_detection": {
          "type": "server_vad",
          "threshold": self.agent._get_config("vad_threshold"),
          "prefix_padding_ms": self.agent._get_config("vad_prefix_padding_ms"),
          "silence_duration_ms": self.agent._get_config("vad_silence_duration_ms"),
        },
        "tools": [realtime_tool],
        "tool_choice": "auto",
        "temperature": 0.8,
        "max_response_output_tokens": 4096,
      },
    }

    await self.websocket.send(json.dumps(config))
    logger.debug(f"Session config sent: voice={self.agent._get_config('voice')}")

  async def handle_events(self) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process events from the realtime API.

    Yields events to forward to the client, handling tool calls
    (delegation to primary agent) internally.

    Yields:
        Event dictionaries from the realtime API

    Note:
        Tool calls (delegate_to_primary) are handled internally
        and not yielded to the caller.
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

        # Track response state
        if event_type == "response.created":
          self.is_responding = True
          logger.debug("📝 Response started")

        if event_type in ("response.done", "response.cancelled"):
          self.is_responding = False
          logger.debug("📝 Response ended")

        # Track output audio buffer state using actual OpenAI Realtime API events
        # Note: output_audio_buffer.started/stopped don't exist in the API
        # We use response.audio.delta (first chunk) and response.audio.done to track state
        if event_type == "response.audio.delta":
          if not self.is_output_audio_active:
            self.is_output_audio_active = True
            logger.debug("🔊 Output audio started")

        # Reset audio active flag when audio stream completes
        if event_type == "response.audio.done":
          if self.is_output_audio_active:
            self.is_output_audio_active = False
            logger.debug("🔇 Output audio stopped")

        # Detect interruption: user started speaking while assistant is responding/playing audio
        if event_type == "input_audio_buffer.speech_started":
          was_interrupted = self.is_responding or self.is_output_audio_active
          logger.debug(
            f"🎤 Speech started: is_responding={self.is_responding}, is_output_audio_active={self.is_output_audio_active}, was_interrupted={was_interrupted}"
          )
          if was_interrupted:
            await self._handle_interruption()
          # Mark the event so _translate_event knows this was an interruption
          event["_was_interrupted"] = was_interrupted
          # Reset transcript state for new turn
          self.pending_assistant_events = []
          self.user_transcript_received = False

        # Handle function call arguments being accumulated
        if event_type == "response.function_call_arguments.done":
          await self._handle_function_call(event)
          # Don't yield - handled internally
          continue

        # Track user transcription for primary agent context
        if event_type == "conversation.item.input_audio_transcription.completed":
          transcript = event.get("transcript", "")
          if transcript:
            self.conversation_history.append({"role": "user", "content": transcript})
            logger.info(f"🗣️ User said: {transcript}")
            self.user_transcript_received = True

        # Track assistant responses for context
        if event_type == "response.audio_transcript.done":
          transcript = event.get("transcript", "")
          if transcript:
            self.conversation_history.append({"role": "assistant", "content": transcript})
            logger.info(f"🤖 Assistant said: {transcript}")

        # Forward event to client
        yield event

    except websockets.exceptions.ConnectionClosed as e:
      logger.info(f"WebSocket connection closed: {e}")
      self.connected = False
    except Exception as e:
      logger.error(f"Error receiving events: {e}")
      self.connected = False
      raise

  async def _handle_function_call(self, event: Dict[str, Any]):
    """
    Handle function call from the realtime API.

    Processes delegate_to_primary tool calls by invoking the
    primary agent and sending the result back.

    Args:
        event: Function call event from realtime API
    """
    call_id = event.get("call_id")
    name = event.get("name")
    arguments = event.get("arguments", "{}")

    logger.info(f"📞 Function call: {name} (call_id={call_id})")

    if name == "delegate_to_primary":
      logger.info("🔄 Delegating to primary agent...")

      try:
        # Invoke the delegation tool
        result = await self.delegate_tool.invoke(arguments)

        logger.info(f"✅ Primary agent returned: {str(result)[:100]}...")

        # Send function result back to realtime API
        await self.websocket.send(
          json.dumps(
            {
              "type": "conversation.item.create",
              "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
              },
            }
          )
        )

        # Request response (voice agent will speak the primary agent's response)
        await self.websocket.send(json.dumps({"type": "response.create"}))

        logger.info("📤 Sent primary agent response to realtime API")

      except Exception as e:
        logger.error(f"❌ Error handling delegation: {e}")

        # Send error response
        await self.websocket.send(
          json.dumps(
            {
              "type": "conversation.item.create",
              "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                  {
                    "response": "I apologize, but I encountered an issue. Could you please try again?",
                    "speak_verbatim": True,
                  }
                ),
              },
            }
          )
        )

        await self.websocket.send(json.dumps({"type": "response.create"}))

    else:
      logger.warning(f"Unknown function call: {name}")

  async def send_audio(self, audio_chunk: bytes):
    """
    Send audio to the realtime API.

    Args:
        audio_chunk: Raw audio bytes (PCM16 format)

    Raises:
        RuntimeError: If session is not connected
    """
    if not self.connected or not self.websocket:
      raise RuntimeError("Session not connected. Call connect() first.")

    audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")

    await self.websocket.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))

    logger.debug(f"Sent audio chunk: {len(audio_chunk)} bytes")

  async def send_event(self, event: Dict[str, Any]):
    """
    Send a custom event to the realtime API.

    Args:
        event: Event dictionary (must have "type" field)

    Raises:
        RuntimeError: If session is not connected
    """
    if not self.connected or not self.websocket:
      raise RuntimeError("Session not connected. Call connect() first.")

    await self.websocket.send(json.dumps(event))
    logger.debug(f"Sent event: {event.get('type')}")

  async def commit_audio_buffer(self):
    """
    Commit the audio buffer for processing.

    Call this when you want to force processing of buffered audio
    without waiting for server-side VAD.
    """
    if not self.connected or not self.websocket:
      raise RuntimeError("Session not connected. Call connect() first.")

    await self.websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
    logger.debug("Committed audio buffer")

  async def cancel_response(self):
    """
    Cancel the current response.

    Useful for interrupting the assistant when the user starts speaking.
    """
    if not self.connected or not self.websocket:
      raise RuntimeError("Session not connected. Call connect() first.")

    await self.websocket.send(json.dumps({"type": "response.cancel"}))
    logger.debug("Cancelled response")

  async def _handle_interruption(self):
    """
    Handle user interruption using two-part cancellation.

    This method is called when the user starts speaking while the assistant
    is responding or playing audio. It:
    1. Cancels the in-progress response (response.cancel)
    2. Clears the output audio buffer (output_audio_buffer.clear)

    This ensures both the response generation and audio playback are stopped
    immediately, allowing the user to speak without the assistant talking over them.
    """
    logger.info("⚡ User interruption detected - cancelling response and clearing audio")

    # 1. Cancel the response if one is in progress
    if self.is_responding:
      try:
        await self.websocket.send(json.dumps({"type": "response.cancel"}))
        self.is_responding = False
        logger.debug("📝 Response cancelled due to interruption")
      except Exception as e:
        logger.warning(f"Failed to cancel response: {e}")

    # 2. Clear output audio buffer if audio is active
    if self.is_output_audio_active:
      try:
        await self.websocket.send(json.dumps({"type": "output_audio_buffer.clear"}))
        self.is_output_audio_active = False
        logger.debug("🔇 Output audio buffer cleared due to interruption")
      except Exception as e:
        logger.warning(f"Failed to clear output audio buffer: {e}")

  def _translate_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Translate a realtime API event to client protocol.

    Args:
        event: Event from OpenAI Realtime API

    Returns:
        Client message dict, or None if event should not be forwarded
    """
    event_type = event.get("type")

    if event_type == "response.audio.delta":
      audio_delta = event.get("delta", "")
      if audio_delta:
        return {"type": "audio", "audio": audio_delta}

    elif event_type == "response.audio_transcript.delta":
      transcript_delta = event.get("delta", "")
      if transcript_delta:
        assistant_event = {"type": "transcript_delta", "text": transcript_delta, "role": "assistant"}
        # If user transcript hasn't arrived yet, buffer the assistant event
        if not self.user_transcript_received:
          self.pending_assistant_events.append(assistant_event)
          return None
        return assistant_event

    elif event_type == "response.audio_transcript.done":
      transcript = event.get("transcript", "")
      if transcript:
        assistant_event = {"type": "transcript", "text": transcript, "role": "assistant"}
        # If user transcript hasn't arrived yet, buffer the assistant event
        if not self.user_transcript_received:
          self.pending_assistant_events.append(assistant_event)
          return None
        return assistant_event

    elif event_type == "conversation.item.input_audio_transcription.completed":
      # User transcript arrived - emit it along with any buffered assistant events
      user_transcript = event.get("transcript", "")
      if user_transcript:
        user_event = {"type": "transcript", "text": user_transcript, "role": "user"}
        # If we have buffered assistant events, emit them all in correct order
        if self.pending_assistant_events:
          events = [user_event] + self.pending_assistant_events
          self.pending_assistant_events = []
          return {"type": "transcript_batch", "events": events}
        return user_event
      return None

    elif event_type == "input_audio_buffer.speech_started":
      # Note: _handle_interruption is called before _translate_event in handle_events,
      # but we check the original state by looking at whether we just handled an interruption.
      # The interrupted flag tells the client to clear its audio queue immediately.
      was_interrupted = event.get("_was_interrupted", False)
      logger.debug(f"🎤 Speech started (interrupted={was_interrupted})")
      return {"type": "speech_started", "interrupted": was_interrupted}

    elif event_type == "input_audio_buffer.speech_stopped":
      logger.debug("⏸️ Speech stopped")
      return {"type": "speech_stopped"}

    elif event_type == "response.done":
      logger.debug("✅ Response complete")
      return {"type": "response_complete"}

    elif event_type == "error":
      error = event.get("error", {})
      error_msg = error.get("message", "Unknown error")
      logger.error(f"❌ Realtime API error: {error_msg}")
      return {"type": "error", "error": error_msg}

    # Session events - log only, don't forward
    elif event_type == "session.created":
      logger.debug("✅ Realtime session created")

    elif event_type == "session.updated":
      logger.debug("✅ Realtime session updated")

    return None

  async def run_with_client(
    self,
    send_json: Callable[[Dict[str, Any]], Awaitable[None]],
    receive_text: Callable[[], Awaitable[str]],
    voice_config: "VoiceConfig",
    agent_name: str,
  ):
    """
    Run bidirectional communication with a client WebSocket.

    This is the main loop that:
    - Receives audio/events from client and forwards to realtime API
    - Receives events from realtime API and forwards to client

    Args:
        send_json: Async function to send JSON dict to client
        receive_text: Async function to receive text from client
        voice_config: Voice configuration (for connection confirmation)
        agent_name: Name of the agent (for logging and confirmation)
    """
    # Send connection confirmation
    await send_json(
      {
        "type": "connected",
        "message": f"Connected to '{agent_name}' voice (Voice Interface pattern)",
        "config": {
          "realtime_model": get_voice_config_value(voice_config, "realtime_model"),
          "voice": get_voice_config_value(voice_config, "voice"),
        },
        "scope": self.scope,
        "conversation": self.conversation,
      }
    )

    async def receive_from_client():
      try:
        while True:
          message = await receive_text()
          data = json.loads(message)
          msg_type = data.get("type")

          if msg_type == "audio":
            audio_base64 = data.get("audio", "")
            if audio_base64:
              audio_bytes = base64.b64decode(audio_base64)
              await self.send_audio(audio_bytes)

          elif msg_type == "event":
            event = data.get("event", {})
            if event:
              await self.send_event(event)

          elif msg_type == "config":
            logger.info(f"📝 Received config from client: {data}")

          elif msg_type == "close":
            logger.info(f"📪 Client requested close for agent '{agent_name}'")
            break

      except Exception as e:
        logger.info(f"📪 Client disconnected from agent '{agent_name}': {e}")

    async def receive_from_realtime():
      try:
        async for event in self.handle_events():
          client_msg = self._translate_event(event)
          if client_msg:
            # Handle transcript_batch by sending individual events in order
            if client_msg.get("type") == "transcript_batch":
              for sub_event in client_msg.get("events", []):
                await send_json(sub_event)
            else:
              await send_json(client_msg)
      except Exception as e:
        logger.error(f"❌ Error receiving from realtime API: {e}")

    await asyncio.gather(receive_from_client(), receive_from_realtime(), return_exceptions=True)

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
    """Context manager entry - connection already established."""
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit - close connection."""
    await self.close()

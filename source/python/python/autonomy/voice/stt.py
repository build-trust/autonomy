from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import base64
import os
from ..logs import get_logger

logger = get_logger("stt")


class STTProvider(ABC):
  """Abstract base class for Speech-to-Text providers"""

  @abstractmethod
  async def transcribe_stream(self, audio_stream: AsyncIterator[bytes], language: str = "en") -> AsyncIterator[str]:
    """
    Transcribe streaming audio to text.

    Yields text chunks as they become available.

    Args:
        audio_stream: Async iterator of audio bytes (PCM16 format)
        language: Language code (e.g., "en", "es", "fr")

    Yields:
        Transcribed text chunks
    """
    pass

  @abstractmethod
  async def transcribe(self, audio: bytes, language: str = "en") -> str:
    """
    Transcribe a complete audio buffer to text.

    Args:
        audio: Complete audio buffer (PCM16 format)
        language: Language code (e.g., "en", "es", "fr")

    Returns:
        Transcribed text
    """
    pass


class OpenAISTT(STTProvider):
  """OpenAI Whisper STT implementation"""

  def __init__(self, model: str = "whisper-1", api_key: Optional[str] = None, sample_rate: int = 16000):
    """
    Initialize OpenAI STT provider.

    Args:
        model: Whisper model name (default: "whisper-1")
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var, not needed if using LiteLLM proxy)
        sample_rate: Audio sample rate in Hz (default: 16000)
    """
    self.model = model
    self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
    self.sample_rate = sample_rate

    # Check for LiteLLM proxy configuration
    self.litellm_proxy_base = os.environ.get("LITELLM_PROXY_API_BASE")
    self.litellm_proxy_key = os.environ.get("LITELLM_PROXY_API_KEY")
    self.use_proxy = bool(self.litellm_proxy_base)

    # Only require API key if not using proxy
    if not self.use_proxy and not self.api_key:
      raise ValueError(
        "OpenAI API key required (set OPENAI_API_KEY env var) or configure LiteLLM proxy (LITELLM_PROXY_API_BASE)"
      )

    if self.use_proxy:
      logger.info(f"OpenAI STT initialized: model={model}, using LiteLLM proxy at {self.litellm_proxy_base}")
    else:
      logger.info(f"OpenAI STT initialized: model={model}, using OpenAI API directly")

  async def transcribe_stream(self, audio_stream: AsyncIterator[bytes], language: str = "en") -> AsyncIterator[str]:
    """
    Transcribe streaming audio to text.

    Note: OpenAI Whisper API doesn't support true streaming,
    so we buffer audio chunks and transcribe complete utterances.

    Args:
        audio_stream: Async iterator of audio bytes
        language: Language code

    Yields:
        Transcribed text (one chunk per complete utterance)
    """
    # Buffer audio chunks
    audio_buffer = bytearray()

    async for audio_chunk in audio_stream:
      audio_buffer.extend(audio_chunk)

      # When stream ends, transcribe the complete buffer
      # (In practice, VAD will determine utterance boundaries)

    if audio_buffer:
      text = await self.transcribe(bytes(audio_buffer), language)
      if text:
        yield text

  async def transcribe(self, audio: bytes, language: str = "en") -> str:
    """
    Transcribe a complete audio buffer to text using OpenAI Whisper API.

    Args:
        audio: Complete audio buffer (raw PCM16 or WAV format)
        language: Language code

    Returns:
        Transcribed text
    """
    try:
      # Import here to avoid requiring openai if not using this provider
      from openai import AsyncOpenAI

      # Configure client based on proxy settings
      if self.use_proxy:
        client = AsyncOpenAI(
          api_key=self.litellm_proxy_key or "dummy",  # LiteLLM proxy may not need a key
          base_url=self.litellm_proxy_base,
        )
      else:
        client = AsyncOpenAI(api_key=self.api_key)

      # Convert raw PCM16 to WAV format if needed
      # OpenAI Whisper API expects a file-like object with proper headers
      from io import BytesIO
      import wave

      # Create WAV file in memory
      wav_buffer = BytesIO()
      with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(self.sample_rate)
        wav_file.writeframes(audio)

      # Seek to beginning
      wav_buffer.seek(0)
      wav_buffer.name = "audio.wav"  # OpenAI client checks for filename

      # Transcribe
      logger.debug(f"Transcribing {len(audio)} bytes with Whisper")
      response = await client.audio.transcriptions.create(
        model=self.model, file=wav_buffer, language=language, response_format="text"
      )

      # Handle both string and dict responses (LiteLLM proxy may return JSON)
      if isinstance(response, str):
        # Try to parse as JSON first (LiteLLM proxy behavior)
        try:
          import json

          response_dict = json.loads(response)
          text = response_dict.get("text", "").strip()
        except (json.JSONDecodeError, AttributeError):
          # Not JSON, use as plain text
          text = response.strip()
      elif isinstance(response, dict):
        # Direct dict response
        text = response.get("text", "").strip()
      else:
        # Unknown response type
        text = str(response).strip() if response else ""

      logger.info(f"✅ Transcribed: {text[:100]}...")
      return text

    except Exception as e:
      logger.error(f"❌ OpenAI STT error: {e}")
      raise


class DeepgramSTT(STTProvider):
  """Deepgram streaming STT implementation"""

  def __init__(self, model: str = "nova-2", api_key: Optional[str] = None, sample_rate: int = 24000):
    """
    Initialize Deepgram STT provider.

    Args:
        model: Deepgram model name (default: "nova-2")
        api_key: Deepgram API key (defaults to DEEPGRAM_API_KEY env var)
        sample_rate: Audio sample rate in Hz
    """
    self.model = model
    self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    self.sample_rate = sample_rate

    if not self.api_key:
      raise ValueError("Deepgram API key required (set DEEPGRAM_API_KEY env var)")

    logger.info(f"Deepgram STT initialized: model={model}, sample_rate={sample_rate}")

  async def transcribe_stream(self, audio_stream: AsyncIterator[bytes], language: str = "en") -> AsyncIterator[str]:
    """
    Transcribe streaming audio to text using Deepgram WebSocket API.

    Args:
        audio_stream: Async iterator of audio bytes
        language: Language code

    Yields:
        Transcribed text chunks as they become available
    """
    try:
      # Import here to avoid requiring deepgram-sdk if not using this provider
      from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

      # Create Deepgram client
      deepgram = DeepgramClient(self.api_key)

      # Configure streaming options
      options = LiveOptions(
        model=self.model,
        language=language,
        encoding="linear16",
        sample_rate=self.sample_rate,
        channels=1,
        interim_results=False,  # Only send final transcripts
        punctuate=True,
        smart_format=True,
      )

      # Start live transcription connection
      connection = deepgram.listen.live.v("1")

      # Set up event handlers
      transcripts = []

      def on_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if sentence:
          transcripts.append(sentence)

      def on_error(self, error, **kwargs):
        logger.error(f"Deepgram error: {error}")

      connection.on(LiveTranscriptionEvents.Transcript, on_message)
      connection.on(LiveTranscriptionEvents.Error, on_error)

      # Start the connection
      if not await connection.start(options):
        raise Exception("Failed to start Deepgram connection")

      # Stream audio chunks
      async for audio_chunk in audio_stream:
        connection.send(audio_chunk)

        # Yield any new transcripts
        while transcripts:
          yield transcripts.pop(0)

      # Finish the stream
      connection.finish()

      # Yield remaining transcripts
      while transcripts:
        yield transcripts.pop(0)

    except ImportError:
      logger.error("deepgram-sdk not installed. Install with: pip install deepgram-sdk")
      raise
    except Exception as e:
      logger.error(f"❌ Deepgram STT error: {e}")
      raise

  async def transcribe(self, audio: bytes, language: str = "en") -> str:
    """
    Transcribe a complete audio buffer to text using Deepgram REST API.

    Args:
        audio: Complete audio buffer (raw PCM16 format)
        language: Language code

    Returns:
        Transcribed text
    """
    try:
      from deepgram import DeepgramClient, PrerecordedOptions

      # Create Deepgram client
      deepgram = DeepgramClient(self.api_key)

      # Configure options
      options = PrerecordedOptions(
        model=self.model,
        language=language,
        punctuate=True,
        smart_format=True,
      )

      # Transcribe
      logger.debug(f"Transcribing {len(audio)} bytes with Deepgram")
      response = await deepgram.listen.prerecorded.v("1").transcribe_file(
        {"buffer": audio, "mimetype": "audio/raw"}, options
      )

      text = response.results.channels[0].alternatives[0].transcript
      logger.info(f"✅ Transcribed: {text[:100]}...")
      return text

    except ImportError:
      logger.error("deepgram-sdk not installed. Install with: pip install deepgram-sdk")
      raise
    except Exception as e:
      logger.error(f"❌ Deepgram STT error: {e}")
      raise


class AssemblyAISTT(STTProvider):
  """AssemblyAI STT implementation"""

  def __init__(self, api_key: Optional[str] = None):
    """
    Initialize AssemblyAI STT provider.

    Args:
        api_key: AssemblyAI API key (defaults to ASSEMBLYAI_API_KEY env var)
    """
    self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")

    if not self.api_key:
      raise ValueError("AssemblyAI API key required (set ASSEMBLYAI_API_KEY env var)")

    logger.info("AssemblyAI STT initialized")

  async def transcribe_stream(self, audio_stream: AsyncIterator[bytes], language: str = "en") -> AsyncIterator[str]:
    """
    Transcribe streaming audio to text.

    Note: AssemblyAI supports real-time streaming via WebSocket.

    Args:
        audio_stream: Async iterator of audio bytes
        language: Language code

    Yields:
        Transcribed text chunks
    """
    # Buffer for now - streaming implementation can be added later
    audio_buffer = bytearray()
    async for audio_chunk in audio_stream:
      audio_buffer.extend(audio_chunk)

    if audio_buffer:
      text = await self.transcribe(bytes(audio_buffer), language)
      if text:
        yield text

  async def transcribe(self, audio: bytes, language: str = "en") -> str:
    """
    Transcribe a complete audio buffer to text using AssemblyAI API.

    Args:
        audio: Complete audio buffer
        language: Language code

    Returns:
        Transcribed text
    """
    try:
      import assemblyai as aai

      aai.settings.api_key = self.api_key

      # Create transcriber
      transcriber = aai.Transcriber()

      # Upload audio and transcribe
      logger.debug(f"Transcribing {len(audio)} bytes with AssemblyAI")

      # AssemblyAI requires a file path or URL
      # Save to temporary file
      import tempfile

      with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Convert to WAV
        import wave

        with wave.open(tmp.name, "wb") as wav_file:
          wav_file.setnchannels(1)
          wav_file.setsampwidth(2)
          wav_file.setframerate(24000)
          wav_file.writeframes(audio)

        # Transcribe
        transcript = transcriber.transcribe(tmp.name)

      # Clean up temp file
      import os

      os.unlink(tmp.name)

      text = transcript.text or ""
      logger.info(f"✅ Transcribed: {text[:100]}...")
      return text

    except ImportError:
      logger.error("assemblyai not installed. Install with: pip install assemblyai")
      raise
    except Exception as e:
      logger.error(f"❌ AssemblyAI STT error: {e}")
      raise


def create_stt_provider(
  provider: str, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> STTProvider:
  """
  Factory function to create STT provider instances.

  Args:
      provider: Provider name ("openai", "deepgram", "assemblyai")
      model: Model name (provider-specific)
      api_key: API key (falls back to environment variables)
      **kwargs: Additional provider-specific options

  Returns:
      STTProvider instance

  Raises:
      ValueError: If provider is unknown
  """
  provider = provider.lower()

  if provider == "openai":
    sample_rate = kwargs.get("sample_rate", 16000)
    return OpenAISTT(model=model or "whisper-1", api_key=api_key, sample_rate=sample_rate)
  elif provider == "deepgram":
    return DeepgramSTT(model=model or "nova-2", api_key=api_key, **kwargs)
  elif provider == "assemblyai":
    return AssemblyAISTT(api_key=api_key)
  else:
    raise ValueError(f"Unknown STT provider: {provider}. Supported providers: openai, deepgram, assemblyai")

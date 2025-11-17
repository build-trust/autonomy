from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import os
from ..logs import get_logger

logger = get_logger("tts")


class TTSProvider(ABC):
  """Abstract base class for Text-to-Speech providers"""

  @abstractmethod
  async def synthesize_stream(self, text: str, voice: str = "alloy") -> AsyncIterator[bytes]:
    """
    Convert text to speech, yielding audio chunks.

    Enables streaming audio back to client as it's generated.

    Args:
        text: Text to convert to speech
        voice: Voice ID (provider-specific)

    Yields:
        Audio chunks (PCM16 format)
    """
    pass

  @abstractmethod
  async def synthesize(self, text: str, voice: str = "alloy") -> bytes:
    """
    Convert text to speech, return complete audio.

    Args:
        text: Text to convert to speech
        voice: Voice ID (provider-specific)

    Returns:
        Complete audio buffer (PCM16 format)
    """
    pass


class OpenAITTS(TTSProvider):
  """OpenAI TTS implementation"""

  def __init__(self, model: str = "tts-1", api_key: Optional[str] = None, response_format: str = "pcm"):
    """
    Initialize OpenAI TTS provider.

    Args:
        model: TTS model name ("tts-1" or "tts-1-hd")
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var, not needed if using LiteLLM proxy)
        response_format: Audio format ("pcm", "opus", "aac", "flac")
    """
    self.model = model
    self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
    self.response_format = response_format

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
      logger.info(
        f"OpenAI TTS initialized: model={model}, format={response_format}, using LiteLLM proxy at {self.litellm_proxy_base}"
      )
    else:
      logger.info(f"OpenAI TTS initialized: model={model}, format={response_format}, using OpenAI API directly")

  async def synthesize_stream(self, text: str, voice: str = "alloy") -> AsyncIterator[bytes]:
    """
    Convert text to speech, yielding audio chunks.

    Args:
        text: Text to convert to speech
        voice: Voice ID (alloy, echo, fable, onyx, nova, shimmer)

    Yields:
        Audio chunks
    """
    try:
      from openai import AsyncOpenAI

      # Configure client based on proxy settings
      if self.use_proxy:
        client = AsyncOpenAI(
          api_key=self.litellm_proxy_key or "dummy",  # LiteLLM proxy may not need a key
          base_url=self.litellm_proxy_base,
        )
      else:
        client = AsyncOpenAI(api_key=self.api_key)

      logger.debug(f"Synthesizing text with OpenAI TTS: {text[:100]}...")

      # OpenAI TTS supports streaming
      async with client.audio.speech.with_streaming_response.create(
        model=self.model,
        voice=voice,
        input=text,
        response_format=self.response_format,
      ) as response:
        # Stream audio chunks
        async for chunk in response.iter_bytes(chunk_size=4096):
          if chunk:
            yield chunk

      logger.info(f"✅ TTS synthesis complete")

    except Exception as e:
      logger.error(f"❌ OpenAI TTS error: {e}")
      raise

  async def synthesize(self, text: str, voice: str = "alloy") -> bytes:
    """
    Convert text to speech, return complete audio.

    Args:
        text: Text to convert to speech
        voice: Voice ID (alloy, echo, fable, onyx, nova, shimmer)

    Returns:
        Complete audio buffer
    """
    try:
      from openai import AsyncOpenAI

      # Configure client based on proxy settings
      if self.use_proxy:
        client = AsyncOpenAI(
          api_key=self.litellm_proxy_key or "dummy",  # LiteLLM proxy may not need a key
          base_url=self.litellm_proxy_base,
        )
      else:
        client = AsyncOpenAI(api_key=self.api_key)

      logger.debug(f"Synthesizing text with OpenAI TTS: {text[:100]}...")

      # Get complete audio
      response = await client.audio.speech.create(
        model=self.model,
        voice=voice,
        input=text,
        response_format=self.response_format,
      )

      # Read all audio data
      audio_data = response.content

      logger.info(f"✅ TTS synthesis complete ({len(audio_data)} bytes)")
      return audio_data

    except Exception as e:
      logger.error(f"❌ OpenAI TTS error: {e}")
      raise


class ElevenLabsTTS(TTSProvider):
  """ElevenLabs TTS implementation"""

  def __init__(self, model: str = "eleven_monolingual_v1", api_key: Optional[str] = None):
    """
    Initialize ElevenLabs TTS provider.

    Args:
        model: Model ID (e.g., "eleven_monolingual_v1", "eleven_multilingual_v2")
        api_key: ElevenLabs API key (defaults to ELEVENLABS_API_KEY env var)
    """
    self.model = model
    self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")

    if not self.api_key:
      raise ValueError("ElevenLabs API key required (set ELEVENLABS_API_KEY env var)")

    logger.info(f"ElevenLabs TTS initialized: model={model}")

  async def synthesize_stream(
    self,
    text: str,
    voice: str = "21m00Tcm4TlvDq8ikWAM",  # Default voice ID
  ) -> AsyncIterator[bytes]:
    """
    Convert text to speech, yielding audio chunks.

    Args:
        text: Text to convert to speech
        voice: Voice ID (ElevenLabs voice ID)

    Yields:
        Audio chunks
    """
    try:
      from elevenlabs import AsyncElevenLabs

      client = AsyncElevenLabs(api_key=self.api_key)

      logger.debug(f"Synthesizing text with ElevenLabs: {text[:100]}...")

      # ElevenLabs supports streaming
      audio_stream = await client.text_to_speech.convert(
        voice_id=voice,
        text=text,
        model_id=self.model,
        output_format="pcm_24000",
      )

      # Stream audio chunks
      async for chunk in audio_stream:
        if chunk:
          yield chunk

      logger.info(f"✅ ElevenLabs TTS synthesis complete")

    except ImportError:
      logger.error("elevenlabs not installed. Install with: pip install elevenlabs")
      raise
    except Exception as e:
      logger.error(f"❌ ElevenLabs TTS error: {e}")
      raise

  async def synthesize(self, text: str, voice: str = "21m00Tcm4TlvDq8ikWAM") -> bytes:
    """
    Convert text to speech, return complete audio.

    Args:
        text: Text to convert to speech
        voice: Voice ID (ElevenLabs voice ID)

    Returns:
        Complete audio buffer
    """
    # Collect all chunks from streaming API
    chunks = []
    async for chunk in self.synthesize_stream(text, voice):
      chunks.append(chunk)

    audio_data = b"".join(chunks)
    logger.info(f"✅ ElevenLabs TTS synthesis complete ({len(audio_data)} bytes)")
    return audio_data


class PlayHTTTS(TTSProvider):
  """Play.ht TTS implementation"""

  def __init__(self, voice: str = "en-US-JennyNeural", api_key: Optional[str] = None, user_id: Optional[str] = None):
    """
    Initialize Play.ht TTS provider.

    Args:
        voice: Voice ID
        api_key: Play.ht API key (defaults to PLAYHT_API_KEY env var)
        user_id: Play.ht user ID (defaults to PLAYHT_USER_ID env var)
    """
    self.voice = voice
    self.api_key = api_key or os.environ.get("PLAYHT_API_KEY")
    self.user_id = user_id or os.environ.get("PLAYHT_USER_ID")

    if not self.api_key or not self.user_id:
      raise ValueError("Play.ht API key and user ID required (set PLAYHT_API_KEY and PLAYHT_USER_ID env vars)")

    logger.info(f"Play.ht TTS initialized: voice={voice}")

  async def synthesize_stream(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
    """
    Convert text to speech, yielding audio chunks.

    Args:
        text: Text to convert to speech
        voice: Voice ID (optional, uses default if not provided)

    Yields:
        Audio chunks
    """
    import aiohttp

    voice = voice or self.voice

    logger.debug(f"Synthesizing text with Play.ht: {text[:100]}...")

    url = "https://api.play.ht/api/v2/tts/stream"
    headers = {
      "Authorization": f"Bearer {self.api_key}",
      "X-User-ID": self.user_id,
      "Content-Type": "application/json",
    }
    payload = {
      "text": text,
      "voice": voice,
      "output_format": "pcm",
      "sample_rate": 24000,
    }

    try:
      async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
          if response.status != 200:
            error_text = await response.text()
            raise Exception(f"Play.ht API error: {response.status} - {error_text}")

          # Stream audio chunks
          async for chunk in response.content.iter_chunked(4096):
            if chunk:
              yield chunk

      logger.info(f"✅ Play.ht TTS synthesis complete")

    except Exception as e:
      logger.error(f"❌ Play.ht TTS error: {e}")
      raise

  async def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
    """
    Convert text to speech, return complete audio.

    Args:
        text: Text to convert to speech
        voice: Voice ID (optional, uses default if not provided)

    Returns:
        Complete audio buffer
    """
    # Collect all chunks from streaming API
    chunks = []
    async for chunk in self.synthesize_stream(text, voice):
      chunks.append(chunk)

    audio_data = b"".join(chunks)
    logger.info(f"✅ Play.ht TTS synthesis complete ({len(audio_data)} bytes)")
    return audio_data


def create_tts_provider(
  provider: str, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> TTSProvider:
  """
  Factory function to create TTS provider instances.

  Args:
      provider: Provider name ("openai", "elevenlabs", "play.ht")
      model: Model name (provider-specific)
      api_key: API key (falls back to environment variables)
      **kwargs: Additional provider-specific options

  Returns:
      TTSProvider instance

  Raises:
      ValueError: If provider is unknown
  """
  provider = provider.lower()

  if provider == "openai":
    return OpenAITTS(model=model or "tts-1", api_key=api_key, **kwargs)
  elif provider == "elevenlabs":
    return ElevenLabsTTS(model=model or "eleven_monolingual_v1", api_key=api_key)
  elif provider == "play.ht" or provider == "playht":
    return PlayHTTTS(api_key=api_key, **kwargs)
  else:
    raise ValueError(f"Unknown TTS provider: {provider}. Supported providers: openai, elevenlabs, play.ht")

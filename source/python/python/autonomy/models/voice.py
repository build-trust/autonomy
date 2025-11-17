from typing import Literal, Optional


class Voice:
  """
  Configuration for voice input/output.

  This is NOT a model - just I/O settings for how the agent
  communicates via voice. The agent's main Model handles all reasoning;
  Voice only configures how audio is transcribed (STT) and synthesized (TTS).

  Architecture:
    Audio Input → STT → Agent (uses main Model) → TTS → Audio Output

  Example:
      agent = await Agent.start(
          model=Model("claude-sonnet-4-v1"),  # Handles reasoning
          voice=Voice(                        # Just I/O settings
              voice="alloy",
              stt_provider="openai",
              tts_provider="openai",
          )
      )
  """

  def __init__(
    self,
    voice: str = "alloy",
    stt_provider: Literal["openai", "deepgram", "assemblyai"] = "openai",
    tts_provider: Literal["openai", "elevenlabs", "play.ht"] = "openai",
    stt_model: str = "whisper-1",
    tts_model: str = "tts-1",
    language: str = "en",
    vad_enabled: bool = True,
    vad_method: Literal["energy", "webrtc"] = "webrtc",
    vad_threshold: float = 0.5,
    vad_silence_duration_ms: int = 500,
    audio_format: str = "pcm16",
    sample_rate: int = 24000,
  ):
    """
    Initialize Voice I/O configuration.

    Args:
        voice: TTS voice ID (OpenAI: alloy, echo, fable, onyx, nova, shimmer)
        stt_provider: Speech-to-text provider (openai, deepgram, assemblyai)
        tts_provider: Text-to-speech provider (openai, elevenlabs, play.ht)
        stt_model: STT model name (e.g., "whisper-1")
        tts_model: TTS model name (e.g., "tts-1", "tts-1-hd")
        language: Language code for transcription (e.g., "en", "es", "fr")
        vad_enabled: Enable Voice Activity Detection
        vad_method: VAD method (energy or webrtc - webrtc is more accurate)
        vad_threshold: VAD sensitivity (0.0-1.0, higher = more sensitive, only for energy method)
        vad_silence_duration_ms: Silence duration to detect utterance end (ms)
        audio_format: Audio format (pcm16, g711_ulaw, g711_alaw)
        sample_rate: Audio sample rate in Hz (16000, 24000, 44100, 48000)
    """
    self.voice = voice
    self.stt_provider = stt_provider
    self.tts_provider = tts_provider
    self.stt_model = stt_model
    self.tts_model = tts_model
    self.language = language
    self.vad_enabled = vad_enabled
    self.vad_method = vad_method
    self.vad_threshold = vad_threshold
    self.vad_silence_duration_ms = vad_silence_duration_ms
    self.audio_format = audio_format
    self.sample_rate = sample_rate

  def __repr__(self) -> str:
    return (
      f"Voice(voice={self.voice!r}, "
      f"stt_provider={self.stt_provider!r}, "
      f"tts_provider={self.tts_provider!r}, "
      f"language={self.language!r})"
    )

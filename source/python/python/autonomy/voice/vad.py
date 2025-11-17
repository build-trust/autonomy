import asyncio
import time
from typing import Optional, Tuple
import numpy as np
from ..logs import get_logger

logger = get_logger("vad")


class VoiceActivityDetector:
  """
  Detect when user starts/stops speaking.

  Buffers audio and detects speech boundaries to know when
  to send audio for transcription.

  Uses energy-based detection with configurable threshold and
  silence duration.
  """

  def __init__(
    self,
    threshold: float = 0.5,
    silence_duration_ms: int = 500,
    sample_rate: int = 24000,
    frame_duration_ms: int = 20,
  ):
    """
    Initialize Voice Activity Detector.

    Args:
        threshold: VAD sensitivity (0.0-1.0, higher = more sensitive)
        silence_duration_ms: Silence duration to detect utterance end (ms)
        sample_rate: Audio sample rate in Hz
        frame_duration_ms: Audio frame duration in milliseconds
    """
    self.threshold = threshold
    self.silence_duration_ms = silence_duration_ms
    self.sample_rate = sample_rate
    self.frame_duration_ms = frame_duration_ms

    # Calculate frame size in bytes (PCM16 = 2 bytes per sample)
    self.frame_size_samples = int(sample_rate * frame_duration_ms / 1000)
    self.frame_size_bytes = self.frame_size_samples * 2  # 16-bit = 2 bytes

    # State tracking
    self.is_speaking = False
    self.silence_start: Optional[float] = None
    self.audio_buffer = bytearray()

    logger.info(
      f"VAD initialized: threshold={threshold}, silence_duration={silence_duration_ms}ms, sample_rate={sample_rate}Hz"
    )

  def _calculate_energy(self, audio_chunk: bytes) -> float:
    """
    Calculate energy (RMS) of audio chunk.

    Args:
        audio_chunk: Raw PCM16 audio bytes

    Returns:
        Normalized energy level (0.0-1.0)
    """
    try:
      # Convert bytes to numpy array (int16)
      samples = np.frombuffer(audio_chunk, dtype=np.int16)

      # Calculate RMS (Root Mean Square)
      rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

      # Normalize to 0.0-1.0 range
      # Max value for int16 is 32767
      normalized = min(rms / 32767.0, 1.0)

      return normalized

    except Exception as e:
      logger.warning(f"Error calculating energy: {e}")
      return 0.0

  def _is_speech(self, audio_chunk: bytes) -> bool:
    """
    Determine if audio chunk contains speech.

    Args:
        audio_chunk: Raw PCM16 audio bytes

    Returns:
        True if speech detected, False otherwise
    """
    energy = self._calculate_energy(audio_chunk)

    # Simple energy-based VAD
    # Speech typically has energy above threshold
    is_speech = energy > self.threshold

    if is_speech:
      logger.info(f"🎤 Speech detected (energy={energy:.3f})")
    else:
      logger.info(f"🔇 Silence (energy={energy:.3f})")

    return is_speech

  async def process_audio(self, audio_chunk: bytes) -> Tuple[bool, Optional[bytes]]:
    """
    Process audio chunk and detect speech boundaries.

    This method buffers incoming audio and detects when the user
    starts and stops speaking. When silence is detected after speech,
    it returns the complete utterance.

    Args:
        audio_chunk: Raw PCM16 audio bytes

    Returns:
        Tuple of (is_complete_utterance, buffered_audio)
        - is_complete_utterance: True if utterance is complete (ready to transcribe)
        - buffered_audio: Complete audio buffer if utterance is complete, else None
    """
    # Add to buffer
    self.audio_buffer.extend(audio_chunk)

    # Check if this chunk contains speech
    has_speech = self._is_speech(audio_chunk)

    current_time = time.time()

    # State machine for speech detection
    if has_speech:
      # Speech detected
      if not self.is_speaking:
        # Speech started
        logger.info("🎤 Speech started")
        self.is_speaking = True
        self.silence_start = None
      else:
        # Still speaking - reset silence timer
        self.silence_start = None

    else:
      # Silence detected
      if self.is_speaking:
        # We were speaking, now silence
        if self.silence_start is None:
          # Start silence timer
          self.silence_start = current_time
          logger.debug("⏸️  Silence detected, starting timer")
        else:
          # Check if silence duration exceeded
          silence_duration = (current_time - self.silence_start) * 1000  # to ms

          if silence_duration >= self.silence_duration_ms:
            # Silence long enough - utterance complete!
            logger.info(f"✅ Utterance complete ({len(self.audio_buffer)} bytes)")

            # Return complete utterance
            complete_audio = bytes(self.audio_buffer)

            # Reset state
            self.is_speaking = False
            self.silence_start = None
            self.audio_buffer = bytearray()

            return (True, complete_audio)

    # Utterance not complete yet
    return (False, None)

  def reset(self):
    """Reset VAD state and clear buffer."""
    self.is_speaking = False
    self.silence_start = None
    self.audio_buffer = bytearray()
    logger.debug("VAD state reset")

  def get_buffer_size(self) -> int:
    """Get current buffer size in bytes."""
    return len(self.audio_buffer)

  def has_audio(self) -> bool:
    """Check if there's audio in the buffer."""
    return len(self.audio_buffer) > 0


class WebRTCVAD(VoiceActivityDetector):
  """
  Voice Activity Detector using WebRTC VAD library.

  More sophisticated than energy-based VAD, but requires webrtcvad package.
  """

  def __init__(
    self,
    aggressiveness: int = 2,
    silence_duration_ms: int = 500,
    sample_rate: int = 24000,
    frame_duration_ms: int = 30,
  ):
    """
    Initialize WebRTC VAD.

    Args:
        aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
        silence_duration_ms: Silence duration to detect utterance end (ms)
        sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
        frame_duration_ms: Frame duration (must be 10, 20, or 30 ms)
    """
    # WebRTC VAD has specific requirements for sample rate
    if sample_rate not in [8000, 16000, 32000, 48000]:
      logger.warning(
        f"WebRTC VAD requires sample rate of 8000, 16000, 32000, or 48000 Hz. Got {sample_rate}. Falling back to 16000."
      )
      sample_rate = 16000

    if frame_duration_ms not in [10, 20, 30]:
      logger.warning(
        f"WebRTC VAD requires frame duration of 10, 20, or 30 ms. Got {frame_duration_ms}. Falling back to 30."
      )
      frame_duration_ms = 30

    # Initialize parent
    super().__init__(
      threshold=0.5,  # Not used for WebRTC VAD
      silence_duration_ms=silence_duration_ms,
      sample_rate=sample_rate,
      frame_duration_ms=frame_duration_ms,
    )

    try:
      import webrtcvad

      self.vad = webrtcvad.Vad(aggressiveness)
      self.use_webrtc = True
      logger.info(f"WebRTC VAD initialized: aggressiveness={aggressiveness}")
    except ImportError:
      logger.warning("webrtcvad not installed. Falling back to energy-based VAD. Install with: pip install webrtcvad")
      self.use_webrtc = False

  def _is_speech(self, audio_chunk: bytes) -> bool:
    """
    Determine if audio chunk contains speech using WebRTC VAD.

    Args:
        audio_chunk: Raw PCM16 audio bytes

    Returns:
        True if speech detected, False otherwise
    """
    if not self.use_webrtc:
      # Fall back to energy-based detection
      return super()._is_speech(audio_chunk)

    try:
      # WebRTC VAD requires exact frame size
      # Pad or truncate if needed
      required_size = self.frame_size_bytes

      if len(audio_chunk) < required_size:
        # Pad with zeros
        audio_chunk = audio_chunk + b"\x00" * (required_size - len(audio_chunk))
      elif len(audio_chunk) > required_size:
        # Truncate
        audio_chunk = audio_chunk[:required_size]

      # Use WebRTC VAD
      is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)

      if is_speech:
        logger.debug("🎤 Speech detected (WebRTC VAD)")
      else:
        logger.debug("🔇 Silence (WebRTC VAD)")

      return is_speech

    except Exception as e:
      logger.warning(f"WebRTC VAD error: {e}. Falling back to energy-based.")
      return super()._is_speech(audio_chunk)


def create_vad(
  method: str = "energy", threshold: float = 0.5, silence_duration_ms: int = 500, sample_rate: int = 24000, **kwargs
) -> VoiceActivityDetector:
  """
  Factory function to create VAD instances.

  Args:
      method: VAD method ("energy", "webrtc")
      threshold: Energy threshold for energy-based VAD (0.0-1.0)
      silence_duration_ms: Silence duration to detect utterance end
      sample_rate: Audio sample rate in Hz
      **kwargs: Additional method-specific options

  Returns:
      VoiceActivityDetector instance
  """
  method = method.lower()

  if method == "webrtc":
    aggressiveness = kwargs.get("aggressiveness", 2)
    return WebRTCVAD(
      aggressiveness=aggressiveness,
      silence_duration_ms=silence_duration_ms,
      sample_rate=sample_rate,
    )
  elif method == "energy":
    return VoiceActivityDetector(
      threshold=threshold,
      silence_duration_ms=silence_duration_ms,
      sample_rate=sample_rate,
    )
  else:
    raise ValueError(f"Unknown VAD method: {method}. Supported methods: energy, webrtc")

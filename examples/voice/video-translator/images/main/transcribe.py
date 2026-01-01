"""Transcribe audio using speech-to-text models.

This module provides functions to:
- Transcribe audio files with speaker diarization (gpt-4o-transcribe-diarize)
- Format diarized transcription results
- Handle chunked transcription for long videos
"""

from audio import extract_audio_chunk, get_video_duration
from autonomy import Model
from pathlib import Path
from speakers import match_speakers_across_chunks, prefix_speakers_with_chunk


async def transcribe_audio(audio_path: str, use_diarization: bool = True) -> tuple:
    """Transcribe audio file using GPT-4o with diarization or Whisper fallback.

    Args:
      audio_path: Path to the audio file
      use_diarization: If True, use gpt-4o-transcribe-diarize for speaker labels

    Returns:
      Tuple of (transcript_text, raw_result) where raw_result contains segments
    """
    if use_diarization:
        # Use GPT-4o transcribe with diarization for speaker labels
        model = Model("gpt-4o-transcribe-diarize")

        with open(audio_path, "rb") as audio_file:
            # Request diarized_json to get speaker-labeled segments
            # chunking_strategy="auto" is required for audio longer than 30 seconds
            result = await model.speech_to_text(
                audio_file=audio_file,
                language=None,  # Auto-detect language
                model="gpt-4o-transcribe-diarize",
                response_format="diarized_json",
                chunking_strategy="auto",
            )

        # Parse diarized response and extract speaker-labeled text
        transcript = format_diarized_transcript(result)
        return transcript, result
    else:
        # Fall back to basic Whisper
        model = Model("whisper-1")

        with open(audio_path, "rb") as audio_file:
            transcript = await model.speech_to_text(
                audio_file=audio_file,
                language=None,  # Auto-detect language
            )

        return transcript, None


def normalize_speaker_label(speaker: str) -> str:
    """Normalize a speaker label, handling edge cases from diarization.

    Handles:
    - Single letter speakers (A -> Speaker A)
    - @ symbol (gpt-4o-transcribe-diarize uses this when speaker is unknown)
    - Empty/null speakers
    - Generic "Speaker" without identifier

    Note: We intentionally use "Unknown Speaker" instead of falling back to
    the previous speaker. This preserves speaker change signals from the
    diarization model and lets the speaker matcher handle identification.

    Args:
      speaker: Raw speaker label from diarization

    Returns:
      Normalized speaker label
    """
    # Handle None, empty, or whitespace-only
    if not speaker or not speaker.strip():
        return "Unknown Speaker"

    speaker = speaker.strip()

    # Handle @ symbol - gpt-4o-transcribe-diarize uses this when speaker is unidentified
    if speaker == "@":
        return "Unknown Speaker"

    # Handle single letter speakers (A -> Speaker A)
    if len(speaker) == 1:
        if speaker.isalpha():
            return f"Speaker {speaker}"
        else:
            # Other single non-letter chars (like numbers)
            return "Unknown Speaker"

    # Handle bare "Speaker" without identifier
    if speaker == "Speaker":
        return "Unknown Speaker"

    return speaker


def format_diarized_transcript(result) -> str:
    """Format diarized transcription result into readable text with speaker labels.

    Handles edge cases from gpt-4o-transcribe-diarize diarization including:
    - @ symbols when speaker is unknown
    - Missing or empty speaker labels
    - Single-character speaker identifiers

    Args:
      result: The response from gpt-4o-transcribe-diarize (diarized_json format)

    Returns:
      Formatted transcript with speaker labels
    """
    segments = None

    if hasattr(result, "model_dump"):
        raw = result.model_dump()
        segments = raw.get("segments", [])
    elif hasattr(result, "segments"):
        segments = result.segments
    elif isinstance(result, dict):
        segments = result.get("segments", [])

    if segments:
        lines = []
        current_speaker = None
        current_text = []

        for segment in segments:
            if isinstance(segment, dict):
                speaker = segment.get("speaker", "Speaker")
                text = segment.get("text", "").strip()
            else:
                speaker = getattr(segment, "speaker", None) or "Speaker"
                text = getattr(segment, "text", "").strip()

            if not text:
                continue

            speaker = normalize_speaker_label(speaker)

            if speaker != current_speaker:
                if current_speaker and current_text:
                    lines.append(f"{current_speaker}: {' '.join(current_text)}")
                current_speaker = speaker
                current_text = [text]
            else:
                current_text.append(text)

        if current_speaker and current_text:
            lines.append(f"{current_speaker}: {' '.join(current_text)}")

        if lines:
            return "\n\n".join(lines)

    if hasattr(result, "text") and result.text:
        return result.text
    elif isinstance(result, dict) and result.get("text"):
        return result.get("text")

    return str(result)


async def transcribe_audio_chunked(
    video_path: str,
    temp_dir: str,
    use_diarization: bool = True,
    progress_callback=None,
) -> str:
    """Transcribe a video by extracting and processing audio in chunks to avoid OOM.

    Never extracts the full audio - always works in chunks to minimize memory usage.
    Uses gpt-4o-transcribe-diarize for native speaker diarization.

    Chunking strategy (based on the model's 25MB file size limit):
    - Videos producing <20MB audio: Single transcription call for consistent speaker labels
    - Larger videos: Chunked into segments that produce ~15MB audio each

    Speaker matching (for multi-chunk videos):
    - Uses the Speaker Matcher to analyze context clues
    - Agent matches speakers across chunks based on names, roles, and conversation flow
    - Results in unified speaker labels (e.g., "Maria (Host)" instead of "Speaker 1A, 2A, 3A")

    Args:
      video_path: Path to the video file
      temp_dir: Temporary directory for audio chunks
      use_diarization: If True, use gpt-4o-transcribe-diarize for speaker labels
      progress_callback: Optional async callback(message, percent) for progress updates

    Returns:
      Combined transcript text
    """
    total_duration = await get_video_duration(video_path)

    if total_duration <= 0:
        print("Could not determine video duration, trying first chunk...")
        test_audio = Path(temp_dir) / "test.mp3"
        success = await extract_audio_chunk(video_path, str(test_audio), 0, 60)
        if not success or not test_audio.exists() or test_audio.stat().st_size == 0:
            return ""
        test_audio.unlink(missing_ok=True)
        total_duration = 3600  # Assume 1 hour if we can't determine

    print(
        f"Video duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} min)"
    )
    print(f"Using diarization: {use_diarization}")

    MAX_CHUNK_SIZE_MB = 5
    ESTIMATED_MB_PER_MIN = 1.0

    max_chunk_duration_for_size = (MAX_CHUNK_SIZE_MB / ESTIMATED_MB_PER_MIN) * 60

    MAX_CHUNK_DURATION = 300

    chunk_duration = min(max_chunk_duration_for_size, MAX_CHUNK_DURATION)
    if not use_diarization:
        chunk_duration = 600

    needs_chunking = total_duration > chunk_duration
    if not needs_chunking:
        print(
            f"Video fits in single chunk ({total_duration / 60:.1f} min), using single transcription call"
        )
        chunk_duration = total_duration + 60
    else:
        estimated_chunks = int((total_duration + chunk_duration - 1) / chunk_duration)
        print(
            f"Video will be chunked into ~{estimated_chunks} segments of {chunk_duration / 60:.0f} min each"
        )
        print(
            "Note: Speaker labels may differ across chunks for content with many speakers"
        )

    transcripts = []
    current_time = 0
    chunk_num = 0
    total_chunks = max(1, int((total_duration + chunk_duration - 1) / chunk_duration))

    while current_time < total_duration:
        chunk_num += 1
        chunk_audio = Path(temp_dir) / f"chunk_{chunk_num}.mp3"
        chunk_end = min(current_time + chunk_duration, total_duration)

        base_progress = 10
        progress_range = 70
        chunk_progress = base_progress + int(
            (chunk_num - 1) / total_chunks * progress_range
        )

        if progress_callback:
            await progress_callback(
                f"Extracting audio (chunk {chunk_num}/{total_chunks})...",
                chunk_progress,
            )

        print(
            f"Chunk {chunk_num}: extracting audio {current_time / 60:.1f}-{chunk_end / 60:.1f} min..."
        )

        success = await extract_audio_chunk(
            video_path, str(chunk_audio), current_time, chunk_duration
        )

        if success and chunk_audio.exists() and chunk_audio.stat().st_size > 0:
            audio_size_kb = chunk_audio.stat().st_size / 1024
            print(
                f"Chunk {chunk_num}: audio extracted ({audio_size_kb:.0f} KB), transcribing with {'diarization' if use_diarization else 'whisper'}..."
            )

            transcribe_progress = base_progress + int(
                (chunk_num - 0.5) / total_chunks * progress_range
            )
            if progress_callback:
                await progress_callback(
                    f"Transcribing speech (chunk {chunk_num}/{total_chunks})...",
                    transcribe_progress,
                )

            try:
                chunk_transcript, _ = await transcribe_audio(
                    str(chunk_audio), use_diarization=use_diarization
                )

                if chunk_transcript and chunk_transcript.strip():
                    if needs_chunking and use_diarization and total_chunks > 1:
                        chunk_transcript = prefix_speakers_with_chunk(
                            chunk_transcript, chunk_num
                        )

                    transcripts.append(chunk_transcript.strip())
                    print(
                        f"Chunk {chunk_num}: transcribed {len(chunk_transcript)} chars"
                    )
            except Exception as e:
                print(f"Error transcribing chunk {chunk_num} with diarization: {e}")
                if use_diarization:
                    print(f"Chunk {chunk_num}: falling back to whisper...")
                    try:
                        chunk_transcript, _ = await transcribe_audio(
                            str(chunk_audio), use_diarization=False
                        )
                        if chunk_transcript and chunk_transcript.strip():
                            transcripts.append(chunk_transcript.strip())
                            print(
                                f"Chunk {chunk_num}: transcribed {len(chunk_transcript)} chars (whisper fallback)"
                            )
                    except Exception as e2:
                        print(
                            f"Error transcribing chunk {chunk_num} with whisper fallback: {e2}"
                        )
            finally:
                chunk_audio.unlink(missing_ok=True)
        else:
            print(f"Chunk {chunk_num}: no audio or extraction failed, skipping")

        current_time += chunk_duration

    print(f"Transcription complete: {len(transcripts)} chunks processed")

    combined_transcript = "\n\n".join(transcripts)

    if needs_chunking and use_diarization and len(transcripts) > 1:
        print("Using speaker matching agent to unify speaker labels across chunks...")
        if progress_callback:
            await progress_callback("Matching speakers across chunks...", 80)
        try:
            unified_transcript = await match_speakers_across_chunks(combined_transcript)
            if unified_transcript:
                return unified_transcript
            else:
                print(
                    "Speaker matching returned empty result, using original transcript"
                )
        except Exception as e:
            print(f"Speaker matching failed: {e}, using chunk-prefixed labels")

        header = f"[Note: This {total_duration / 60:.0f}-minute video was transcribed in {len(transcripts)} chunks. Speaker labels are prefixed with chunk numbers (e.g., Speaker 1A, Speaker 2A). The same person may have different labels across chunks.]\n\n"
        return header + combined_transcript

    return combined_transcript

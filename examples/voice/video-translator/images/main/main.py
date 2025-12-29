import asyncio
import tempfile
import hashlib
import base64
import json
import shutil
import uuid
import gc
from pathlib import Path
from os import environ, getenv
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks, Form, WebSocket, WebSocketDisconnect, Header
from fastapi.responses import JSONResponse, HTMLResponse
from autonomy import Node, Agent, Model, HttpServer

# Lazy import Box - only if credentials are configured
Box = None

def get_box_client():
  """Lazy load Box client class."""
  global Box
  if Box is None:
    try:
      from box import Box as BoxClass
      Box = BoxClass
    except ImportError:
      return None
  return Box

app = FastAPI()

# Global state
translator_agent = None
box_client = None

# Video file extensions we support
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}

# Store for chunked uploads: upload_id -> {path, filename, expected_size, received_chunks, temp_dir}
chunked_uploads = {}


TRANSLATOR_INSTRUCTIONS = """
You are a professional translator. You will receive transcribed text that may be
in any language. The transcription may already include speaker labels (e.g., "Speaker A:", "Speaker B:").

Your job is to:
1. Identify the source language
2. Translate the text accurately to English
3. PRESERVE any existing speaker labels from the transcription

Output format:
---
Source Language: [detected language]

Translation:
[speaker label if present]: [translated text]
---

CRITICAL RULES:
- PRESERVE existing speaker labels exactly as they appear in the input (e.g., "Speaker A:", "Speaker B:")
- If the input has speaker labels, keep them in the translation
- If the input has NO speaker labels, add "Speaker 1:", "Speaker 2:", etc. based on dialogue turns
- Keep each speaker's dialogue on separate lines with their label
- Be accurate and natural while preserving the meaning and tone

Example input with speaker labels:
Speaker A: Hola, ¿cómo estás?
Speaker B: Muy bien, gracias.

Example output:
---
Source Language: Spanish

Translation:
Speaker A: Hello, how are you?
Speaker B: Very well, thank you.
---

Example input without speaker labels:
Hola, bienvenidos al programa de hoy.

Example output:
---
Source Language: Spanish

Translation:
Speaker 1: Hello, welcome to today's program.
---

If the text is already in English, still format it the same way but note "Source Language: English"
and provide any cleanup/formatting improvements while adding speaker labels.
"""


async def extract_audio(video_path: str, audio_path: str) -> bool:
  """Extract audio from video using ffmpeg with memory-efficient settings."""
  cmd = [
    "ffmpeg",
    "-i", video_path,
    "-vn",
    "-acodec", "libmp3lame",
    "-ar", "16000",
    "-ac", "1",
    "-q:a", "4",
    "-threads", "1",  # Limit threads to reduce memory
    "-y",
    audio_path
  ]

  process = await asyncio.create_subprocess_exec(
    *cmd,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
  )

  _, stderr = await process.communicate()

  if process.returncode != 0:
    print(f"FFmpeg error: {stderr.decode()}")
    return False

  return True


async def get_audio_duration(audio_path: str) -> float:
  """Get audio duration in seconds using ffprobe."""
  cmd = [
    "ffprobe",
    "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    audio_path
  ]

  process = await asyncio.create_subprocess_exec(
    *cmd,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
  )

  stdout, _ = await process.communicate()

  try:
    return float(stdout.decode().strip())
  except:
    return 0.0


async def extract_audio_chunk(video_path: str, audio_path: str, start_time: float, duration: float) -> bool:
  """Extract a chunk of audio from video."""
  cmd = [
    "ffmpeg",
    "-ss", str(start_time),
    "-i", video_path,
    "-t", str(duration),
    "-vn",
    "-acodec", "libmp3lame",
    "-ar", "16000",
    "-ac", "1",
    "-q:a", "4",
    "-threads", "1",
    "-y",
    audio_path
  ]

  process = await asyncio.create_subprocess_exec(
    *cmd,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
  )

  _, stderr = await process.communicate()

  if process.returncode != 0:
    print(f"FFmpeg chunk error: {stderr.decode()}")
    return False

  return True


async def transcribe_audio(audio_path: str, use_diarization: bool = True) -> str:
  """Transcribe audio file using GPT-4o with diarization or Whisper fallback.

  Args:
    audio_path: Path to the audio file
    use_diarization: If True, use gpt-4o-transcribe-diarize for speaker labels

  Returns:
    Transcribed text, with speaker labels if diarization is enabled
  """
  if use_diarization:
    # Use GPT-4o transcribe with diarization for speaker labels
    model = Model("gpt-4o-transcribe-diarize")

    with open(audio_path, "rb") as audio_file:
      # Request diarized_json to get speaker-labeled segments
      result = await model.speech_to_text(
        audio_file=audio_file,
        language=None,  # Auto-detect language
        model="gpt-4o-transcribe-diarize",
        response_format="diarized_json"
      )

    # Parse diarized response - extract speaker-labeled text
    return format_diarized_transcript(result)
  else:
    # Fallback to basic Whisper
    model = Model("whisper-1")

    with open(audio_path, "rb") as audio_file:
      transcript = await model.speech_to_text(
        audio_file=audio_file,
        language=None  # Auto-detect language
      )

    return transcript


def format_diarized_transcript(result) -> str:
  """Format diarized transcription result into readable text with speaker labels.

  Args:
    result: The response from gpt-4o-transcribe-diarize (diarized_json format)

  Returns:
    Formatted transcript with speaker labels
  """
  segments = None

  # Try to get segments from different sources
  if hasattr(result, 'model_dump'):
    # OpenAI SDK response - use model_dump to get raw dict with segments
    raw = result.model_dump()
    segments = raw.get('segments', [])
  elif hasattr(result, 'segments'):
    segments = result.segments
  elif isinstance(result, dict):
    segments = result.get('segments', [])

  if segments:
    # Diarized format with segments
    lines = []
    current_speaker = None
    current_text = []

    for segment in segments:
      # Handle both dict and object segments
      if isinstance(segment, dict):
        speaker = segment.get('speaker', 'Speaker')
        text = segment.get('text', '').strip()
      else:
        speaker = getattr(segment, 'speaker', None) or 'Speaker'
        text = getattr(segment, 'text', '').strip()

      if not text:
        continue

      # Normalize speaker label (A -> Speaker A, etc.)
      if speaker and len(speaker) == 1 and speaker.isalpha():
        speaker = f"Speaker {speaker}"

      if speaker != current_speaker:
        # Output previous speaker's text
        if current_speaker and current_text:
          lines.append(f"{current_speaker}: {' '.join(current_text)}")
        current_speaker = speaker
        current_text = [text]
      else:
        current_text.append(text)

    # Don't forget the last speaker's text
    if current_speaker and current_text:
      lines.append(f"{current_speaker}: {' '.join(current_text)}")

    if lines:
      return '\n'.join(lines)

  # Fallback: if no segments, try to get plain text
  if hasattr(result, 'text') and result.text:
    return result.text
  elif isinstance(result, dict) and result.get('text'):
    return result.get('text')

  # Last resort: convert to string
  return str(result)


async def get_video_duration(video_path: str) -> float:
  """Get video duration in seconds using ffprobe."""
  cmd = [
    "ffprobe",
    "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    video_path
  ]

  process = await asyncio.create_subprocess_exec(
    *cmd,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
  )

  stdout, _ = await process.communicate()

  try:
    return float(stdout.decode().strip())
  except:
    return 0.0


async def transcribe_audio_chunked(video_path: str, temp_dir: str, use_diarization: bool = True) -> str:
  """Transcribe a video by extracting and processing audio in chunks to avoid OOM.

  This never extracts the full audio - always works in chunks to minimize memory usage.
  Uses gpt-4o-transcribe-diarize for native speaker diarization.

  Args:
    video_path: Path to the video file
    temp_dir: Temporary directory for audio chunks
    use_diarization: If True, use gpt-4o-transcribe-diarize for speaker labels
  """
  # Get video duration directly without extracting audio
  total_duration = await get_video_duration(video_path)

  if total_duration <= 0:
    print("Could not determine video duration, trying first chunk...")
    # Try to extract a small chunk to verify video has audio
    test_audio = Path(temp_dir) / "test.mp3"
    success = await extract_audio_chunk(video_path, str(test_audio), 0, 60)
    if not success or not test_audio.exists() or test_audio.stat().st_size == 0:
      return ""
    test_audio.unlink(missing_ok=True)
    total_duration = 3600  # Assume 1 hour if we can't determine

  print(f"Video duration: {total_duration:.1f} seconds ({total_duration/60:.1f} min)")
  print(f"Using diarization: {use_diarization}")

  # gpt-4o-transcribe-diarize has a 1400 second (~23 min) limit
  # Use 20 minute chunks to stay safely under the limit
  chunk_duration = 1200 if use_diarization else 600  # 20 min for diarize, 10 min for whisper
  transcripts = []
  current_time = 0
  chunk_num = 0

  while current_time < total_duration:
    chunk_num += 1
    chunk_audio = Path(temp_dir) / f"chunk_{chunk_num}.mp3"
    chunk_end = min(current_time + chunk_duration, total_duration)

    print(f"Chunk {chunk_num}: extracting audio {current_time/60:.1f}-{chunk_end/60:.1f} min...")

    # Extract just this chunk of audio from the video
    success = await extract_audio_chunk(
      video_path,
      str(chunk_audio),
      current_time,
      chunk_duration
    )

    if success and chunk_audio.exists() and chunk_audio.stat().st_size > 0:
      audio_size_kb = chunk_audio.stat().st_size / 1024
      print(f"Chunk {chunk_num}: audio extracted ({audio_size_kb:.0f} KB), transcribing with {'diarization' if use_diarization else 'whisper'}...")

      try:
        chunk_transcript = await transcribe_audio(str(chunk_audio), use_diarization=use_diarization)
        if chunk_transcript and chunk_transcript.strip():
          transcripts.append(chunk_transcript.strip())
          print(f"Chunk {chunk_num}: transcribed {len(chunk_transcript)} chars")
      except Exception as e:
        print(f"Error transcribing chunk {chunk_num} with diarization: {e}")
        # Fallback to whisper if diarization fails
        if use_diarization:
          print(f"Chunk {chunk_num}: falling back to whisper...")
          try:
            chunk_transcript = await transcribe_audio(str(chunk_audio), use_diarization=False)
            if chunk_transcript and chunk_transcript.strip():
              transcripts.append(chunk_transcript.strip())
              print(f"Chunk {chunk_num}: transcribed {len(chunk_transcript)} chars (whisper fallback)")
          except Exception as e2:
            print(f"Error transcribing chunk {chunk_num} with whisper fallback: {e2}")
      finally:
        # Clean up chunk immediately to save disk space and memory
        chunk_audio.unlink(missing_ok=True)
    else:
      print(f"Chunk {chunk_num}: no audio or extraction failed, skipping")

    current_time += chunk_duration

  print(f"Transcription complete: {len(transcripts)} chunks processed")
  return "\n\n".join(transcripts)


async def translate_to_english(text: str) -> str:
  """Translate transcribed text to English using the agent."""
  global translator_agent

  if not translator_agent:
    raise RuntimeError("Translator agent not initialized")

  response = await translator_agent.send(
    f"Please translate the following transcription to English:\n\n{text}",
    timeout=180
  )

  return response[-1].content.text


async def process_video_file(video_bytes: bytes, filename: str) -> dict:
  """Process a video file from bytes: extract audio, transcribe, translate."""

  with tempfile.TemporaryDirectory() as temp_dir:
    # Save video to temp file
    video_path = Path(temp_dir) / filename
    with open(video_path, "wb") as f:
      f.write(video_bytes)

    # Extract audio
    audio_path = Path(temp_dir) / "audio.mp3"
    success = await extract_audio(str(video_path), str(audio_path))

    if not success:
      return {
        "success": False,
        "error": "Failed to extract audio from video",
        "filename": filename
      }

    # Check if audio file exists and has content
    if not audio_path.exists() or audio_path.stat().st_size == 0:
      return {
        "success": False,
        "error": "No audio track found in video",
        "filename": filename
      }

    # Transcribe
    transcript = await transcribe_audio(str(audio_path))

    if not transcript or not transcript.strip():
      return {
        "success": False,
        "error": "No speech detected in audio",
        "filename": filename
      }

    # Translate
    translation = await translate_to_english(transcript)

    return {
      "success": True,
      "filename": filename,
      "original_transcript": transcript,
      "english_translation": translation,
      "processed_at": datetime.utcnow().isoformat()
    }


async def process_video_file_from_disk(video_path: str, filename: str) -> dict:
  """Process a video file already on disk: extract audio, transcribe, translate."""

  temp_dir = tempfile.mkdtemp()
  try:
    print(f"Processing video from disk: {video_path}")

    # Use chunked transcription to handle long videos
    print("Starting chunked transcription...")
    transcript = await transcribe_audio_chunked(video_path, temp_dir)

    if not transcript or not transcript.strip():
      return {
        "success": False,
        "error": "No speech detected in audio",
        "filename": filename
      }

    print(f"Transcription complete: {len(transcript)} characters. Starting translation...")

    # Translate
    translation = await translate_to_english(transcript)

    print(f"Translation complete: {len(translation)} characters.")

    return {
      "success": True,
      "filename": filename,
      "original_transcript": transcript,
      "english_translation": translation,
      "processed_at": datetime.utcnow().isoformat()
    }
  finally:
    # Clean up temp audio directory
    shutil.rmtree(temp_dir, ignore_errors=True)


def generate_result_markdown(result: dict) -> str:
  """Generate a markdown file with the transcription and translation results."""
  if not result.get("success"):
    return f"# Processing Failed\n\nError: {result.get('error', 'Unknown error')}\n"

  return f"""# Video Transcription & Translation

**Source File:** {result['filename']}
**Processed:** {result['processed_at']}

---

## Original Transcription

{result['original_transcript']}

---

## English Translation

{result['english_translation']}
"""


# =============================================================================
# Box Webhook Endpoint
# =============================================================================

@app.post("/webhook/box")
async def box_webhook(request: Request, background_tasks: BackgroundTasks):
  """
  Handle Box webhook notifications for FILE.UPLOADED events.

  Box sends a POST request when a file is uploaded to the watched folder.
  We verify the webhook, download the video, process it, and upload results.
  """
  global box_client

  # Get webhook payload
  try:
    payload = await request.json()
  except Exception:
    return JSONResponse({"error": "Invalid JSON"}, status_code=400)

  # Box sends a challenge for webhook validation
  if "challenge" in payload:
    return JSONResponse({"challenge": payload["challenge"]})

  # Extract event info
  trigger = payload.get("trigger")
  source = payload.get("source", {})
  file_id = source.get("id")
  file_name = source.get("name", "")
  file_type = source.get("type")

  # Only process file upload events for videos
  if trigger != "FILE.UPLOADED" or file_type != "file":
    return JSONResponse({"status": "ignored", "reason": "Not a file upload event"})

  # Check if it's a video file
  ext = Path(file_name).suffix.lower()
  if ext not in VIDEO_EXTENSIONS:
    return JSONResponse({"status": "ignored", "reason": "Not a video file"})

  # Process in background to respond quickly to Box
  background_tasks.add_task(process_box_video, file_id, file_name)

  return JSONResponse({"status": "processing", "file_id": file_id, "filename": file_name})


async def process_box_video(file_id: str, filename: str):
  """Background task to process a video from Box."""
  global box_client

  if not box_client:
    BoxClass = get_box_client()
    if BoxClass:
      box_client = BoxClass()
    else:
      print(f"Box SDK not available, cannot process {filename}")
      return

  try:
    print(f"Processing video from Box: {filename} (ID: {file_id})")

    # Download the video
    video_bytes = await box_client.download_file(file_id)

    # Process it
    result = await process_video_file(video_bytes, filename)

    # Generate result markdown
    result_md = generate_result_markdown(result)

    # Upload result back to Box (same folder as original)
    parent_folder_id = await box_client.get_file_parent_folder(file_id)
    result_filename = Path(filename).stem + "_translation.md"

    await box_client.upload_file(
      folder_id=parent_folder_id,
      filename=result_filename,
      content=result_md.encode("utf-8")
    )

    print(f"Uploaded translation result: {result_filename}")

  except Exception as e:
    print(f"Error processing Box video {filename}: {e}")


# =============================================================================
# Direct Upload Endpoint (Demo UI)
# =============================================================================

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
  """
  Direct video upload endpoint for the demo UI.

  Accepts a video file, streams it to disk, processes it, and returns the results.
  Uses streaming to avoid loading entire file into memory (prevents OOM on large files).
  """
  # Validate file type
  ext = Path(file.filename or "video.mp4").suffix.lower()
  if ext not in VIDEO_EXTENSIONS:
    return JSONResponse(
      {"success": False, "error": f"Unsupported file type: {ext}"},
      status_code=400
    )

  # Stream file to disk to avoid OOM on large files
  temp_dir = tempfile.mkdtemp()
  try:
    filename = file.filename or "uploaded_video.mp4"
    video_path = Path(temp_dir) / filename
    chunk_size = 1024 * 1024  # 1MB chunks

    print(f"Streaming upload to disk: {filename}")
    with open(video_path, "wb") as f:
      while True:
        chunk = await file.read(chunk_size)
        if not chunk:
          break
        f.write(chunk)

    file_size = video_path.stat().st_size
    print(f"Upload complete: {filename}, {file_size / (1024*1024):.1f} MB")

    # Process the video from disk
    result = await process_video_file_from_disk(str(video_path), filename)
  finally:
    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    gc.collect()

  return JSONResponse(result)


# =============================================================================
# Chunked HTTP Upload Endpoints (for large files - more reliable than WebSocket)
# =============================================================================

# Store for chunked uploads: upload_id -> {path, filename, expected_size, temp_dir, file_handle, lock}
active_uploads = {}
import threading

@app.post("/upload/start")
async def start_chunked_upload(request: Request):
  """
  Start a chunked upload session for parallel chunk uploads.

  Request body: {"filename": "video.mp4", "size": 12345678}
  Returns: {"upload_id": "uuid", "chunk_size": 8388608}
  """
  try:
    data = await request.json()
  except:
    return JSONResponse({"error": "Invalid JSON"}, status_code=400)

  filename = data.get("filename", "uploaded_video.mp4")
  expected_size = data.get("size", 0)

  # Validate file type
  ext = Path(filename).suffix.lower()
  if ext not in VIDEO_EXTENSIONS:
    return JSONResponse({"error": f"Unsupported file type: {ext}"}, status_code=400)

  # Create upload session
  upload_id = str(uuid.uuid4())
  temp_dir = tempfile.mkdtemp()
  video_path = Path(temp_dir) / filename

  chunk_size = 1 * 1024 * 1024  # 1MB chunks (safely under proxy limit)
  total_chunks = (expected_size + chunk_size - 1) // chunk_size

  # Pre-allocate file with expected size for parallel writes
  with open(video_path, "wb") as f:
    f.seek(expected_size - 1)
    f.write(b'\0')

  active_uploads[upload_id] = {
    "path": str(video_path),
    "filename": filename,
    "expected_size": expected_size,
    "received_size": 0,
    "temp_dir": temp_dir,
    "chunk_size": chunk_size,
    "total_chunks": total_chunks,
    "chunks_received": set(),
    "lock": threading.Lock(),
  }

  print(f"Started parallel chunked upload {upload_id}: {filename}, {expected_size} bytes, {total_chunks} chunks of {chunk_size} bytes")

  return JSONResponse({
    "upload_id": upload_id,
    "chunk_size": chunk_size,
    "total_chunks": total_chunks,
  })


@app.post("/upload/chunk/{upload_id}/{chunk_index}")
async def upload_chunk(upload_id: str, chunk_index: int, request: Request):
  """
  Upload a single chunk at specific index. Supports parallel uploads.
  Chunk data is sent as raw bytes in request body.
  Uses streaming read to avoid buffering entire request in memory.
  """
  if upload_id not in active_uploads:
    return JSONResponse({"error": "Upload session not found or expired"}, status_code=404)

  upload = active_uploads[upload_id]

  # Check if chunk already received
  with upload["lock"]:
    if chunk_index in upload["chunks_received"]:
      return JSONResponse({"status": "ok", "message": "Already received"})

  # Stream body directly to file to avoid memory buffering
  offset = chunk_index * upload["chunk_size"]
  chunk_len = 0

  try:
    with open(upload["path"], "r+b") as f:
      f.seek(offset)
      # Stream the request body in small pieces
      async for chunk in request.stream():
        f.write(chunk)
        chunk_len += len(chunk)
      f.flush()
  except Exception as e:
    return JSONResponse({"error": f"Failed to write chunk: {e}"}, status_code=500)

  if chunk_len == 0:
    return JSONResponse({"error": "Empty chunk"}, status_code=400)

  # Update tracking with lock for thread safety
  with upload["lock"]:
    upload["chunks_received"].add(chunk_index)
    upload["received_size"] += chunk_len
    chunks_done = len(upload["chunks_received"])

  # Force garbage collection frequently to prevent memory buildup
  if chunks_done % 10 == 0:
    gc.collect()

  progress = int((chunks_done / upload["total_chunks"]) * 100) if upload["total_chunks"] > 0 else 0

  # Log progress every 10%
  if progress % 10 == 0 and upload.get("last_logged_progress", -1) < progress:
    upload["last_logged_progress"] = progress
    print(f"Upload {upload_id}: {progress}% ({chunks_done}/{upload['total_chunks']} chunks)")

  return JSONResponse({
    "status": "ok",
    "chunks_done": chunks_done,
    "progress": progress,
  })


@app.post("/upload/finish/{upload_id}")
async def finish_chunked_upload(upload_id: str):
  """
  Complete the chunked upload and start processing.
  Verifies all chunks were received before processing.
  """
  if upload_id not in active_uploads:
    return JSONResponse({"error": "Upload session not found"}, status_code=404)

  upload = active_uploads[upload_id]

  # Verify all chunks received
  with upload["lock"]:
    chunks_received = len(upload["chunks_received"])
    total_chunks = upload["total_chunks"]

  if chunks_received < total_chunks:
    missing = set(range(total_chunks)) - upload["chunks_received"]
    missing_list = sorted(list(missing))[:10]  # First 10 missing
    return JSONResponse({
      "error": f"Missing {total_chunks - chunks_received} chunks",
      "missing_sample": missing_list,
    }, status_code=400)

  print(f"Upload {upload_id} complete: {upload['filename']}, {chunks_received} chunks. Processing...")

  # Process the video
  try:
    result = await process_video_file_from_disk(upload["path"], upload["filename"])
  except Exception as e:
    print(f"Error processing upload {upload_id}: {e}")
    result = {"success": False, "error": str(e), "filename": upload["filename"]}
  finally:
    # Clean up
    shutil.rmtree(upload["temp_dir"], ignore_errors=True)
    del active_uploads[upload_id]
    gc.collect()

  return JSONResponse(result)


@app.delete("/upload/{upload_id}")
async def cancel_upload(upload_id: str):
  """
  Cancel and clean up a chunked upload.
  """
  if upload_id not in active_uploads:
    return JSONResponse({"error": "Upload session not found"}, status_code=404)

  upload = active_uploads[upload_id]

  # Clean up temp directory
  shutil.rmtree(upload["temp_dir"], ignore_errors=True)
  del active_uploads[upload_id]

  print(f"Upload {upload_id} cancelled")
  return JSONResponse({"status": "cancelled"})


# =============================================================================
# Chunked Upload Endpoints (for large files)
# =============================================================================

@app.post("/upload/init")
async def init_chunked_upload(request: Request):
  """
  Initialize a chunked upload session.

  Request body: {"filename": "video.mp4", "size": 12345678, "chunk_size": 52428800}
  Returns: {"upload_id": "uuid", "chunk_size": 52428800}
  """
  try:
    data = await request.json()
  except:
    return JSONResponse({"error": "Invalid JSON"}, status_code=400)

  filename = data.get("filename", "uploaded_video.mp4")
  expected_size = data.get("size", 0)
  chunk_size = data.get("chunk_size", 50 * 1024 * 1024)  # Default 50MB chunks

  # Validate file type
  ext = Path(filename).suffix.lower()
  if ext not in VIDEO_EXTENSIONS:
    return JSONResponse({"error": f"Unsupported file type: {ext}"}, status_code=400)

  # Create upload session
  upload_id = str(uuid.uuid4())
  temp_dir = tempfile.mkdtemp()
  video_path = Path(temp_dir) / filename

  # Create empty file
  video_path.touch()

  chunked_uploads[upload_id] = {
    "path": str(video_path),
    "filename": filename,
    "expected_size": expected_size,
    "received_size": 0,
    "received_chunks": set(),
    "temp_dir": temp_dir,
    "chunk_size": chunk_size,
  }

  total_chunks = (expected_size + chunk_size - 1) // chunk_size
  print(f"Initialized chunked upload {upload_id}: {filename}, {expected_size} bytes, {total_chunks} chunks")

  return JSONResponse({
    "upload_id": upload_id,
    "chunk_size": chunk_size,
    "total_chunks": total_chunks,
  })


@app.post("/upload/chunk/{upload_id}/{chunk_index}")
async def upload_chunk(upload_id: str, chunk_index: int, request: Request):
  """
  Upload a single chunk.

  The chunk data should be sent as raw bytes in the request body.
  """
  if upload_id not in chunked_uploads:
    return JSONResponse({"error": "Upload session not found"}, status_code=404)

  upload = chunked_uploads[upload_id]

  # Check if chunk already received
  if chunk_index in upload["received_chunks"]:
    return JSONResponse({
      "status": "already_received",
      "chunk_index": chunk_index,
      "received_size": upload["received_size"],
    })

  # Read chunk data
  chunk_data = await request.body()
  chunk_size = len(chunk_data)

  if chunk_size == 0:
    return JSONResponse({"error": "Empty chunk"}, status_code=400)

  # Write chunk to file at correct offset
  offset = chunk_index * upload["chunk_size"]

  try:
    with open(upload["path"], "r+b") as f:
      f.seek(offset)
      f.write(chunk_data)
  except Exception as e:
    return JSONResponse({"error": f"Failed to write chunk: {e}"}, status_code=500)

  # Update tracking
  upload["received_chunks"].add(chunk_index)
  upload["received_size"] += chunk_size

  progress = int((upload["received_size"] / upload["expected_size"]) * 100) if upload["expected_size"] > 0 else 0

  # Log progress every 10%
  if progress % 10 == 0 and progress > 0:
    print(f"Upload {upload_id}: {progress}% ({upload['received_size']}/{upload['expected_size']} bytes)")

  return JSONResponse({
    "status": "ok",
    "chunk_index": chunk_index,
    "received_size": upload["received_size"],
    "progress": progress,
  })


@app.post("/upload/complete/{upload_id}")
async def complete_chunked_upload(upload_id: str, background_tasks: BackgroundTasks):
  """
  Complete the chunked upload and start processing.
  """
  if upload_id not in chunked_uploads:
    return JSONResponse({"error": "Upload session not found"}, status_code=404)

  upload = chunked_uploads[upload_id]

  # Verify all chunks received
  expected_chunks = (upload["expected_size"] + upload["chunk_size"] - 1) // upload["chunk_size"]
  received_chunks = len(upload["received_chunks"])

  if received_chunks < expected_chunks:
    missing = set(range(expected_chunks)) - upload["received_chunks"]
    return JSONResponse({
      "error": "Missing chunks",
      "missing_chunks": list(missing)[:20],  # Return first 20 missing
      "expected": expected_chunks,
      "received": received_chunks,
    }, status_code=400)

  print(f"Upload {upload_id} complete: {upload['filename']}, {upload['received_size']} bytes. Processing...")

  # Process the video
  try:
    result = await process_video_file_from_disk(upload["path"], upload["filename"])
  except Exception as e:
    print(f"Error processing upload {upload_id}: {e}")
    result = {"success": False, "error": str(e), "filename": upload["filename"]}
  finally:
    # Clean up
    shutil.rmtree(upload["temp_dir"], ignore_errors=True)
    del chunked_uploads[upload_id]

  return JSONResponse(result)


@app.get("/upload/status/{upload_id}")
async def get_upload_status(upload_id: str):
  """
  Get the status of a chunked upload.
  """
  if upload_id not in chunked_uploads:
    return JSONResponse({"error": "Upload session not found"}, status_code=404)

  upload = chunked_uploads[upload_id]
  expected_chunks = (upload["expected_size"] + upload["chunk_size"] - 1) // upload["chunk_size"]

  return JSONResponse({
    "upload_id": upload_id,
    "filename": upload["filename"],
    "expected_size": upload["expected_size"],
    "received_size": upload["received_size"],
    "expected_chunks": expected_chunks,
    "received_chunks": len(upload["received_chunks"]),
    "progress": int((upload["received_size"] / upload["expected_size"]) * 100) if upload["expected_size"] > 0 else 0,
  })


@app.delete("/upload/{upload_id}")
async def cancel_chunked_upload(upload_id: str):
  """
  Cancel and clean up a chunked upload.
  """
  if upload_id not in chunked_uploads:
    return JSONResponse({"error": "Upload session not found"}, status_code=404)

  upload = chunked_uploads[upload_id]
  shutil.rmtree(upload["temp_dir"], ignore_errors=True)
  del chunked_uploads[upload_id]

  return JSONResponse({"status": "cancelled"})


# =============================================================================
# WebSocket Upload Endpoint (for large files)
# =============================================================================

# Raw ASGI WebSocket handler to bypass Starlette buffering
async def raw_websocket_upload(scope, receive, send):
  """
  Raw ASGI WebSocket handler for chunked video uploads.
  Bypasses Starlette/FastAPI buffering for better memory efficiency.
  """
  # Accept the WebSocket connection
  await send({"type": "websocket.accept"})
  print("Raw WebSocket upload connection accepted")

  gc.collect()

  filename = "uploaded_video.mp4"
  expected_size = 0
  received_size = 0
  temp_dir = None
  video_file = None
  video_path = None
  total_chunks = 0
  chunks_received = 0
  last_progress_log = 0

  async def send_json(data):
    await send({
      "type": "websocket.send",
      "text": json.dumps(data)
    })

  try:
    while True:
      message = await receive()

      if message["type"] == "websocket.disconnect":
        print(f"WS disconnected: {filename} ({chunks_received}/{total_chunks} chunks)")
        break

      if message["type"] == "websocket.receive":
        if "text" in message and message["text"]:
          data = json.loads(message["text"])
          msg_type = data.get("type")

          if msg_type == "ping":
            await send_json({"type": "pong"})
            continue

          if msg_type == "start":
            filename = data.get("filename", "uploaded_video.mp4")
            expected_size = data.get("size", 0)
            total_chunks = data.get("total_chunks", 0)
            received_size = 0
            chunks_received = 0
            last_progress_log = 0

            ext = Path(filename).suffix.lower()
            if ext not in VIDEO_EXTENSIONS:
              await send_json({"type": "error", "error": f"Unsupported file type: {ext}"})
              break

            temp_dir = tempfile.mkdtemp()
            video_path = Path(temp_dir) / filename
            video_file = open(video_path, "wb")

            print(f"WS upload started: {filename}, {expected_size} bytes, {total_chunks} chunks")
            await send_json({"type": "ready"})

          elif msg_type == "end":
            if video_file:
              video_file.close()
              video_file = None

            print(f"WS upload complete: {filename}, {received_size} bytes, {chunks_received} chunks. Processing...")
            await send_json({"type": "status", "message": "Processing video..."})

            result = await process_video_file_from_disk(str(video_path), filename)

            if temp_dir:
              shutil.rmtree(temp_dir, ignore_errors=True)
              temp_dir = None

            print(f"Processing done: {filename}, success={result.get('success', False)}")
            await send_json({"type": "result", **result})
            break

        elif "bytes" in message and message["bytes"]:
          chunk_data = message["bytes"]
          chunk_len = len(chunk_data)

          if video_file:
            video_file.write(chunk_data)
            # Don't flush every chunk - let OS handle buffering
            if chunks_received % 10 == 0:
              video_file.flush()

          # Immediately clear reference
          del chunk_data
          del message

          received_size += chunk_len
          chunks_received += 1

          # GC every 100 chunks
          if chunks_received % 100 == 0:
            gc.collect()

          # Send ack
          await send_json({"type": "chunk_ack"})

          # Log progress every 10%
          progress = int((chunks_received / total_chunks) * 100) if total_chunks > 0 else 0
          if progress >= last_progress_log + 10:
            last_progress_log = progress
            print(f"WS upload {filename}: {progress}% ({chunks_received}/{total_chunks} chunks)")

  except Exception as e:
    print(f"WS error: {filename}: {e}")
    try:
      await send_json({"type": "error", "error": str(e)})
    except:
      pass
  finally:
    if video_file:
      try:
        video_file.close()
      except:
        pass
    if temp_dir:
      shutil.rmtree(temp_dir, ignore_errors=True)
    try:
      await send({"type": "websocket.close"})
    except:
      pass


# Mount raw ASGI handler for /ws/upload
from starlette.routing import Route, WebSocketRoute

@app.websocket("/ws/upload")
async def websocket_upload_wrapper(websocket: WebSocket):
  """Wrapper that delegates to raw ASGI handler"""
  await raw_websocket_upload(
    websocket.scope,
    websocket.receive,
    websocket.send
  )


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health")
async def health():
  """Health check endpoint."""
  return {"status": "healthy", "agent_ready": translator_agent is not None}


@app.get("/debug/disk")
async def debug_disk():
  """Check available disk space for uploads."""
  # Check temp directory space
  temp_dir = tempfile.gettempdir()
  total, used, free = shutil.disk_usage(temp_dir)

  # Convert to human-readable
  def human_readable(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
      if size_bytes < 1024:
        return f"{size_bytes:.2f} {unit}"
      size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

  return {
    "temp_directory": temp_dir,
    "total": human_readable(total),
    "used": human_readable(used),
    "free": human_readable(free),
    "total_bytes": total,
    "used_bytes": used,
    "free_bytes": free,
    "max_recommended_upload_mb": int(free / (1024 * 1024) * 0.8),  # 80% of free space
  }


@app.get("/webhook/status")
async def webhook_status():
  """Check webhook configuration status."""
  return {
    "webhook_url": "/webhook/box",
    "supported_triggers": ["FILE.UPLOADED"],
    "supported_extensions": list(VIDEO_EXTENSIONS),
    "box_configured": all([
      environ.get("BOX_CLIENT_ID"),
      environ.get("BOX_CLIENT_SECRET"),
      environ.get("BOX_ENTERPRISE_ID")
    ])
  }


# =============================================================================
# Application Startup
# =============================================================================

async def main(node: Node):
  global translator_agent, box_client

  # Start the translation agent
  translator_agent = await Agent.start(
    node=node,
    name="translator",
    instructions=TRANSLATOR_INSTRUCTIONS,
    model=Model("claude-sonnet-4-v1"),
  )

  # Initialize Box client if credentials are available
  box_configured = all([
    getenv("BOX_CLIENT_ID"),
    getenv("BOX_CLIENT_SECRET"),
    getenv("BOX_ENTERPRISE_ID")
  ])

  if box_configured:
    try:
      BoxClass = get_box_client()
      if BoxClass:
        box_client = BoxClass()
        print("Box client initialized")

      # Optionally set up webhook on startup
      folder_id = getenv("BOX_FOLDER_ID")
      webhook_url = getenv("WEBHOOK_BASE_URL")
      if folder_id and webhook_url:
        try:
          await box_client.ensure_webhook(folder_id)
          print(f"Box webhook configured for folder: {folder_id}")
        except Exception as e:
          print(f"Warning: Could not set up Box webhook: {e}")
    except Exception as e:
      print(f"Warning: Could not initialize Box client: {e}")
      box_client = None
  else:
    print("Box credentials not configured - Box integration disabled")
    print("Demo UI upload is still available at /")

  print("Video translator ready!")


Node.start(main, http_server=HttpServer(app=app))

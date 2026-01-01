"""Serve the Video Translator FastAPI application.

This is the main entry point for the video translator service.
Provides HTTP and WebSocket endpoints for video upload, transcription,
translation, and Box integration.
"""

import box
import shutil
import tempfile
import translate

from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse
from autonomy import Node, HttpServer
from transcribe import transcribe_audio_chunked
from jobs import get_queue_status
from upload import handle_websocket_upload, VIDEO_EXTENSIONS


app = FastAPI()


# =============================================================================
# Video Processing Functions
# =============================================================================


async def process_video(path: str, filename: str, progress_callback=None) -> dict:
    """Process a video file: extract audio, transcribe, and translate.

    Args:
      path: Path to the video file
      filename: Original filename for display
      progress_callback: Optional async callback(message, percent) for progress updates

    Returns:
      Dictionary with success status, transcript, translation, and metadata
    """
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Processing video: {path}")

        # Use chunked transcription to handle long videos
        print("Starting chunked transcription...")
        transcript = await transcribe_audio_chunked(
            path,
            temp_dir,
            progress_callback=progress_callback,
        )

        if not transcript or not transcript.strip():
            return {
                "success": False,
                "error": "No speech detected in audio",
                "filename": filename,
            }

        print(
            f"Transcription complete: {len(transcript)} characters. Starting translation..."
        )

        if progress_callback:
            await progress_callback("Translating to English...", 85)
        translation = await translate.translate_to_english(
            transcript, progress_callback=progress_callback
        )

        print(f"Translation complete: {len(translation)} characters.")

        return {
            "success": True,
            "filename": filename,
            "original_transcript": transcript,
            "english_translation": translation,
            "processed_at": datetime.utcnow().isoformat(),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def generate_result_markdown(result: dict) -> str:
    """Generate a markdown file with transcription and translation results.

    Args:
      result: Processing result dictionary

    Returns:
      Formatted markdown string
    """
    if not result.get("success"):
        return f"# Processing Failed\n\nError: {result.get('error', 'Unknown error')}\n"

    return f"""# Video Transcription & Translation

**Source File:** {result["filename"]}
**Processed:** {result["processed_at"]}

---

## Original Transcription

{result["original_transcript"]}

---

## English Translation

{result["english_translation"]}
"""


# =============================================================================
# Box Webhook Endpoint
# =============================================================================


@app.post("/webhook/box")
async def box_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Box webhook notifications for FILE.UPLOADED events.

    Box sends a POST request when a file is uploaded to the watched folder.
    Verifies the webhook, downloads the video, processes it, and uploads results.

    Args:
      request: FastAPI request object
      background_tasks: FastAPI background tasks for async processing

    Returns:
      JSON response with processing status
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # Validate payload and determine action
    result = box.validate_webhook_payload(payload, VIDEO_EXTENSIONS)

    if result["action"] == "challenge":
        return JSONResponse({"challenge": result["challenge"]})

    if result["action"] == "ignore":
        return JSONResponse(result["response"])

    # action == "process"
    file_id = result["file_id"]
    filename = result["filename"]

    background_tasks.add_task(
        box.process_video,
        file_id,
        filename,
        process_video,
        generate_result_markdown,
    )

    return JSONResponse(
        {"status": "processing", "file_id": file_id, "filename": filename}
    )


# =============================================================================
# WebSocket Upload Endpoint
# =============================================================================


@app.websocket("/ws/upload")
async def websocket_upload(websocket: WebSocket):
    """Handle chunked video uploads via WebSocket."""
    await handle_websocket_upload(
        websocket.scope, websocket.receive, websocket.send, process_video
    )


# =============================================================================
# Queue Status Endpoint
# =============================================================================


@app.get("/queue/status")
async def queue_status():
    """Return current processing queue status."""
    return get_queue_status()


# =============================================================================
# Health Endpoint
# =============================================================================


@app.get("/health")
async def health():
    """Return health check status."""
    return {
        "status": "healthy",
        "translator": translate.is_ready(),
        "box": box.is_configured(),
    }


# =============================================================================
# Application Startup
# =============================================================================


async def main(node: Node):
    """Initialize the video translator service.

    Starts the translator agent and optionally configures Box integration.

    Args:
      node: Autonomy node instance
    """
    await translate.initialize(node)
    await box.initialize()

    print("Video translator ready!")


Node.start(main, http_server=HttpServer(app=app))

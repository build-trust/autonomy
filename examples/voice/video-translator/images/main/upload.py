"""Handle WebSocket uploads for video files.

This module provides a raw ASGI WebSocket handler that receives chunked video
uploads, bypassing Starlette/FastAPI buffering for better memory efficiency.
"""

import asyncio
import gc
import json
import shutil
import tempfile
import uuid

from datetime import datetime
from pathlib import Path


# Supported video file extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


async def handle_websocket_upload(scope, receive, send, process_video_func):
    """Handle chunked video uploads via raw ASGI WebSocket.

    Bypasses Starlette/FastAPI buffering to improve memory efficiency.

    Args:
      scope: ASGI scope
      receive: ASGI receive callable
      send: ASGI send callable
      process_video_func: Async function that processes the video file (path, filename) -> result dict
    """
    import jobs as jobs_module
    from jobs import ProcessingJob, processing_lock, job_queue, get_job_position

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
        await send({"type": "websocket.send", "text": json.dumps(data)})

    try:
        while True:
            message = await receive()

            if message["type"] == "websocket.disconnect":
                print(
                    f"WS disconnected: {filename} ({chunks_received}/{total_chunks} chunks)"
                )
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
                            await send_json(
                                {
                                    "type": "error",
                                    "error": f"Unsupported file type: {ext}",
                                }
                            )
                            break

                        temp_dir = tempfile.mkdtemp()
                        video_path = Path(temp_dir) / filename
                        video_file = open(video_path, "wb")

                        print(
                            f"WS upload started: {filename}, {expected_size} bytes, {total_chunks} chunks"
                        )
                        await send_json({"type": "ready"})

                    elif msg_type == "end":
                        if video_file:
                            video_file.close()
                            video_file = None

                        print(
                            f"WS upload complete: {filename}, {received_size} bytes, {chunks_received} chunks."
                        )

                        job_id = str(uuid.uuid4())
                        job = ProcessingJob(
                            id=job_id,
                            filename=filename,
                            status="queued",
                            progress=0,
                            queued_at=datetime.utcnow(),
                        )

                        async with processing_lock:
                            if jobs_module.current_job is not None:
                                job_queue.append(job)
                                position = len(job_queue)
                                print(f"Job {job_id} queued at position {position}")
                                await send_json(
                                    {
                                        "type": "queued",
                                        "position": position,
                                        "message": f"Another video is being processed. You are #{position} in queue.",
                                    }
                                )
                            else:
                                jobs_module.current_job = job
                                job.mark_started()

                        if job.status == "queued":
                            while True:
                                await asyncio.sleep(2)

                                async with processing_lock:
                                    if (
                                        job_queue
                                        and job_queue[0].id == job_id
                                        and jobs_module.current_job is None
                                    ):
                                        job_queue.popleft()
                                        jobs_module.current_job = job
                                        job.mark_started()
                                        break

                                    position = get_job_position(job_id)
                                    if position:
                                        await send_json(
                                            {
                                                "type": "queue_update",
                                                "position": position,
                                                "message": f"You are #{position} in queue.",
                                            }
                                        )
                                    else:
                                        break

                        await send_json(
                            {
                                "type": "status",
                                "message": "Processing video...",
                                "percent": 5,
                            }
                        )

                        async def progress_callback(message: str, percent: int):
                            await send_json(
                                {
                                    "type": "status",
                                    "message": message,
                                    "percent": percent,
                                }
                            )

                        try:
                            result = await process_video_func(
                                str(video_path),
                                filename,
                                progress_callback=progress_callback,
                            )
                            job.mark_completed(
                                success=result.get("success", False),
                                error=result.get("error"),
                            )
                        except Exception as e:
                            job.mark_completed(success=False, error=str(e))
                            result = {
                                "success": False,
                                "error": str(e),
                                "filename": filename,
                            }
                        finally:
                            # Release the processing slot
                            async with processing_lock:
                                jobs_module.current_job = None

                        if temp_dir:
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            temp_dir = None

                        print(
                            f"Processing done: {filename}, success={result.get('success', False)}"
                        )
                        await send_json({"type": "result", **result})
                        break

                elif "bytes" in message and message["bytes"]:
                    chunk_data = message["bytes"]
                    chunk_len = len(chunk_data)

                    if video_file:
                        video_file.write(chunk_data)
                        if chunks_received % 10 == 0:
                            video_file.flush()

                    del chunk_data
                    del message

                    received_size += chunk_len
                    chunks_received += 1

                    if chunks_received % 100 == 0:
                        gc.collect()

                    await send_json({"type": "chunk_ack"})

                    progress = (
                        int((chunks_received / total_chunks) * 100)
                        if total_chunks > 0
                        else 0
                    )
                    if progress >= last_progress_log + 10:
                        last_progress_log = progress
                        print(
                            f"WS upload {filename}: {progress}% ({chunks_received}/{total_chunks} chunks)"
                        )

    except Exception as e:
        print(f"WS error: {filename}: {e}")
        try:
            await send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        if video_file:
            try:
                video_file.close()
            except Exception:
                pass
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            await send({"type": "websocket.close"})
        except Exception:
            pass

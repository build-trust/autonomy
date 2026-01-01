"""Extract audio from video files using ffmpeg.

This module provides functions to extract audio from video files
and to get video/audio duration information.
"""

import asyncio


async def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio from video using ffmpeg with memory-efficient settings.

    Args:
      video_path: Path to the input video file
      audio_path: Path to save the extracted audio (MP3 format)

    Returns:
      True if extraction succeeded, False otherwise
    """
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-q:a",
        "4",
        "-threads",
        "1",  # Limit threads to reduce memory
        "-y",
        audio_path,
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    _, stderr = await process.communicate()

    if process.returncode != 0:
        print(f"FFmpeg error: {stderr.decode()}")
        return False

    return True


async def extract_audio_chunk(
    video_path: str, audio_path: str, start_time: float, duration: float
) -> bool:
    """Extract a chunk of audio from video.

    Args:
      video_path: Path to the input video file
      audio_path: Path to save the extracted audio chunk (MP3 format)
      start_time: Start time in seconds
      duration: Duration in seconds to extract

    Returns:
      True if extraction succeeded, False otherwise
    """
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-q:a",
        "4",
        "-threads",
        "1",
        "-y",
        audio_path,
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    _, stderr = await process.communicate()

    if process.returncode != 0:
        print(f"FFmpeg chunk error: {stderr.decode()}")
        return False

    return True


async def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe.

    Args:
      video_path: Path to the video file

    Returns:
      Duration in seconds, or 0.0 if duration cannot be determined
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, _ = await process.communicate()

    try:
        return float(stdout.decode().strip())
    except Exception:
        return 0.0

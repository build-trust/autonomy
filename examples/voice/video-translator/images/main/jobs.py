"""Manage processing queue for video translation.

This module provides the ProcessingJob dataclass and queue state management
to ensure only one video processes at a time.
"""

import asyncio

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ProcessingJob:
    """Represent a video processing job in the queue."""

    id: str
    filename: str
    status: str  # "queued", "processing", "transcribing", "translating", "complete", "error"
    progress: int  # 0-100
    queued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert job to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "progress": self.progress,
            "queued_at": self.queued_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error": self.error,
        }

    def mark_started(self) -> None:
        """Mark job as started and record timestamp."""
        self.status = "processing"
        self.started_at = datetime.utcnow()

    def mark_completed(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark job as completed and record timestamp."""
        self.status = "complete" if success else "error"
        self.progress = 100
        self.completed_at = datetime.utcnow()
        if error:
            self.error = error

    def update_progress(self, progress: int, status: Optional[str] = None) -> None:
        """Update job progress and optionally status."""
        self.progress = max(0, min(100, progress))
        if status:
            self.status = status


# Global queue state
processing_lock = asyncio.Lock()
current_job: Optional[ProcessingJob] = None
job_queue: deque[ProcessingJob] = deque()


def get_queue_status() -> dict:
    """Return current processing queue status.

    Returns:
      Dictionary with current job info, queue length, and estimated wait time
    """
    return {
        "current_job": current_job.to_dict() if current_job else None,
        "queue_length": len(job_queue),
        "queue": [j.to_dict() for j in job_queue],
        "estimated_wait_minutes": len(job_queue) * 10,  # Estimate 10 min per video
    }


def get_job_position(job_id: str) -> Optional[int]:
    """Find position of a job in the queue (1-indexed).

    Returns:
      Position in queue (1 = next up), or None if not found
    """
    for i, job in enumerate(job_queue):
        if job.id == job_id:
            return i + 1
    return None


def is_queue_empty() -> bool:
    """Check if the queue has no waiting jobs."""
    return len(job_queue) == 0


def is_processing() -> bool:
    """Check if a job is currently being processed."""
    return current_job is not None

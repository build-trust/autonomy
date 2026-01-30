"""SRE Incident Diagnosis - Main Entry Point

Provides HTTP API for incident diagnosis with human-in-the-loop credential approval.
"""

from autonomy import HttpServer, Node
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime


# === Request/Response Models ===

class DiagnoseRequest(BaseModel):
  problem: str
  environment: str = "prod"
  context: Optional[dict] = None


class DiagnoseResponse(BaseModel):
  session_id: str
  status: str
  message: str


class ApproveRequest(BaseModel):
  approved: bool


class StatusResponse(BaseModel):
  session_id: str
  status: str
  phase: str
  created_at: str
  message: Optional[str] = None


# === Session State (in-memory for MVP) ===

sessions: dict = {}


# === FastAPI App ===

app = FastAPI(
  title="SRE Incident Diagnosis",
  description="Diagnose infrastructure problems with autonomous agents",
  version="0.1.0"
)


@app.get("/")
async def index():
  """Serve the dashboard."""
  return FileResponse("index.html")


@app.get("/health")
async def health():
  """Health check endpoint."""
  return JSONResponse(
    content={"status": "healthy", "service": "sre-diagnose"},
    status_code=200
  )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
  """Start a new diagnostic session."""
  session_id = str(uuid.uuid4())[:8]

  sessions[session_id] = {
    "id": session_id,
    "status": "created",
    "phase": "initialized",
    "problem": request.problem,
    "environment": request.environment,
    "context": request.context or {},
    "created_at": datetime.utcnow().isoformat(),
    "findings": [],
    "approval_pending": False,
  }

  return DiagnoseResponse(
    session_id=session_id,
    status="created",
    message=f"Diagnosis session created for: {request.problem[:50]}..."
  )


@app.post("/approve/{session_id}")
async def approve(session_id: str, request: ApproveRequest):
  """Approve or deny credential access for a diagnostic session."""
  if session_id not in sessions:
    return JSONResponse(
      content={"error": "Session not found"},
      status_code=404
    )

  session = sessions[session_id]

  if not session.get("approval_pending"):
    return JSONResponse(
      content={"error": "No approval pending for this session"},
      status_code=400
    )

  if request.approved:
    session["status"] = "approved"
    session["phase"] = "credentials_approved"
    session["approval_pending"] = False
    return {"session_id": session_id, "status": "approved", "message": "Credential access approved"}
  else:
    session["status"] = "denied"
    session["phase"] = "credentials_denied"
    session["approval_pending"] = False
    return {"session_id": session_id, "status": "denied", "message": "Credential access denied, diagnosis cancelled"}


@app.get("/status/{session_id}", response_model=StatusResponse)
async def status(session_id: str):
  """Get the current status of a diagnostic session."""
  if session_id not in sessions:
    return JSONResponse(
      content={"error": "Session not found"},
      status_code=404
    )

  session = sessions[session_id]
  return StatusResponse(
    session_id=session_id,
    status=session["status"],
    phase=session["phase"],
    created_at=session["created_at"],
    message=session.get("message")
  )


@app.get("/sessions")
async def list_sessions():
  """List all diagnostic sessions."""
  return {
    "sessions": [
      {
        "session_id": s["id"],
        "status": s["status"],
        "phase": s["phase"],
        "problem": s["problem"][:50] + "..." if len(s["problem"]) > 50 else s["problem"],
        "created_at": s["created_at"]
      }
      for s in sessions.values()
    ]
  }


# === Start Node ===

Node.start(http_server=HttpServer(app=app))

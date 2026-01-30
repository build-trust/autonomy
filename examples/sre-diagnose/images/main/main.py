"""SRE Incident Diagnosis - Main Entry Point

Provides HTTP API for incident diagnosis with human-in-the-loop credential approval.
"""

from autonomy import Agent, HttpServer, Model, Node, NodeDep
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json
import secrets
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
  analysis: Optional[str] = None


# === Session State (in-memory for MVP) ===

sessions: dict = {}


# === Orchestrator Agent Instructions ===

ORCHESTRATOR_INSTRUCTIONS = """You are an expert SRE (Site Reliability Engineer) incident diagnosis agent.

Your role is to analyze infrastructure problems and provide a structured diagnosis.

When given a problem description:
1. Identify the type of incident (database, network, application, cloud infrastructure, etc.)
2. List potential root causes in order of likelihood
3. Suggest diagnostic steps to investigate each potential cause
4. Identify what credentials/access would be needed to investigate further

Format your analysis as follows:

## Incident Classification
[Type of incident and severity assessment]

## Potential Root Causes
1. [Most likely cause] - [Brief explanation]
2. [Second likely cause] - [Brief explanation]
3. [Third likely cause] - [Brief explanation]

## Recommended Diagnostic Steps
1. [First step to investigate]
2. [Second step to investigate]
3. [Additional steps...]

## Required Access
- [List of systems/credentials needed for deeper investigation]

Be concise but thorough. Focus on actionable insights."""


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


def json_serializer(obj):
  """JSON serializer for objects not serializable by default."""
  if hasattr(obj, "isoformat"):
    return obj.isoformat()
  if hasattr(obj, "__dict__"):
    return obj.__dict__
  return str(obj)


@app.post("/diagnose")
async def diagnose(request: DiagnoseRequest, node: NodeDep):
  """Start a new diagnostic session with streaming response."""
  session_id = str(uuid.uuid4())[:8]
  agent_name = f"orchestrator_{session_id}_{secrets.token_hex(4)}"

  sessions[session_id] = {
    "id": session_id,
    "status": "analyzing",
    "phase": "initial_analysis",
    "problem": request.problem,
    "environment": request.environment,
    "context": request.context or {},
    "created_at": datetime.utcnow().isoformat(),
    "findings": [],
    "approval_pending": False,
    "analysis": "",
    "agent_name": agent_name,
  }

  # Create the orchestrator agent
  agent = await Agent.start(
    node=node,
    name=agent_name,
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    model=Model("claude-sonnet-4-v1"),
  )

  # Build the diagnosis message
  context_str = ""
  if request.context:
    context_str = "\n".join(f"- {k}: {v}" for k, v in request.context.items())
    context_str = f"\n\nAdditional Context:\n{context_str}"

  message = f"""Analyze this infrastructure incident:

**Problem**: {request.problem}
**Environment**: {request.environment}{context_str}

Provide your initial diagnosis and assessment."""

  async def stream_response():
    session = sessions[session_id]
    full_analysis = ""

    try:
      # Send initial session info
      yield json.dumps({
        "type": "session_start",
        "session_id": session_id,
        "status": "analyzing"
      }) + "\n"

      # Stream the agent's analysis
      async for response in agent.send_stream(message, timeout=120):
        snippet = response.snippet
        yield json.dumps(snippet, default=json_serializer) + "\n"

        # Accumulate the analysis text
        if hasattr(snippet, "text"):
          full_analysis += snippet.text
        elif isinstance(snippet, dict) and "text" in snippet:
          full_analysis += snippet["text"]

      # Update session with completed analysis
      session["analysis"] = full_analysis
      session["status"] = "analyzed"
      session["phase"] = "analysis_complete"

      yield json.dumps({
        "type": "session_complete",
        "session_id": session_id,
        "status": "analyzed",
        "phase": "analysis_complete"
      }) + "\n"

    except Exception as e:
      session["status"] = "error"
      session["phase"] = "analysis_failed"
      session["message"] = str(e)
      yield json.dumps({
        "type": "error",
        "session_id": session_id,
        "message": str(e)
      }) + "\n"

    finally:
      # Clean up agent
      try:
        await Agent.stop(node, agent_name)
      except Exception:
        pass

  return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@app.post("/diagnose/sync", response_model=DiagnoseResponse)
async def diagnose_sync(request: DiagnoseRequest, node: NodeDep):
  """Start a new diagnostic session (non-streaming, returns immediately)."""
  session_id = str(uuid.uuid4())[:8]
  agent_name = f"orchestrator_{session_id}_{secrets.token_hex(4)}"

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
    "analysis": "",
    "agent_name": agent_name,
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
    return {
      "session_id": session_id,
      "status": "approved",
      "message": "Credential access approved"
    }
  else:
    session["status"] = "denied"
    session["phase"] = "credentials_denied"
    session["approval_pending"] = False
    return {
      "session_id": session_id,
      "status": "denied",
      "message": "Credential access denied, diagnosis cancelled"
    }


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
    message=session.get("message"),
    analysis=session.get("analysis")
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

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
import asyncio
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
  message: Optional[str] = None


class StatusResponse(BaseModel):
  session_id: str
  status: str
  phase: str
  created_at: str
  message: Optional[str] = None
  analysis: Optional[str] = None
  approval_prompt: Optional[str] = None


# === Session State (in-memory for MVP) ===

sessions: dict = {}

# Store active agents for resume capability
active_agents: dict = {}


# === Orchestrator Agent Instructions ===

ORCHESTRATOR_INSTRUCTIONS = """You are an expert SRE (Site Reliability Engineer) incident diagnosis agent.

Your role is to analyze infrastructure problems and provide a structured diagnosis.

IMPORTANT WORKFLOW:
1. First, analyze the problem and identify what credentials/access you would need
2. Use the ask_user_for_input tool to request approval before accessing any systems
3. Wait for approval before proceeding with deeper diagnosis

When given a problem description:
1. Identify the type of incident (database, network, application, cloud infrastructure, etc.)
2. List potential root causes in order of likelihood
3. Identify what credentials/access would be needed to investigate further
4. Use ask_user_for_input to request approval for the credentials you need

Format your initial analysis as follows:

## Incident Classification
[Type of incident and severity assessment]

## Potential Root Causes
1. [Most likely cause] - [Brief explanation]
2. [Second likely cause] - [Brief explanation]
3. [Third likely cause] - [Brief explanation]

## Required Access for Investigation
- [Credential 1]: [Why it's needed]
- [Credential 2]: [Why it's needed]

After your analysis, you MUST use ask_user_for_input to request approval with a message like:
"To proceed with diagnosis, I need access to the following credentials:
- op://Infrastructure/prod-db (read-only database access)
- op://Infrastructure/aws-cloudwatch (log access)

Reply 'approve' to grant access, or 'deny' to cancel."

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
    "approval_prompt": None,
    "analysis": "",
    "agent_name": agent_name,
    "node": node,
  }

  # Create the orchestrator agent with human-in-the-loop enabled
  agent = await Agent.start(
    node=node,
    name=agent_name,
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    model=Model("claude-sonnet-4-v1"),
    enable_ask_for_user_input=True,
  )

  # Store agent reference for later resume
  active_agents[session_id] = agent

  # Build the diagnosis message
  context_str = ""
  if request.context:
    context_str = "\n".join(f"- {k}: {v}" for k, v in request.context.items())
    context_str = f"\n\nAdditional Context:\n{context_str}"

  message = f"""Analyze this infrastructure incident and request approval for any credentials you need:

**Problem**: {request.problem}
**Environment**: {request.environment}{context_str}

After your initial analysis, use ask_user_for_input to request approval for the credentials needed."""

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

        # Check for waiting_for_input phase
        if hasattr(snippet, "phase") and snippet.phase == "waiting_for_input":
          session["status"] = "waiting_for_approval"
          session["phase"] = "waiting_for_approval"
          session["approval_pending"] = True
          session["approval_prompt"] = getattr(snippet, "content", full_analysis)
          session["analysis"] = full_analysis

          yield json.dumps({
            "type": "approval_required",
            "session_id": session_id,
            "status": "waiting_for_approval",
            "phase": "waiting_for_approval",
            "prompt": session["approval_prompt"]
          }) + "\n"
          return  # Stop streaming, wait for approval

        elif isinstance(snippet, dict) and snippet.get("phase") == "waiting_for_input":
          session["status"] = "waiting_for_approval"
          session["phase"] = "waiting_for_approval"
          session["approval_pending"] = True
          session["approval_prompt"] = snippet.get("content", full_analysis)
          session["analysis"] = full_analysis

          yield json.dumps({
            "type": "approval_required",
            "session_id": session_id,
            "status": "waiting_for_approval",
            "phase": "waiting_for_approval",
            "prompt": session["approval_prompt"]
          }) + "\n"
          return  # Stop streaming, wait for approval

      # If we get here without requesting approval, analysis is complete
      session["analysis"] = full_analysis
      session["status"] = "analyzed"
      session["phase"] = "analysis_complete"

      yield json.dumps({
        "type": "session_complete",
        "session_id": session_id,
        "status": "analyzed",
        "phase": "analysis_complete"
      }) + "\n"

      # Clean up if analysis completes without needing approval
      if session_id in active_agents:
        del active_agents[session_id]
      try:
        await Agent.stop(node, agent_name)
      except Exception:
        pass

    except Exception as e:
      session["status"] = "error"
      session["phase"] = "analysis_failed"
      session["message"] = str(e)
      yield json.dumps({
        "type": "error",
        "session_id": session_id,
        "message": str(e)
      }) + "\n"

      # Clean up on error
      if session_id in active_agents:
        del active_agents[session_id]
      try:
        await Agent.stop(node, agent_name)
      except Exception:
        pass

  return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@app.post("/approve/{session_id}")
async def approve(session_id: str, request: ApproveRequest):
  """Approve or deny credential access and resume the agent."""
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

  if session_id not in active_agents:
    return JSONResponse(
      content={"error": "Agent no longer active for this session"},
      status_code=400
    )

  agent = active_agents[session_id]

  if request.approved:
    session["status"] = "approved"
    session["phase"] = "credentials_approved"
    session["approval_pending"] = False

    # Resume the agent by sending the approval response
    # This will trigger the agent to continue from WAITING_FOR_INPUT state
    async def stream_resume():
      try:
        yield json.dumps({
          "type": "approval_accepted",
          "session_id": session_id,
          "status": "approved"
        }) + "\n"

        # Send approval message to resume the agent
        resume_message = request.message or "approve"
        async for response in agent.send_stream(resume_message, timeout=120):
          snippet = response.snippet
          yield json.dumps(snippet, default=json_serializer) + "\n"

          # Update analysis with continued response
          if hasattr(snippet, "text"):
            session["analysis"] += snippet.text
          elif isinstance(snippet, dict) and "text" in snippet:
            session["analysis"] += snippet["text"]

        session["status"] = "completed"
        session["phase"] = "diagnosis_complete"

        yield json.dumps({
          "type": "diagnosis_complete",
          "session_id": session_id,
          "status": "completed",
          "phase": "diagnosis_complete"
        }) + "\n"

      except Exception as e:
        session["status"] = "error"
        session["phase"] = "diagnosis_failed"
        session["message"] = str(e)
        yield json.dumps({
          "type": "error",
          "session_id": session_id,
          "message": str(e)
        }) + "\n"

      finally:
        # Clean up agent
        if session_id in active_agents:
          del active_agents[session_id]
        try:
          node = session.get("node")
          if node:
            await Agent.stop(node, session["agent_name"])
        except Exception:
          pass

    return StreamingResponse(stream_resume(), media_type="application/x-ndjson")

  else:
    # Denial - cancel the diagnosis
    session["status"] = "denied"
    session["phase"] = "credentials_denied"
    session["approval_pending"] = False

    # Clean up agent
    if session_id in active_agents:
      del active_agents[session_id]
    try:
      node = session.get("node")
      if node:
        await Agent.stop(node, session["agent_name"])
    except Exception:
      pass

    return JSONResponse(
      content={
        "session_id": session_id,
        "status": "denied",
        "message": "Credential access denied, diagnosis cancelled"
      }
    )


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
    analysis=session.get("analysis"),
    approval_prompt=session.get("approval_prompt")
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
        "created_at": s["created_at"],
        "approval_pending": s.get("approval_pending", False)
      }
      for s in sessions.values()
    ]
  }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
  """Delete a diagnostic session and clean up resources."""
  if session_id not in sessions:
    return JSONResponse(
      content={"error": "Session not found"},
      status_code=404
    )

  session = sessions[session_id]

  # Clean up agent if active
  if session_id in active_agents:
    del active_agents[session_id]
    try:
      node = session.get("node")
      if node:
        await Agent.stop(node, session["agent_name"])
    except Exception:
      pass

  del sessions[session_id]

  return {"status": "deleted", "session_id": session_id}


# === Start Node ===

Node.start(http_server=HttpServer(app=app))

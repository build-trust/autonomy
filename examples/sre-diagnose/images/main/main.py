"""SRE Incident Diagnosis - Main Entry Point

Provides HTTP API for incident diagnosis with human-in-the-loop credential approval.
"""

from autonomy import Agent, HttpServer, Model, Node, NodeDep, tool
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json
import secrets
import httpx
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
  credentials_retrieved: Optional[list] = None


# === Session State (in-memory for MVP) ===

sessions: dict = {}

# Store active agents for resume capability
active_agents: dict = {}

# HTTP client for calling mock 1Password server
http_client = httpx.AsyncClient(timeout=30.0)

# Mock 1Password server URL (running in same pod)
MOCK_1PASSWORD_URL = "http://localhost:8080"


# === Credential Retrieval Tool ===

def create_get_credential_tool(session_id: str):
  """Create a get_credential tool bound to a specific session."""

  @tool(
    name="get_credential",
    description="""Retrieve a credential from 1Password by its reference.

Reference format: op://vault/item/field
Examples:
- op://Infrastructure/prod-db/password
- op://Infrastructure/aws-cloudwatch/access-key

IMPORTANT: You must have approval before calling this tool.
The actual credential value is stored securely and NOT returned to you.
You will only receive confirmation that the credential was retrieved."""
  )
  async def get_credential(reference: str) -> str:
    """
    Retrieve a credential from 1Password.

    The actual credential value is stored securely and NOT returned to the LLM.
    Only a confirmation message is returned.
    """
    session = sessions.get(session_id)
    if not session:
      return f"Error: Session {session_id} not found"

    # Check if approval was granted
    if session.get("status") != "approved" and session.get("phase") != "credentials_approved":
      return "Error: Credential access not approved. Use ask_user_for_input to request approval first."

    # Normalize reference
    if not reference.startswith("op://"):
      reference = f"op://{reference}"

    try:
      # Call mock 1Password server
      response = await http_client.get(
        f"{MOCK_1PASSWORD_URL}/secrets/{reference}"
      )

      if response.status_code == 200:
        data = response.json()
        # Store credential in session (for later use by diagnostic tools)
        # NEVER return actual credential value to LLM
        if "credentials" not in session:
          session["credentials"] = {}
        session["credentials"][reference] = data["value"]

        # Track which credentials have been retrieved
        if "credentials_retrieved" not in session:
          session["credentials_retrieved"] = []
        if reference not in session["credentials_retrieved"]:
          session["credentials_retrieved"].append(reference)

        # Return only confirmation (NOT the actual credential!)
        return f"✓ Successfully retrieved credential: {reference}"

      elif response.status_code == 404:
        return f"Error: Credential not found: {reference}"
      else:
        return f"Error retrieving credential: HTTP {response.status_code}"

    except httpx.RequestError as e:
      return f"Error connecting to credential store: {str(e)}"

  return get_credential


# === Orchestrator Agent Instructions ===

ORCHESTRATOR_INSTRUCTIONS = """You are an expert SRE (Site Reliability Engineer) incident diagnosis agent.

Your role is to analyze infrastructure problems and provide a structured diagnosis.

IMPORTANT WORKFLOW:
1. First, analyze the problem and identify what credentials/access you would need
2. Use the ask_user_for_input tool to request approval before accessing any systems
3. Wait for approval before proceeding with deeper diagnosis
4. After approval, use the get_credential tool to retrieve each credential you need
5. Proceed with diagnosis using the retrieved credentials

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
- op://Infrastructure/prod-db/password (read-only database access)
- op://Infrastructure/aws-cloudwatch/access-key (log access)

Reply 'approve' to grant access, or 'deny' to cancel."

After receiving approval:
1. Use get_credential for each credential you need
2. You will receive confirmation that credentials were retrieved (not the actual values)
3. Continue with your diagnosis knowing the credentials are available

Be concise but thorough. Focus on actionable insights."""


# === FastAPI App ===

app = FastAPI(
  title="SRE Incident Diagnosis",
  description="Diagnose infrastructure problems with autonomous agents",
  version="0.2.0"
)


@app.get("/")
async def index():
  """Serve the dashboard."""
  return FileResponse("index.html")


@app.get("/health")
async def health():
  """Health check endpoint."""
  # Also check mock 1Password server health
  onepass_status = "unknown"
  try:
    response = await http_client.get(f"{MOCK_1PASSWORD_URL}/health")
    if response.status_code == 200:
      onepass_status = "healthy"
    else:
      onepass_status = f"unhealthy (HTTP {response.status_code})"
  except httpx.RequestError as e:
    onepass_status = f"unreachable ({type(e).__name__})"

  return JSONResponse(
    content={
      "status": "healthy",
      "service": "sre-diagnose",
      "dependencies": {
        "mock-1password": onepass_status
      }
    },
    status_code=200
  )


def extract_text_from_snippet(snippet):
  """Extract text content from a conversation snippet."""
  text = ""
  phase = None

  # Handle the snippet object
  if hasattr(snippet, "messages"):
    for msg in snippet.messages:
      if hasattr(msg, "content") and hasattr(msg.content, "text"):
        text += msg.content.text
      if hasattr(msg, "phase"):
        # Get the string value of the phase enum
        if hasattr(msg.phase, "value"):
          phase = msg.phase.value
        else:
          phase = str(msg.phase)

  return text, phase


def serialize_snippet(snippet):
  """Serialize a snippet to a simple JSON-friendly dict."""
  text, phase = extract_text_from_snippet(snippet)

  result = {"text": text}
  if phase:
    result["phase"] = phase

  return result


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
    "credentials": {},  # Store retrieved credentials here (never exposed to LLM)
    "credentials_retrieved": [],  # Track which credentials have been retrieved
  }

  # Create the session-bound credential tool
  credential_tool = create_get_credential_tool(session_id)

  # Create the orchestrator agent with human-in-the-loop enabled and credential tool
  agent = await Agent.start(
    node=node,
    name=agent_name,
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    model=Model("claude-sonnet-4-v1"),
    enable_ask_for_user_input=True,
    tools=[credential_tool],
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

        # Extract text and phase from snippet
        text, phase = extract_text_from_snippet(snippet)

        if text:
          full_analysis += text
          yield json.dumps({"type": "text", "text": text}) + "\n"

        # Check for waiting_for_input phase
        if phase == "waiting_for_input":
          session["status"] = "waiting_for_approval"
          session["phase"] = "waiting_for_approval"
          session["approval_pending"] = True
          session["approval_prompt"] = text or full_analysis
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
    async def stream_resume():
      try:
        yield json.dumps({
          "type": "approval_accepted",
          "session_id": session_id,
          "status": "approved"
        }) + "\n"

        # Send approval message to resume the agent
        # The agent can now use get_credential tool
        resume_message = request.message or "Approved. You may now retrieve the credentials you requested using the get_credential tool."
        async for response in agent.send_stream(resume_message, timeout=120):
          snippet = response.snippet
          text, phase = extract_text_from_snippet(snippet)

          if text:
            session["analysis"] += text
            yield json.dumps({"type": "text", "text": text}) + "\n"

          # Check for tool calls (credential retrieval)
          if hasattr(snippet, "messages"):
            for msg in snippet.messages:
              if hasattr(msg, "tool_calls"):
                for tool_call in msg.tool_calls:
                  yield json.dumps({
                    "type": "tool_call",
                    "tool": getattr(tool_call, "name", "unknown"),
                  }) + "\n"

        session["status"] = "completed"
        session["phase"] = "diagnosis_complete"

        yield json.dumps({
          "type": "diagnosis_complete",
          "session_id": session_id,
          "status": "completed",
          "phase": "diagnosis_complete",
          "credentials_retrieved": session.get("credentials_retrieved", [])
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
    approval_prompt=session.get("approval_prompt"),
    credentials_retrieved=session.get("credentials_retrieved", [])
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
        "approval_pending": s.get("approval_pending", False),
        "credentials_retrieved": len(s.get("credentials_retrieved", []))
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


# === Debug Endpoints (for credential security verification) ===

@app.get("/debug/sessions/{session_id}/credentials")
async def debug_credentials(session_id: str):
  """
  Debug endpoint to verify credentials are stored in session.
  ONLY shows that credentials exist, NOT the actual values.
  In production, this endpoint should be protected or removed.
  """
  if session_id not in sessions:
    return JSONResponse(
      content={"error": "Session not found"},
      status_code=404
    )

  session = sessions[session_id]
  credentials = session.get("credentials", {})

  # Return only metadata, never actual values
  return {
    "session_id": session_id,
    "credentials_count": len(credentials),
    "credentials_references": list(credentials.keys()),
    "note": "Actual credential values are stored securely and never exposed via API"
  }


# === Start Node ===

Node.start(http_server=HttpServer(app=app))

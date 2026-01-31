"""SRE Incident Diagnosis - Main Entry Point

Provides HTTP API for incident diagnosis with human-in-the-loop credential approval.
Uses a two-phase approach:
1. Analysis phase: Agent analyzes problem and identifies needed credentials
2. Diagnosis phase: After approval, new agent runs diagnosis with credentials
"""

from autonomy import Agent, HttpServer, Model, Node, NodeDep, Tool
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json
import secrets
import httpx
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sre-diagnose")


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

# HTTP client for calling mock 1Password server
http_client = httpx.AsyncClient(timeout=30.0)

# Mock 1Password server URL (running in same pod)
MOCK_1PASSWORD_URL = "http://localhost:8080"


# === Credential Retrieval Function ===

async def retrieve_credential(reference: str, session: dict) -> str:
  """
  Retrieve a credential from 1Password.
  Returns the result message and stores credential in session.
  """
  # Normalize reference
  if not reference.startswith("op://"):
    reference = f"op://{reference}"

  try:
    response = await http_client.get(f"{MOCK_1PASSWORD_URL}/secrets/{reference}")

    if response.status_code == 200:
      data = response.json()
      # Store credential in session (for later use by diagnostic tools)
      if "credentials" not in session:
        session["credentials"] = {}
      session["credentials"][reference] = data["value"]

      # Track which credentials have been retrieved
      if "credentials_retrieved" not in session:
        session["credentials_retrieved"] = []
      if reference not in session["credentials_retrieved"]:
        session["credentials_retrieved"].append(reference)

      return f"Successfully retrieved: {reference}"
    elif response.status_code == 404:
      return f"Not found: {reference}"
    else:
      return f"Error (HTTP {response.status_code}): {reference}"

  except httpx.RequestError as e:
    return f"Connection error: {str(e)}"


# === Agent Instructions ===

ANALYSIS_INSTRUCTIONS = """You are an expert SRE (Site Reliability Engineer) analyzing an infrastructure incident.

Your task is to analyze the problem and identify what credentials would be needed to diagnose it.

AVAILABLE CREDENTIALS (you can request any of these):
- op://Infrastructure/prod-db/password - Production database password
- op://Infrastructure/prod-db/username - Production database username
- op://Infrastructure/prod-db/host - Production database host
- op://Infrastructure/staging-db/password - Staging database password
- op://Infrastructure/staging-db/username - Staging database username
- op://Infrastructure/aws-cloudwatch/access-key - CloudWatch access key
- op://Infrastructure/aws-cloudwatch/secret-key - CloudWatch secret key
- op://Infrastructure/aws-ec2/access-key - EC2 access key
- op://Infrastructure/aws-ec2/secret-key - EC2 secret key
- op://Infrastructure/k8s-prod/token - Kubernetes production token
- op://Infrastructure/datadog/api-key - Datadog API key
- op://Infrastructure/grafana/admin-password - Grafana admin password
- op://Services/payment-api/api-key - Payment API key
- op://Services/email-service/smtp-password - Email service SMTP password

Provide your analysis in this format:

## Incident Classification
[Type and severity assessment]

## Potential Root Causes
1. [Most likely] - [Brief explanation]
2. [Second likely] - [Brief explanation]
3. [Third likely] - [Brief explanation]

## Required Credentials
List the specific credentials needed from the available list above:
- op://... : [why needed]
- op://... : [why needed]

## Recommended Investigation Steps
1. [First step with specific credential]
2. [Second step]
3. [Third step]

Be concise and specific. Only request credentials that are actually needed."""


DIAGNOSIS_INSTRUCTIONS = """You are an expert SRE (Site Reliability Engineer) completing a diagnosis.

You have been given access to the requested credentials (stored securely, not visible to you).
Based on the initial analysis, provide your final diagnosis and recommendations.

Focus on:
1. Most likely root cause based on the incident type
2. Immediate remediation steps (what to do right now)
3. Long-term prevention measures (how to prevent recurrence)
4. Monitoring improvements (what alerts or metrics to add)

Be specific and actionable. Provide commands or configurations where applicable."""


# === FastAPI App ===

app = FastAPI(
  title="SRE Incident Diagnosis",
  description="Diagnose infrastructure problems with autonomous agents",
  version="0.3.0"
)


@app.get("/")
async def index():
  """Serve the dashboard."""
  return FileResponse("index.html")


@app.get("/health")
async def health():
  """Health check endpoint."""
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
      "version": "0.3.0",
      "dependencies": {
        "mock-1password": onepass_status
      }
    },
    status_code=200
  )


def extract_text_from_snippet(snippet):
  """Extract text content from a conversation snippet."""
  text = ""
  if hasattr(snippet, "messages"):
    for msg in snippet.messages:
      if hasattr(msg, "content"):
        if hasattr(msg.content, "text"):
          text += msg.content.text
        elif isinstance(msg.content, str):
          text += msg.content
  return text


def extract_credential_refs(text: str) -> list:
  """Extract op:// credential references from text."""
  pattern = r'op://[^\s\)\]\,\:\*]+(?:/[^\s\)\]\,\:\*]+)*'
  matches = re.findall(pattern, text)
  # Clean up any trailing punctuation or markdown formatting
  cleaned = []
  for m in matches:
    m = m.rstrip('.,;:*')
    if m not in cleaned:
      cleaned.append(m)
  return cleaned


@app.post("/diagnose")
async def diagnose(request: DiagnoseRequest, node: NodeDep):
  """Start a new diagnostic session - Phase 1: Analysis."""
  session_id = str(uuid.uuid4())[:8]
  agent_name = f"analyzer_{session_id}_{secrets.token_hex(4)}"

  sessions[session_id] = {
    "id": session_id,
    "status": "analyzing",
    "phase": "analysis",
    "problem": request.problem,
    "environment": request.environment,
    "context": request.context or {},
    "created_at": datetime.utcnow().isoformat(),
    "analysis": "",
    "requested_credentials": [],
    "credentials": {},
    "credentials_retrieved": [],
    "agent_name": agent_name,
  }

  # Create analysis agent (no tools needed, just analysis)
  agent = await Agent.start(
    node=node,
    name=agent_name,
    instructions=ANALYSIS_INSTRUCTIONS,
    model=Model("claude-sonnet-4-5"),
  )

  # Build the analysis message
  context_str = ""
  if request.context:
    context_str = "\n".join(f"- {k}: {v}" for k, v in request.context.items())
    context_str = f"\n\nAdditional Context:\n{context_str}"

  message = f"""Analyze this infrastructure incident:

**Problem**: {request.problem}
**Environment**: {request.environment}{context_str}

Provide your analysis including the specific credentials needed for investigation."""

  async def stream_response():
    session = sessions[session_id]
    full_analysis = ""

    try:
      yield json.dumps({
        "type": "session_start",
        "session_id": session_id,
        "status": "analyzing"
      }) + "\n"

      # Stream the agent's analysis
      async for response in agent.send_stream(message, timeout=120):
        snippet = response.snippet
        text = extract_text_from_snippet(snippet)

        if text:
          full_analysis += text
          yield json.dumps({"type": "text", "text": text}) + "\n"

      # Store analysis
      session["analysis"] = full_analysis

      # Extract requested credentials from analysis
      requested_creds = extract_credential_refs(full_analysis)
      session["requested_credentials"] = requested_creds

      # Update status
      session["status"] = "waiting_for_approval"
      session["phase"] = "awaiting_approval"

      yield json.dumps({
        "type": "approval_required",
        "session_id": session_id,
        "status": "waiting_for_approval",
        "requested_credentials": requested_creds,
        "message": f"Analysis complete. {len(requested_creds)} credentials requested. Approve to continue."
      }) + "\n"

    except Exception as e:
      logger.error(f"Error in analysis: {str(e)}")
      session["status"] = "error"
      session["phase"] = "analysis_failed"
      session["message"] = str(e)
      yield json.dumps({
        "type": "error",
        "session_id": session_id,
        "message": str(e)
      }) + "\n"

    finally:
      # Clean up analysis agent
      try:
        await Agent.stop(node, agent_name)
      except Exception:
        pass

  return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@app.post("/approve/{session_id}")
async def approve(session_id: str, request: ApproveRequest, node: NodeDep):
  """Approve or deny credential access - Phase 2: Diagnosis."""
  if session_id not in sessions:
    return JSONResponse(content={"error": "Session not found"}, status_code=404)

  session = sessions[session_id]

  if session.get("status") != "waiting_for_approval":
    return JSONResponse(
      content={"error": f"Session not waiting for approval (status: {session.get('status')})"},
      status_code=400
    )

  if not request.approved:
    session["status"] = "denied"
    session["phase"] = "credentials_denied"
    return JSONResponse(content={
      "session_id": session_id,
      "status": "denied",
      "message": "Credential access denied, diagnosis cancelled"
    })

  # Approved - retrieve credentials and start diagnosis
  session["status"] = "retrieving_credentials"
  session["phase"] = "credential_retrieval"

  diagnosis_agent_name = f"diagnoser_{session_id}_{secrets.token_hex(4)}"

  async def stream_diagnosis():
    try:
      yield json.dumps({
        "type": "approval_accepted",
        "session_id": session_id,
        "status": "approved"
      }) + "\n"

      # Retrieve all requested credentials
      requested = session.get("requested_credentials", [])
      for ref in requested:
        result = await retrieve_credential(ref, session)
        success = "Successfully" in result
        yield json.dumps({
          "type": "credential_retrieved",
          "reference": ref,
          "success": success
        }) + "\n"
        logger.info(f"Credential retrieval: {result}")

      # Update status
      session["status"] = "diagnosing"
      session["phase"] = "diagnosis"

      yield json.dumps({
        "type": "diagnosis_started",
        "session_id": session_id,
        "credentials_retrieved": len(session.get("credentials_retrieved", []))
      }) + "\n"

      # Create diagnosis agent
      agent = await Agent.start(
        node=node,
        name=diagnosis_agent_name,
        instructions=DIAGNOSIS_INSTRUCTIONS,
        model=Model("claude-sonnet-4-5"),
      )

      # Build diagnosis prompt with context from analysis
      creds_summary = ", ".join(session.get("credentials_retrieved", [])) or "none"
      diagnosis_message = f"""Complete the diagnosis for this incident.

**Original Problem**: {session.get('problem')}
**Environment**: {session.get('environment')}

**Initial Analysis**:
{session.get('analysis', 'No analysis available')}

**Credentials Retrieved**: {creds_summary}

Based on this analysis and with access to the credentials, provide your final diagnosis and remediation recommendations."""

      # Stream diagnosis
      diagnosis_text = ""
      async for response in agent.send_stream(diagnosis_message, timeout=120):
        snippet = response.snippet
        text = extract_text_from_snippet(snippet)

        if text:
          diagnosis_text += text
          yield json.dumps({"type": "text", "text": text}) + "\n"

      # Update session with diagnosis
      session["diagnosis"] = diagnosis_text
      session["status"] = "completed"
      session["phase"] = "diagnosis_complete"

      yield json.dumps({
        "type": "diagnosis_complete",
        "session_id": session_id,
        "status": "completed",
        "credentials_retrieved": session.get("credentials_retrieved", [])
      }) + "\n"

    except Exception as e:
      logger.error(f"Error in diagnosis: {str(e)}")
      import traceback
      logger.error(f"Traceback: {traceback.format_exc()}")
      session["status"] = "error"
      session["phase"] = "diagnosis_failed"
      session["message"] = str(e)
      yield json.dumps({
        "type": "error",
        "session_id": session_id,
        "message": str(e)
      }) + "\n"

    finally:
      # Clean up diagnosis agent
      try:
        await Agent.stop(node, diagnosis_agent_name)
      except Exception:
        pass

  return StreamingResponse(stream_diagnosis(), media_type="application/x-ndjson")


@app.get("/status/{session_id}", response_model=StatusResponse)
async def status(session_id: str):
  """Get the current status of a diagnostic session."""
  if session_id not in sessions:
    return JSONResponse(content={"error": "Session not found"}, status_code=404)

  session = sessions[session_id]
  return StatusResponse(
    session_id=session_id,
    status=session["status"],
    phase=session["phase"],
    created_at=session["created_at"],
    message=session.get("message"),
    analysis=session.get("analysis"),
    approval_prompt=f"Requested credentials: {session.get('requested_credentials', [])}",
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
        "credentials_retrieved": len(s.get("credentials_retrieved", []))
      }
      for s in sessions.values()
    ]
  }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
  """Delete a diagnostic session."""
  if session_id not in sessions:
    return JSONResponse(content={"error": "Session not found"}, status_code=404)

  del sessions[session_id]
  return {"status": "deleted", "session_id": session_id}


# === Debug Endpoints ===

@app.get("/debug/sessions/{session_id}/credentials")
async def debug_credentials(session_id: str):
  """Debug endpoint to verify credentials are stored (not actual values)."""
  if session_id not in sessions:
    return JSONResponse(content={"error": "Session not found"}, status_code=404)

  session = sessions[session_id]
  credentials = session.get("credentials", {})

  return {
    "session_id": session_id,
    "credentials_count": len(credentials),
    "credentials_references": list(credentials.keys()),
    "note": "Actual credential values are stored securely and never exposed via API"
  }


@app.get("/debug/test-credential/{reference:path}")
async def test_credential(reference: str):
  """Debug endpoint to test credential retrieval directly."""
  test_session = {"credentials": {}, "credentials_retrieved": []}
  result = await retrieve_credential(reference, test_session)
  return {
    "reference": reference,
    "result": result,
    "success": "Successfully" in result
  }


# === Start Node ===

Node.start(http_server=HttpServer(app=app))

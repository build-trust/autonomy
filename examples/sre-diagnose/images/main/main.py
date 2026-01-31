"""SRE Incident Diagnosis - Main Entry Point

Provides HTTP API for incident diagnosis with human-in-the-loop credential approval.
Uses a two-phase approach:
1. Analysis phase: Agent analyzes problem and identifies needed credentials
2. Diagnosis phase: After approval, new agent runs diagnosis with credentials

Supports two 1Password modes:
- mock: Uses local mock 1Password server (default, for development)
- sdk: Uses real 1Password SDK with service account (for production)

Set ONEPASSWORD_MODE=sdk and OP_SERVICE_ACCOUNT_TOKEN for production use.
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
import asyncio
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sre-diagnose")

# === 1Password Configuration ===

# Mode: "mock" (default) or "sdk"
ONEPASSWORD_MODE = os.environ.get("ONEPASSWORD_MODE", "mock").lower()

# 1Password SDK client (initialized lazily for sdk mode)
_op_client = None

async def get_onepassword_client():
  """Get or create the 1Password SDK client (lazy initialization)."""
  global _op_client
  if _op_client is None and ONEPASSWORD_MODE == "sdk":
    try:
      from onepassword import Client
      token = os.environ.get("OP_SERVICE_ACCOUNT_TOKEN")
      if not token:
        raise ValueError("OP_SERVICE_ACCOUNT_TOKEN environment variable is required for sdk mode")
      _op_client = await Client.authenticate(
        auth=token,
        integration_name="sre-diagnose",
        integration_version=VERSION
      )
      logger.info("1Password SDK client initialized successfully")
    except ImportError:
      raise ImportError("onepassword-sdk package not installed. Run: pip install onepassword-sdk")
  return _op_client

logger.info(f"1Password mode: {ONEPASSWORD_MODE}")

# === Configuration Constants ===

VERSION = "0.6.0"

# Timeouts (in seconds)
ANALYSIS_TIMEOUT = 120
SPECIALIST_TIMEOUT = 90
SYNTHESIS_TIMEOUT = 120
CREDENTIAL_TIMEOUT = 10

# Retry configuration
CREDENTIAL_RETRY_ATTEMPTS = 3
CREDENTIAL_RETRY_DELAY = 1.0  # seconds

# Session configuration
SESSION_EXPIRY_HOURS = 24
MAX_SESSIONS = 100


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


# === Graph State for Visualization ===

graph_state = {
  "nodes": [],       # All agent nodes
  "edges": [],       # Parent-child relationships
  "reports": {},     # Agent reports/findings
  "transcripts": {}, # Agent conversation logs
  "activity": [],    # Recent activity feed
  "status": "idle",  # idle, running, completed
}
graph_lock = asyncio.Lock()
MAX_ACTIVITY_ITEMS = 100


async def add_node(node_id: str, name: str, node_type: str, parent_id: str = None, meta: dict = None):
  """Add a node to the visualization graph."""
  async with graph_lock:
    node = {
      "id": node_id,
      "name": name,
      "type": node_type,  # root, region, service, runner, diagnostic-agent, sub-agent, synthesis
      "status": "pending",
      "parent": parent_id,
      "meta": meta or {},
      "created_at": datetime.utcnow().isoformat(),
    }
    graph_state["nodes"].append(node)
    if parent_id:
      graph_state["edges"].append({"source": parent_id, "target": node_id})
    return node


async def add_edge(source_id: str, target_id: str):
  """Add an edge between two nodes."""
  async with graph_lock:
    graph_state["edges"].append({"source": source_id, "target": target_id})


async def update_node_status(node_id: str, status: str, report: dict = None):
  """Update a node's status and optionally add a report."""
  async with graph_lock:
    for node in graph_state["nodes"]:
      if node["id"] == node_id:
        node["status"] = status
        node["updated_at"] = datetime.utcnow().isoformat()
        break
    if report:
      graph_state["reports"][node_id] = report


async def add_transcript_entry(node_id: str, role: str, content: str, entry_type: str = "message"):
  """Add an entry to a node's transcript and activity feed."""
  async with graph_lock:
    if node_id not in graph_state["transcripts"]:
      graph_state["transcripts"][node_id] = []

    entry = {
      "timestamp": datetime.utcnow().isoformat(),
      "role": role,
      "content": content,
      "type": entry_type,
    }
    graph_state["transcripts"][node_id].append(entry)

    # Find node name for activity feed
    node_name = node_id
    for node in graph_state["nodes"]:
      if node["id"] == node_id:
        node_name = node.get("name", node_id)
        break

    # Add to activity feed
    activity_entry = {
      "timestamp": entry["timestamp"],
      "node_id": node_id,
      "node_name": node_name,
      "role": role,
      "content": content[:200] + ("..." if len(content) > 200 else ""),
      "type": entry_type,
    }
    graph_state["activity"].insert(0, activity_entry)

    # Trim activity feed
    if len(graph_state["activity"]) > MAX_ACTIVITY_ITEMS:
      graph_state["activity"] = graph_state["activity"][:MAX_ACTIVITY_ITEMS]


async def reset_graph():
  """Reset the graph state for a new session."""
  async with graph_lock:
    graph_state["nodes"] = []
    graph_state["edges"] = []
    graph_state["reports"] = {}
    graph_state["transcripts"] = {}
    graph_state["activity"] = []
    graph_state["status"] = "idle"

# HTTP client for calling mock 1Password server (only used in mock mode)
http_client = httpx.AsyncClient(timeout=CREDENTIAL_TIMEOUT)

# Mock 1Password server URL (running in same pod, only used in mock mode)
MOCK_1PASSWORD_URL = "http://localhost:8080"


# === Helper Functions ===

def make_event(event_type: str, **kwargs) -> str:
  """Create a JSON event with timestamp."""
  event = {
    "type": event_type,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    **kwargs
  }
  return json.dumps(event) + "\n"


async def cleanup_expired_sessions():
  """Remove sessions older than SESSION_EXPIRY_HOURS."""
  now = datetime.utcnow()
  expired = []
  for session_id, session in sessions.items():
    created = datetime.fromisoformat(session["created_at"].replace("Z", ""))
    if now - created > timedelta(hours=SESSION_EXPIRY_HOURS):
      expired.append(session_id)
  for session_id in expired:
    del sessions[session_id]
    logger.info(f"Cleaned up expired session: {session_id}")
  return len(expired)


def enforce_session_limit():
  """Remove oldest sessions if limit exceeded."""
  if len(sessions) >= MAX_SESSIONS:
    # Sort by created_at and remove oldest
    sorted_sessions = sorted(
      sessions.items(),
      key=lambda x: x[1]["created_at"]
    )
    to_remove = len(sessions) - MAX_SESSIONS + 1
    for session_id, _ in sorted_sessions[:to_remove]:
      del sessions[session_id]
      logger.info(f"Removed old session due to limit: {session_id}")


# === Credential Retrieval Functions ===

async def retrieve_credential_sdk(reference: str, session: dict) -> tuple[bool, str]:
  """
  Retrieve a credential from 1Password using the official SDK.
  Returns (success, message) tuple and stores credential in session.
  """
  # Normalize reference
  if not reference.startswith("op://"):
    reference = f"op://{reference}"

  last_error = None
  for attempt in range(CREDENTIAL_RETRY_ATTEMPTS):
    try:
      client = await get_onepassword_client()
      if client is None:
        return False, "1Password SDK client not initialized"

      # Use the SDK to resolve the secret reference
      value = await client.secrets.resolve(reference)

      # Store credential in session (for later use by diagnostic tools)
      if "credentials" not in session:
        session["credentials"] = {}
      session["credentials"][reference] = value

      # Track which credentials have been retrieved
      if "credentials_retrieved" not in session:
        session["credentials_retrieved"] = []
      if reference not in session["credentials_retrieved"]:
        session["credentials_retrieved"].append(reference)

      return True, f"Successfully retrieved: {reference}"

    except Exception as e:
      last_error = str(e)
      logger.warning(f"SDK credential retrieval attempt {attempt + 1} failed: {e}")

    # Wait before retry (except on last attempt)
    if attempt < CREDENTIAL_RETRY_ATTEMPTS - 1:
      await asyncio.sleep(CREDENTIAL_RETRY_DELAY * (attempt + 1))

  return False, f"Failed after {CREDENTIAL_RETRY_ATTEMPTS} attempts: {last_error}"


async def retrieve_credential_mock(reference: str, session: dict) -> tuple[bool, str]:
  """
  Retrieve a credential from the mock 1Password server.
  Returns (success, message) tuple and stores credential in session.
  """
  # Normalize reference
  if not reference.startswith("op://"):
    reference = f"op://{reference}"

  last_error = None
  for attempt in range(CREDENTIAL_RETRY_ATTEMPTS):
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

        return True, f"Successfully retrieved: {reference}"
      elif response.status_code == 404:
        return False, f"Not found: {reference}"
      else:
        last_error = f"HTTP {response.status_code}"

    except httpx.RequestError as e:
      last_error = str(e)
      logger.warning(f"Mock credential retrieval attempt {attempt + 1} failed: {e}")

    # Wait before retry (except on last attempt)
    if attempt < CREDENTIAL_RETRY_ATTEMPTS - 1:
      await asyncio.sleep(CREDENTIAL_RETRY_DELAY * (attempt + 1))

  return False, f"Failed after {CREDENTIAL_RETRY_ATTEMPTS} attempts: {last_error}"


async def retrieve_credential(reference: str, session: dict) -> tuple[bool, str]:
  """
  Retrieve a credential from 1Password with retry logic.
  Returns (success, message) tuple and stores credential in session.

  Uses SDK mode (real 1Password) or mock mode based on ONEPASSWORD_MODE env var.
  """
  if ONEPASSWORD_MODE == "sdk":
    return await retrieve_credential_sdk(reference, session)
  else:
    return await retrieve_credential_mock(reference, session)


# === Mock Diagnostic Tools ===

async def query_db_connections(environment: str) -> str:
  """Query database connection statistics."""
  # Return mock data based on environment
  if environment == "prod":
    return json.dumps({
      "environment": environment,
      "active_connections": 145,
      "max_connections": 200,
      "idle_connections": 23,
      "waiting_queries": 12,
      "avg_query_time_ms": 45,
      "connection_errors_last_hour": 3
    })
  else:
    return json.dumps({
      "environment": environment,
      "active_connections": 42,
      "max_connections": 100,
      "idle_connections": 15,
      "waiting_queries": 2,
      "avg_query_time_ms": 28,
      "connection_errors_last_hour": 0
    })


async def query_slow_queries(environment: str, threshold_ms: int = 1000) -> str:
  """Query slow database queries above threshold."""
  return json.dumps({
    "environment": environment,
    "threshold_ms": threshold_ms,
    "slow_queries": [
      {
        "query": "SELECT * FROM orders WHERE created_at > ? ORDER BY id DESC LIMIT 1000",
        "avg_duration_ms": 5420,
        "calls_last_hour": 156,
        "table": "orders"
      },
      {
        "query": "UPDATE inventory SET quantity = quantity - ? WHERE product_id = ?",
        "avg_duration_ms": 2340,
        "calls_last_hour": 89,
        "table": "inventory"
      },
      {
        "query": "SELECT u.*, o.* FROM users u JOIN orders o ON u.id = o.user_id WHERE u.status = ?",
        "avg_duration_ms": 1850,
        "calls_last_hour": 234,
        "table": "users, orders"
      }
    ]
  })


async def get_cloudwatch_metrics(service: str, metric: str, period_minutes: int = 60) -> str:
  """Get CloudWatch metrics for a service."""
  # Return mock metrics based on service/metric combination
  import random
  base_values = {
    "CPUUtilization": [75.2, 82.1, 91.5, 88.3, 79.6, 85.4, 92.1, 87.3],
    "MemoryUtilization": [68.5, 72.3, 78.9, 85.2, 82.1, 79.8, 76.4, 80.2],
    "NetworkIn": [1024000, 1256000, 980000, 1450000, 1120000, 1380000, 1290000, 1150000],
    "NetworkOut": [512000, 678000, 590000, 720000, 650000, 710000, 680000, 620000],
    "DiskReadOps": [450, 520, 480, 610, 550, 590, 530, 490],
    "DiskWriteOps": [280, 320, 290, 380, 340, 360, 310, 300],
  }

  values = base_values.get(metric, [random.uniform(50, 90) for _ in range(8)])
  unit_map = {
    "CPUUtilization": "Percent",
    "MemoryUtilization": "Percent",
    "NetworkIn": "Bytes",
    "NetworkOut": "Bytes",
    "DiskReadOps": "Count",
    "DiskWriteOps": "Count",
  }

  return json.dumps({
    "service": service,
    "metric": metric,
    "period_minutes": period_minutes,
    "datapoints": [
      {"timestamp": f"2024-01-15T{10+i}:00:00Z", "value": v}
      for i, v in enumerate(values)
    ],
    "unit": unit_map.get(metric, "None"),
    "statistics": {
      "min": min(values),
      "max": max(values),
      "avg": sum(values) / len(values)
    }
  })


async def check_instance_health(instance_id: str) -> str:
  """Check EC2/RDS instance health status."""
  return json.dumps({
    "instance_id": instance_id,
    "instance_type": "m5.xlarge",
    "status": "running",
    "health_checks": {
      "system_status": "ok",
      "instance_status": "ok"
    },
    "metrics": {
      "cpu_utilization": 78.5,
      "memory_utilization": 85.2,
      "disk_io_wait": 12.3,
      "network_packets_dropped": 0
    },
    "recent_events": [
      {"timestamp": "2024-01-15T08:00:00Z", "event": "Instance started"},
      {"timestamp": "2024-01-15T10:30:00Z", "event": "High CPU alert triggered"}
    ]
  })


async def check_kubernetes_pods(namespace: str, label_selector: str = "") -> str:
  """Check Kubernetes pod status and health."""
  return json.dumps({
    "namespace": namespace,
    "label_selector": label_selector or "app=api",
    "pods": [
      {
        "name": "api-deployment-7d4f8b6c9-abc12",
        "status": "Running",
        "restarts": 3,
        "ready": "1/1",
        "age": "2d",
        "cpu_usage": "250m",
        "memory_usage": "512Mi",
        "last_restart_reason": "OOMKilled"
      },
      {
        "name": "api-deployment-7d4f8b6c9-def34",
        "status": "Running",
        "restarts": 0,
        "ready": "1/1",
        "age": "2d",
        "cpu_usage": "180m",
        "memory_usage": "384Mi",
        "last_restart_reason": None
      },
      {
        "name": "api-deployment-7d4f8b6c9-ghi56",
        "status": "CrashLoopBackOff",
        "restarts": 15,
        "ready": "0/1",
        "age": "1h",
        "cpu_usage": "0m",
        "memory_usage": "0Mi",
        "last_restart_reason": "Error"
      }
    ],
    "summary": {
      "total": 3,
      "running": 2,
      "failed": 1,
      "pending": 0
    }
  })


async def get_application_logs(service: str, level: str = "ERROR", limit: int = 10) -> str:
  """Get recent application logs filtered by level."""
  return json.dumps({
    "service": service,
    "level": level,
    "limit": limit,
    "logs": [
      {
        "timestamp": "2024-01-15T10:45:23Z",
        "level": "ERROR",
        "message": "Database connection timeout after 30000ms",
        "source": "db-pool",
        "trace_id": "abc123"
      },
      {
        "timestamp": "2024-01-15T10:44:18Z",
        "level": "ERROR",
        "message": "Failed to acquire connection from pool: pool exhausted",
        "source": "db-pool",
        "trace_id": "def456"
      },
      {
        "timestamp": "2024-01-15T10:43:55Z",
        "level": "ERROR",
        "message": "Request timeout: /api/orders took 45023ms",
        "source": "http-handler",
        "trace_id": "ghi789"
      },
      {
        "timestamp": "2024-01-15T10:42:30Z",
        "level": "ERROR",
        "message": "Memory pressure detected: heap usage at 92%",
        "source": "memory-monitor",
        "trace_id": "jkl012"
      }
    ]
  })


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


# === Specialized Agent Instructions ===

DB_DIAGNOSTIC_INSTRUCTIONS = """You are a database diagnostic specialist agent.

Your role is to investigate database-related issues using the available diagnostic tools.

DIAGNOSTIC APPROACH:
1. Check connection pool status to identify exhaustion or leaks
2. Query for slow queries that may be holding connections
3. Analyze query patterns for optimization opportunities
4. Look for connection errors and their patterns

Use the tools provided to gather data, then provide a structured analysis.

OUTPUT FORMAT:
## Database Health Assessment
[Overall health status: CRITICAL/WARNING/HEALTHY]

## Connection Pool Analysis
[Details from query_db_connections]

## Slow Query Analysis
[Details from query_slow_queries]

## Root Cause Hypothesis
[Your assessment based on the data]

## Recommended Actions
1. [Immediate action]
2. [Short-term fix]
3. [Long-term improvement]

Be data-driven in your analysis. Reference specific metrics from the tools."""


CLOUD_DIAGNOSTIC_INSTRUCTIONS = """You are a cloud infrastructure diagnostic specialist agent.

Your role is to investigate AWS/cloud infrastructure issues using the available diagnostic tools.

DIAGNOSTIC APPROACH:
1. Check instance health and status
2. Review CloudWatch metrics for CPU, memory, network, disk
3. Identify resource bottlenecks or anomalies
4. Check for scaling issues or capacity problems

Use the tools provided to gather data, then provide a structured analysis.

OUTPUT FORMAT:
## Infrastructure Health Assessment
[Overall health status: CRITICAL/WARNING/HEALTHY]

## Instance Status
[Details from check_instance_health]

## Resource Metrics Analysis
[Details from get_cloudwatch_metrics]

## Anomalies Detected
[List any concerning patterns]

## Recommended Actions
1. [Immediate action]
2. [Scaling recommendation]
3. [Monitoring improvement]

Focus on actionable insights backed by metric data."""


K8S_DIAGNOSTIC_INSTRUCTIONS = """You are a Kubernetes diagnostic specialist agent.

Your role is to investigate Kubernetes cluster and pod issues using the available diagnostic tools.

DIAGNOSTIC APPROACH:
1. Check pod status and health across the namespace
2. Look for crash loops, OOM kills, or failed deployments
3. Analyze resource usage patterns
4. Review recent events and restart reasons

Use the tools provided to gather data, then provide a structured analysis.

OUTPUT FORMAT:
## Kubernetes Health Assessment
[Overall health status: CRITICAL/WARNING/HEALTHY]

## Pod Status Summary
[Details from check_kubernetes_pods]

## Issues Identified
[List pods with problems and their symptoms]

## Root Cause Analysis
[Your assessment based on restart reasons, resource usage]

## Recommended Actions
1. [Immediate remediation]
2. [Resource adjustment]
3. [Deployment fix]

Focus on getting unhealthy pods back to running state."""


# === FastAPI App ===

app = FastAPI(
  title="SRE Incident Diagnosis",
  description="Diagnose infrastructure problems with autonomous agents",
  version=VERSION
)


@app.get("/")
async def index():
  """Serve the dashboard."""
  return FileResponse("index.html")


# === Graph Visualization Endpoints ===

@app.get("/graph")
async def get_graph():
  """Return current graph state for visualization."""
  async with graph_lock:
    return JSONResponse(content={
      "nodes": graph_state["nodes"],
      "edges": graph_state["edges"],
      "status": graph_state["status"],
    })


@app.get("/graph/report/{node_id}")
async def get_node_report(node_id: str):
  """Return report and transcript for a specific node."""
  async with graph_lock:
    return JSONResponse(content={
      "report": graph_state["reports"].get(node_id),
      "transcript": graph_state["transcripts"].get(node_id, []),
    })


@app.get("/activity")
async def get_activity():
  """Return recent activity feed."""
  async with graph_lock:
    return JSONResponse(content=graph_state["activity"][:50])


@app.post("/graph/reset")
async def reset_graph_endpoint():
  """Reset the graph state."""
  await reset_graph()
  return JSONResponse(content={"status": "reset"})


@app.get("/health")
async def health():
  """Health check endpoint."""
  onepass_status = "unknown"
  onepass_mode = ONEPASSWORD_MODE

  if ONEPASSWORD_MODE == "sdk":
    # Check if SDK client can be initialized
    try:
      client = await get_onepassword_client()
      if client is not None:
        onepass_status = "healthy (sdk)"
      else:
        onepass_status = "not configured (missing OP_SERVICE_ACCOUNT_TOKEN)"
    except ImportError:
      onepass_status = "error (onepassword-sdk not installed)"
    except Exception as e:
      onepass_status = f"error ({type(e).__name__}: {str(e)[:50]})"
  else:
    # Check mock 1Password server
    try:
      response = await http_client.get(f"{MOCK_1PASSWORD_URL}/health")
      if response.status_code == 200:
        onepass_status = "healthy (mock)"
      else:
        onepass_status = f"unhealthy (HTTP {response.status_code})"
    except httpx.RequestError as e:
      onepass_status = f"unreachable ({type(e).__name__})"

  return JSONResponse(
    content={
      "status": "healthy",
      "service": "sre-diagnose",
      "version": VERSION,
      "active_sessions": len(sessions),
      "onepassword_mode": onepass_mode,
      "dependencies": {
        "onepassword": onepass_status
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
  # Enforce session limits
  enforce_session_limit()

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

  # === Graph Visualization: Create root and analysis nodes ===
  root_id = f"investigation-{session_id}"
  analysis_node_id = f"analysis-{session_id}"

  # Reset graph for new session and create root node
  await reset_graph()
  await add_node(root_id, "Incident Investigation", "root", meta={
    "problem": request.problem,
    "environment": request.environment,
    "session_id": session_id,
  })
  await update_node_status(root_id, "running")
  graph_state["status"] = "running"

  # Create analysis agent node
  await add_node(analysis_node_id, "Analysis Agent", "diagnostic-agent", root_id, meta={
    "agent_name": agent_name,
  })
  await update_node_status(analysis_node_id, "running")
  await add_transcript_entry(analysis_node_id, "system", f"Starting analysis: {request.problem[:100]}...")

  # Store graph node IDs in session for later use
  sessions[session_id]["root_node_id"] = root_id
  sessions[session_id]["analysis_node_id"] = analysis_node_id

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
      yield make_event("session_start",
        session_id=session_id,
        status="analyzing",
        phase="analysis",
        progress=0
      )

      await add_transcript_entry(analysis_node_id, "agent", "Beginning incident analysis...")

      # Stream the agent's analysis
      try:
        async for response in agent.send_stream(message, timeout=ANALYSIS_TIMEOUT):
          snippet = response.snippet
          text = extract_text_from_snippet(snippet)

          if text:
            full_analysis += text
            yield make_event("text", text=text)
      except asyncio.TimeoutError:
        raise Exception(f"Analysis timed out after {ANALYSIS_TIMEOUT} seconds")

      # Store analysis
      session["analysis"] = full_analysis

      # Update graph with analysis completion
      await add_transcript_entry(analysis_node_id, "model", full_analysis[:500] + "..." if len(full_analysis) > 500 else full_analysis)
      await update_node_status(analysis_node_id, "completed", {"analysis": full_analysis})

      # Extract requested credentials from analysis
      requested_creds = extract_credential_refs(full_analysis)
      session["requested_credentials"] = requested_creds

      # Update status
      session["status"] = "waiting_for_approval"
      session["phase"] = "awaiting_approval"

      await add_transcript_entry(analysis_node_id, "system", f"Identified {len(requested_creds)} credential(s) needed: {', '.join(requested_creds)}")

      yield make_event("approval_required",
        session_id=session_id,
        status="waiting_for_approval",
        requested_credentials=requested_creds,
        message=f"Analysis complete. {len(requested_creds)} credentials requested. Approve to continue.",
        progress=100
      )

    except Exception as e:
      logger.error(f"Error in analysis: {str(e)}")
      import traceback
      logger.error(f"Traceback: {traceback.format_exc()}")
      session["status"] = "error"
      session["phase"] = "analysis_failed"
      session["message"] = str(e)

      # Update graph with error status
      await update_node_status(analysis_node_id, "error", {"error": str(e)})
      await add_transcript_entry(analysis_node_id, "error", f"Analysis failed: {str(e)}")

      yield make_event("error",
        session_id=session_id,
        message=str(e)
      )

    finally:
      # Clean up analysis agent
      try:
        await Agent.stop(node, agent_name)
      except Exception as cleanup_error:
        logger.warning(f"Failed to cleanup agent {agent_name}: {cleanup_error}")

  return StreamingResponse(stream_response(), media_type="application/x-ndjson")


# === Specialized Agent Runner Functions ===

async def run_db_diagnosis(node: Node, session: dict, credentials: dict) -> dict:
  """Run database-focused diagnosis with specialized agent."""
  agent_name = f"db_diag_{session['id']}_{secrets.token_hex(4)}"
  start_time = datetime.utcnow()

  try:
    agent = await Agent.start(
      node=node,
      name=agent_name,
      instructions=DB_DIAGNOSTIC_INSTRUCTIONS,
      model=Model("claude-sonnet-4-5"),
      tools=[
        Tool(query_db_connections),
        Tool(query_slow_queries),
      ]
    )

    environment = session.get("environment", "prod")
    message = f"""Investigate database issues for environment: {environment}

Original problem: {session.get('problem')}

Use your tools to gather diagnostic data and provide your analysis."""

    findings = ""
    try:
      async for response in agent.send_stream(message, timeout=SPECIALIST_TIMEOUT):
        text = extract_text_from_snippet(response.snippet)
        if text:
          findings += text
    except asyncio.TimeoutError:
      findings += f"\n\n[WARNING: Analysis timed out after {SPECIALIST_TIMEOUT}s]"

    duration = (datetime.utcnow() - start_time).total_seconds()
    return {
      "agent": "database_specialist",
      "status": "completed",
      "findings": findings,
      "duration_seconds": round(duration, 2)
    }
  except Exception as e:
    logger.error(f"DB diagnosis error: {e}")
    duration = (datetime.utcnow() - start_time).total_seconds()
    return {
      "agent": "database_specialist",
      "status": "error",
      "findings": f"Error running database diagnosis: {str(e)}",
      "duration_seconds": round(duration, 2)
    }
  finally:
    try:
      await Agent.stop(node, agent_name)
    except Exception as cleanup_error:
      logger.warning(f"Failed to cleanup agent {agent_name}: {cleanup_error}")


async def run_cloud_diagnosis(node: Node, session: dict, credentials: dict) -> dict:
  """Run cloud infrastructure diagnosis with specialized agent."""
  agent_name = f"cloud_diag_{session['id']}_{secrets.token_hex(4)}"
  start_time = datetime.utcnow()

  try:
    agent = await Agent.start(
      node=node,
      name=agent_name,
      instructions=CLOUD_DIAGNOSTIC_INSTRUCTIONS,
      model=Model("claude-sonnet-4-5"),
      tools=[
        Tool(get_cloudwatch_metrics),
        Tool(check_instance_health),
      ]
    )

    environment = session.get("environment", "prod")
    message = f"""Investigate cloud infrastructure issues for environment: {environment}

Original problem: {session.get('problem')}

Use your tools to gather diagnostic data and provide your analysis."""

    findings = ""
    try:
      async for response in agent.send_stream(message, timeout=SPECIALIST_TIMEOUT):
        text = extract_text_from_snippet(response.snippet)
        if text:
          findings += text
    except asyncio.TimeoutError:
      findings += f"\n\n[WARNING: Analysis timed out after {SPECIALIST_TIMEOUT}s]"

    duration = (datetime.utcnow() - start_time).total_seconds()
    return {
      "agent": "cloud_specialist",
      "status": "completed",
      "findings": findings,
      "duration_seconds": round(duration, 2)
    }
  except Exception as e:
    logger.error(f"Cloud diagnosis error: {e}")
    duration = (datetime.utcnow() - start_time).total_seconds()
    return {
      "agent": "cloud_specialist",
      "status": "error",
      "findings": f"Error running cloud diagnosis: {str(e)}",
      "duration_seconds": round(duration, 2)
    }
  finally:
    try:
      await Agent.stop(node, agent_name)
    except Exception as cleanup_error:
      logger.warning(f"Failed to cleanup agent {agent_name}: {cleanup_error}")


async def run_k8s_diagnosis(node: Node, session: dict, credentials: dict) -> dict:
  """Run Kubernetes diagnosis with specialized agent."""
  agent_name = f"k8s_diag_{session['id']}_{secrets.token_hex(4)}"
  start_time = datetime.utcnow()

  try:
    agent = await Agent.start(
      node=node,
      name=agent_name,
      instructions=K8S_DIAGNOSTIC_INSTRUCTIONS,
      model=Model("claude-sonnet-4-5"),
      tools=[
        Tool(check_kubernetes_pods),
        Tool(get_application_logs),
      ]
    )

    environment = session.get("environment", "prod")
    message = f"""Investigate Kubernetes issues for environment: {environment}

Original problem: {session.get('problem')}

Use your tools to gather diagnostic data and provide your analysis."""

    findings = ""
    try:
      async for response in agent.send_stream(message, timeout=SPECIALIST_TIMEOUT):
        text = extract_text_from_snippet(response.snippet)
        if text:
          findings += text
    except asyncio.TimeoutError:
      findings += f"\n\n[WARNING: Analysis timed out after {SPECIALIST_TIMEOUT}s]"

    duration = (datetime.utcnow() - start_time).total_seconds()
    return {
      "agent": "kubernetes_specialist",
      "status": "completed",
      "findings": findings,
      "duration_seconds": round(duration, 2)
    }
  except Exception as e:
    logger.error(f"K8s diagnosis error: {e}")
    duration = (datetime.utcnow() - start_time).total_seconds()
    return {
      "agent": "kubernetes_specialist",
      "status": "error",
      "findings": f"Error running Kubernetes diagnosis: {str(e)}",
      "duration_seconds": round(duration, 2)
    }
  finally:
    try:
      await Agent.stop(node, agent_name)
    except Exception as cleanup_error:
      logger.warning(f"Failed to cleanup agent {agent_name}: {cleanup_error}")


def determine_specialist_agents(problem: str, analysis: str) -> list:
  """Determine which specialist agents to run based on the problem and analysis."""
  agents = []

  problem_lower = problem.lower()
  analysis_lower = analysis.lower()
  combined = problem_lower + " " + analysis_lower

  # Database indicators
  db_keywords = ["database", "db", "connection pool", "query", "sql", "postgres", "mysql",
                 "timeout", "connection", "transaction", "deadlock"]
  if any(kw in combined for kw in db_keywords):
    agents.append("database")

  # Cloud/AWS indicators
  cloud_keywords = ["aws", "ec2", "cloudwatch", "instance", "cpu", "memory usage",
                    "scaling", "load balancer", "rds"]
  if any(kw in combined for kw in cloud_keywords):
    agents.append("cloud")

  # Kubernetes indicators
  k8s_keywords = ["kubernetes", "k8s", "pod", "container", "deployment", "replica",
                  "crashloop", "oom", "health check", "liveness", "readiness"]
  if any(kw in combined for kw in k8s_keywords):
    agents.append("kubernetes")

  # If no specific indicators, run database (most common) and cloud
  if not agents:
    agents = ["database", "cloud"]

  return agents


@app.post("/approve/{session_id}")
async def approve(session_id: str, request: ApproveRequest, node: NodeDep):
  """Approve or deny credential access - Phase 2: Diagnosis with specialized agents."""
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

    # Update graph with denial
    root_id = session.get("root_node_id")
    if root_id:
      await update_node_status(root_id, "error")
      await add_transcript_entry(root_id, "system", "Credential access denied by user")
      graph_state["status"] = "completed"

    return JSONResponse(content={
      "session_id": session_id,
      "status": "denied",
      "message": "Credential access denied, diagnosis cancelled"
    })

  # Approved - retrieve credentials and start diagnosis
  session["status"] = "retrieving_credentials"
  session["phase"] = "credential_retrieval"

  async def stream_diagnosis():
    synthesis_agent_name = None
    diagnosis_start_time = datetime.utcnow()

    # Get graph node IDs from session
    root_id = session.get("root_node_id", f"investigation-{session_id}")
    specialist_node_ids = {}

    try:
      yield make_event("approval_accepted",
        session_id=session_id,
        status="approved",
        progress=0
      )

      await add_transcript_entry(root_id, "system", "Credentials approved, starting diagnosis")

      # Retrieve all requested credentials
      requested = session.get("requested_credentials", [])
      total_creds = len(requested)
      successful_creds = 0
      failed_creds = []

      for i, ref in enumerate(requested):
        success, result = await retrieve_credential(ref, session)
        if success:
          successful_creds += 1
        else:
          failed_creds.append(ref)

        yield make_event("credential_retrieved",
          reference=ref,
          success=success,
          message=result,
          progress=int((i + 1) / total_creds * 20) if total_creds > 0 else 20
        )
        logger.info(f"Credential retrieval: {result}")

      # Log warning if some credentials failed
      if failed_creds:
        logger.warning(f"Failed to retrieve {len(failed_creds)} credentials: {failed_creds}")

      await add_transcript_entry(root_id, "system", f"Retrieved {successful_creds}/{total_creds} credentials")

      # Update status
      session["status"] = "diagnosing"
      session["phase"] = "diagnosis"

      yield make_event("diagnosis_started",
        session_id=session_id,
        credentials_retrieved=successful_creds,
        credentials_failed=len(failed_creds),
        progress=20
      )

      # Build diagnosis prompt with context from analysis
      creds_summary = ", ".join(session.get("credentials_retrieved", [])) or "none"
      # Determine which specialist agents to run
      specialists = determine_specialist_agents(
        session.get("problem", ""),
        session.get("analysis", "")
      )

      yield make_event("specialists_selected",
        session_id=session_id,
        specialists=specialists,
        progress=25
      )

      # Create specialist agent nodes in graph
      for specialist in specialists:
        specialist_node_id = f"{specialist}-specialist-{session_id}"
        specialist_node_ids[specialist] = specialist_node_id
        await add_node(specialist_node_id, f"{specialist.title()} Specialist", "diagnostic-agent", root_id, meta={
          "specialist_type": specialist,
        })
        await update_node_status(specialist_node_id, "running")
        await add_transcript_entry(specialist_node_id, "system", f"Starting {specialist} diagnosis...")

      await add_transcript_entry(root_id, "system", f"Running {len(specialists)} specialist agents in parallel: {', '.join(specialists)}")

      # Run specialist agents in parallel
      specialist_tasks = []
      credentials = session.get("credentials", {})

      for specialist in specialists:
        if specialist == "database":
          specialist_tasks.append(run_db_diagnosis(node, session, credentials))
        elif specialist == "cloud":
          specialist_tasks.append(run_cloud_diagnosis(node, session, credentials))
        elif specialist == "kubernetes":
          specialist_tasks.append(run_k8s_diagnosis(node, session, credentials))

      # Run all specialists in parallel with overall timeout
      try:
        specialist_results = await asyncio.wait_for(
          asyncio.gather(*specialist_tasks, return_exceptions=True),
          timeout=SPECIALIST_TIMEOUT * 1.5  # Allow 50% extra for parallel execution
        )
      except asyncio.TimeoutError:
        logger.error("Specialist agents timed out")
        specialist_results = [
          {"agent": "specialists", "status": "timeout", "findings": "Specialist agents timed out"}
        ]

      # Collect findings from specialists
      all_findings = []
      completed_count = 0
      for result in specialist_results:
        if isinstance(result, Exception):
          yield make_event("specialist_error",
            error=str(result)
          )
        else:
          all_findings.append(result)
          completed_count += 1

          # Update graph node for this specialist
          agent_name = result.get("agent", "unknown")
          # Map agent name back to specialist type
          specialist_type = agent_name.replace("_specialist", "").replace("_", "-")
          for st, node_id in specialist_node_ids.items():
            if st in specialist_type or specialist_type in st:
              status = "completed" if result.get("status") == "completed" else "error"
              await update_node_status(node_id, status, result)
              await add_transcript_entry(node_id, "model", result.get("findings", "")[:300] + "..." if len(result.get("findings", "")) > 300 else result.get("findings", ""))
              break

          yield make_event("specialist_complete",
            agent=result.get("agent"),
            status=result.get("status"),
            duration_seconds=result.get("duration_seconds"),
            progress=25 + int(completed_count / len(specialist_tasks) * 35) if specialist_tasks else 60
          )

      # Store specialist findings
      session["specialist_findings"] = all_findings

      # Now run synthesizer agent to combine findings
      yield make_event("synthesis_started",
        session_id=session_id,
        findings_count=len(all_findings),
        progress=60
      )

      # Create synthesis agent node in graph
      synthesis_node_id = f"synthesis-{session_id}"
      await add_node(synthesis_node_id, "Synthesis Agent", "synthesis", root_id, meta={
        "findings_count": len(all_findings),
      })
      await update_node_status(synthesis_node_id, "running")
      await add_transcript_entry(synthesis_node_id, "system", f"Synthesizing findings from {len(all_findings)} specialists...")

      # Create synthesis agent to combine all findings
      synthesis_agent_name = f"synthesizer_{session_id}_{secrets.token_hex(4)}"
      agent = await Agent.start(
        node=node,
        name=synthesis_agent_name,
        instructions=DIAGNOSIS_INSTRUCTIONS,
        model=Model("claude-sonnet-4-5"),
      )

      # Build synthesis prompt
      findings_text = "\n\n".join([
        f"## {f.get('agent', 'Unknown')} Findings:\n{f.get('findings', 'No findings')}"
        for f in all_findings
      ])

      diagnosis_message = f"""Synthesize the diagnosis for this incident based on specialist agent findings.

**Original Problem**: {session.get('problem')}
**Environment**: {session.get('environment')}

**Initial Analysis**:
{session.get('analysis', 'No analysis available')}

**Credentials Retrieved**: {creds_summary}

**Specialist Agent Findings**:
{findings_text}

Based on all the specialist findings above, provide your final synthesized diagnosis and remediation recommendations.
Focus on:
1. The most likely root cause (considering all specialist findings)
2. Immediate remediation steps
3. Long-term prevention measures
4. Monitoring improvements"""

      # Stream diagnosis with timeout handling
      diagnosis_text = ""
      try:
        async for response in agent.send_stream(diagnosis_message, timeout=SYNTHESIS_TIMEOUT):
          snippet = response.snippet
          text = extract_text_from_snippet(snippet)

          if text:
            diagnosis_text += text
            yield make_event("text", text=text)
      except asyncio.TimeoutError:
        diagnosis_text += "\n\n[WARNING: Synthesis timed out, partial results shown]"
        yield make_event("text", text="\n\n[WARNING: Synthesis timed out]")

      # Update session with diagnosis
      session["diagnosis"] = diagnosis_text
      session["status"] = "completed"
      session["phase"] = "diagnosis_complete"
      session["completed_at"] = datetime.utcnow().isoformat() + "Z"

      # Update graph with completion
      await update_node_status(synthesis_node_id, "completed", {"diagnosis": diagnosis_text})
      await add_transcript_entry(synthesis_node_id, "model", diagnosis_text[:500] + "..." if len(diagnosis_text) > 500 else diagnosis_text)
      await update_node_status(root_id, "completed")
      await add_transcript_entry(root_id, "system", f"Diagnosis complete in {round((datetime.utcnow() - diagnosis_start_time).total_seconds(), 1)}s")
      graph_state["status"] = "completed"

      total_duration = (datetime.utcnow() - diagnosis_start_time).total_seconds()
      yield make_event("diagnosis_complete",
        session_id=session_id,
        status="completed",
        credentials_retrieved=session.get("credentials_retrieved", []),
        specialists_run=len(all_findings),
        duration_seconds=round(total_duration, 2),
        progress=100
      )

    except Exception as e:
      logger.error(f"Error in diagnosis: {str(e)}")
      import traceback
      logger.error(f"Traceback: {traceback.format_exc()}")
      session["status"] = "error"
      session["phase"] = "diagnosis_failed"
      session["message"] = str(e)

      # Update graph with error
      await update_node_status(root_id, "error")
      await add_transcript_entry(root_id, "error", f"Diagnosis failed: {str(e)}")
      graph_state["status"] = "completed"

      yield make_event("error",
        session_id=session_id,
        message=str(e)
      )

    finally:
      # Clean up synthesis agent
      if synthesis_agent_name:
        try:
          await Agent.stop(node, synthesis_agent_name)
        except Exception as cleanup_error:
          logger.warning(f"Failed to cleanup agent {synthesis_agent_name}: {cleanup_error}")

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
  return JSONResponse(content={"message": f"Session {session_id} deleted"})


@app.post("/sessions/cleanup")
async def cleanup_sessions():
  """Clean up expired sessions."""
  cleaned = await cleanup_expired_sessions()
  return JSONResponse(content={
    "message": f"Cleaned up {cleaned} expired sessions",
    "remaining_sessions": len(sessions)
  })


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

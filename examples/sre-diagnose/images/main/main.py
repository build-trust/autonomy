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


# === Specialized Agent Runner Functions ===

async def run_db_diagnosis(node: Node, session: dict, credentials: dict) -> dict:
  """Run database-focused diagnosis with specialized agent."""
  agent_name = f"db_diag_{session['id']}_{secrets.token_hex(4)}"

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
    async for response in agent.send_stream(message, timeout=90):
      text = extract_text_from_snippet(response.snippet)
      if text:
        findings += text

    return {
      "agent": "database_specialist",
      "status": "completed",
      "findings": findings
    }
  except Exception as e:
    logger.error(f"DB diagnosis error: {e}")
    return {
      "agent": "database_specialist",
      "status": "error",
      "findings": f"Error running database diagnosis: {str(e)}"
    }
  finally:
    try:
      await Agent.stop(node, agent_name)
    except Exception:
      pass


async def run_cloud_diagnosis(node: Node, session: dict, credentials: dict) -> dict:
  """Run cloud infrastructure diagnosis with specialized agent."""
  agent_name = f"cloud_diag_{session['id']}_{secrets.token_hex(4)}"

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
    async for response in agent.send_stream(message, timeout=90):
      text = extract_text_from_snippet(response.snippet)
      if text:
        findings += text

    return {
      "agent": "cloud_specialist",
      "status": "completed",
      "findings": findings
    }
  except Exception as e:
    logger.error(f"Cloud diagnosis error: {e}")
    return {
      "agent": "cloud_specialist",
      "status": "error",
      "findings": f"Error running cloud diagnosis: {str(e)}"
    }
  finally:
    try:
      await Agent.stop(node, agent_name)
    except Exception:
      pass


async def run_k8s_diagnosis(node: Node, session: dict, credentials: dict) -> dict:
  """Run Kubernetes diagnosis with specialized agent."""
  agent_name = f"k8s_diag_{session['id']}_{secrets.token_hex(4)}"

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
    async for response in agent.send_stream(message, timeout=90):
      text = extract_text_from_snippet(response.snippet)
      if text:
        findings += text

    return {
      "agent": "kubernetes_specialist",
      "status": "completed",
      "findings": findings
    }
  except Exception as e:
    logger.error(f"K8s diagnosis error: {e}")
    return {
      "agent": "kubernetes_specialist",
      "status": "error",
      "findings": f"Error running Kubernetes diagnosis: {str(e)}"
    }
  finally:
    try:
      await Agent.stop(node, agent_name)
    except Exception:
      pass


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
    return JSONResponse(content={
      "session_id": session_id,
      "status": "denied",
      "message": "Credential access denied, diagnosis cancelled"
    })

  # Approved - retrieve credentials and start diagnosis
  session["status"] = "retrieving_credentials"
  session["phase"] = "credential_retrieval"

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

      # Build diagnosis prompt with context from analysis
      creds_summary = ", ".join(session.get("credentials_retrieved", [])) or "none"
      # Determine which specialist agents to run
      specialists = determine_specialist_agents(
        session.get("problem", ""),
        session.get("analysis", "")
      )

      yield json.dumps({
        "type": "specialists_selected",
        "session_id": session_id,
        "specialists": specialists
      }) + "\n"

      # Run specialist agents in parallel
      import asyncio
      specialist_tasks = []
      credentials = session.get("credentials", {})

      for specialist in specialists:
        if specialist == "database":
          specialist_tasks.append(run_db_diagnosis(node, session, credentials))
        elif specialist == "cloud":
          specialist_tasks.append(run_cloud_diagnosis(node, session, credentials))
        elif specialist == "kubernetes":
          specialist_tasks.append(run_k8s_diagnosis(node, session, credentials))

      # Run all specialists in parallel
      specialist_results = await asyncio.gather(*specialist_tasks, return_exceptions=True)

      # Collect findings from specialists
      all_findings = []
      for result in specialist_results:
        if isinstance(result, Exception):
          yield json.dumps({
            "type": "specialist_error",
            "error": str(result)
          }) + "\n"
        else:
          all_findings.append(result)
          yield json.dumps({
            "type": "specialist_complete",
            "agent": result.get("agent"),
            "status": result.get("status")
          }) + "\n"

      # Store specialist findings
      session["specialist_findings"] = all_findings

      # Now run synthesizer agent to combine findings
      yield json.dumps({
        "type": "synthesis_started",
        "session_id": session_id
      }) + "\n"

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
      # Clean up synthesis agent
      try:
        await Agent.stop(node, synthesis_agent_name)
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

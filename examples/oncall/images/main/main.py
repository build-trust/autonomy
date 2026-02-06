"""OnCall - Incident Response with Autonomous Diagnostic Agents

Provides HTTP API for incident diagnosis with human-in-the-loop credential approval.
Uses a two-phase approach:
1. Analysis phase: Agent analyzes problem and identifies needed credentials
2. Diagnosis phase: After approval, new agent runs diagnosis with credentials

Features a long-running Monitor that:
- Runs persistently from app startup
- Detects anomalies (simulated for demo)
- Spawns diagnostic investigations when issues are detected
- Tracks investigation state for Cursor hook integration

Supports three 1Password modes:
- mock: Uses local mock 1Password server (default, for development)
- sdk: Uses real 1Password SDK with service account
- connect: Uses 1Password Connect server (self-hosted)

Configuration:
- mock: No additional config needed
- sdk: Set ONEPASSWORD_MODE=sdk and OP_SERVICE_ACCOUNT_TOKEN
- connect: Set ONEPASSWORD_MODE=connect, OP_CONNECT_HOST, and OP_CONNECT_TOKEN
"""

from autonomy import Agent, HttpServer, Model, Node, NodeDep, Tool
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import json
import secrets
import httpx
import re
import asyncio
import os
import random
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oncall")

# === 1Password Configuration ===

# Mode: "mock" (default), "sdk" (service account), or "connect" (1Password Connect)
ONEPASSWORD_MODE = os.environ.get("ONEPASSWORD_MODE", "mock").lower()

# 1Password SDK client (initialized lazily for sdk mode)
_op_client = None

# 1Password Connect configuration
OP_CONNECT_HOST = os.environ.get("OP_CONNECT_HOST", "http://localhost:8080")
OP_CONNECT_TOKEN = os.environ.get("OP_CONNECT_TOKEN", "")

# Cache for Connect vault/item lookups
_connect_vault_cache: Dict[str, str] = {}  # vault_name -> vault_id
_connect_item_cache: Dict[str, dict] = {}  # "vault_id/item_name" -> item_data

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
        integration_name="oncall",
        integration_version=VERSION
      )
      logger.info("1Password SDK client initialized successfully")
    except ImportError:
      raise ImportError("onepassword-sdk package not installed. Run: pip install onepassword-sdk")
  return _op_client


async def resolve_connect_reference(reference: str) -> str:
  """
  Resolve an op:// reference using 1Password Connect API.
  Reference format: op://vault/item/field
  Returns the field value.
  """
  global _connect_vault_cache, _connect_item_cache

  # Parse reference
  if not reference.startswith("op://"):
    raise ValueError(f"Invalid reference format: {reference}")

  parts = reference[5:].split("/")  # Remove "op://" prefix
  if len(parts) != 3:
    raise ValueError(f"Invalid reference format: {reference}. Expected op://vault/item/field")

  vault_name, item_name, field_name = parts

  headers = {
    "Authorization": f"Bearer {OP_CONNECT_TOKEN}",
    "Content-Type": "application/json",
  }

  async with httpx.AsyncClient(timeout=30.0) as client:
    # Get vault ID (cached)
    if vault_name not in _connect_vault_cache:
      vaults_resp = await client.get(f"{OP_CONNECT_HOST}/v1/vaults", headers=headers)
      vaults_resp.raise_for_status()
      vaults = vaults_resp.json()
      for vault in vaults:
        _connect_vault_cache[vault["name"]] = vault["id"]

    vault_id = _connect_vault_cache.get(vault_name)
    if not vault_id:
      raise ValueError(f"Vault not found: {vault_name}")

    # Get item (cached)
    cache_key = f"{vault_id}/{item_name}"
    if cache_key not in _connect_item_cache:
      # List items to find by title
      items_resp = await client.get(
        f"{OP_CONNECT_HOST}/v1/vaults/{vault_id}/items",
        headers=headers
      )
      items_resp.raise_for_status()
      items = items_resp.json()

      item_id = None
      for item in items:
        if item.get("title") == item_name:
          item_id = item["id"]
          break

      if not item_id:
        raise ValueError(f"Item not found: {item_name} in vault {vault_name}")

      # Get full item details
      item_resp = await client.get(
        f"{OP_CONNECT_HOST}/v1/vaults/{vault_id}/items/{item_id}",
        headers=headers
      )
      item_resp.raise_for_status()
      _connect_item_cache[cache_key] = item_resp.json()

    item_data = _connect_item_cache[cache_key]

    # Find the field value
    for field in item_data.get("fields", []):
      if field.get("label") == field_name or field.get("id") == field_name:
        return field.get("value", "")

    raise ValueError(f"Field not found: {field_name} in item {item_name}")


logger.info(f"1Password mode: {ONEPASSWORD_MODE}")

# === Configuration Constants ===

VERSION = "0.7.0"

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


class TriggerAnomalyRequest(BaseModel):
  """Request to manually trigger an anomaly for demo purposes."""
  anomaly_type: str = "cascading_failure"
  severity: str = "critical"
  message: Optional[str] = None


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
  requested_credentials: Optional[list] = None


# === Session State (in-memory for MVP) ===

sessions: dict = {}


# === Graph State for Visualization ===

# The OnCall Agent node - always present, represents the monitor
ONCALL_AGENT_NODE = {
  "id": "oncall-agent",
  "name": "OnCall Agent",
  "type": "oncall-agent",
  "status": "monitoring",  # Special status for heartbeat animation
  "parent": None,
  "meta": {"description": "Long-running monitor watching metrics"},
  "created_at": datetime.utcnow().isoformat(),
}

graph_state = {
  "nodes": [],       # All agent nodes (oncall-agent added dynamically)
  "edges": [],       # Parent-child relationships
  "reports": {},     # Agent reports/findings
  "transcripts": {}, # Agent conversation logs
  "activity": [],    # Recent activity feed
  "status": "idle",  # idle, running, completed
}
graph_lock = asyncio.Lock()
MAX_ACTIVITY_ITEMS = 100


# === Investigation State for monitor ===

class AgentStatus(str, Enum):
  """Status of the OnCall monitor."""
  INACTIVE = "inactive"  # Agent not running, needs activation
  WAITING_FOR_ACTIVATION_APPROVAL = "waiting_for_activation_approval"  # Waiting for READ creds to activate
  ACTIVATING = "activating"  # Processing activation
  MONITORING = "monitoring"  # Active and monitoring infrastructure
  INCIDENT_DETECTED = "incident_detected"  # Anomaly detected, spawning diagnostic agents
  DIAGNOSING = "diagnosing"  # Diagnostic agents investigating
  DIAGNOSIS_COMPLETE = "diagnosis_complete"  # Root cause found, preparing remediation
  WAITING_FOR_WRITE_APPROVAL = "waiting_for_write_approval"  # Waiting for WRITE creds to remediate
  REMEDIATING = "remediating"  # Executing remediation
  RESOLVED = "resolved"  # Incident resolved
  ERROR = "error"


# Keep old enum for backwards compatibility during refactor
class InvestigationStatus(str, Enum):
  """Status of a diagnostic investigation (legacy - use AgentStatus)."""
  IDLE = "idle"
  DETECTING = "detecting"
  SPAWNING = "spawning"
  WAITING_FOR_READ_APPROVAL = "waiting_for_read_approval"
  WAITING_FOR_WRITE_APPROVAL = "waiting_for_write_approval"
  RUNNING = "running"
  SYNTHESIZING = "synthesizing"
  COMPLETED = "completed"
  ERROR = "error"


# === Credential Categorization ===

class CredentialCategory(str, Enum):
  """Category of credential based on access level."""
  READ = "read"
  WRITE = "write"


@dataclass
class CredentialGrant:
  """Tracks a granted credential with scope."""
  reference: str
  category: CredentialCategory
  granted_at: str

  def to_dict(self) -> dict:
    return {
      "reference": self.reference,
      "category": self.category.value,
      "granted_at": self.granted_at,
    }


# Credential category mappings - determines which credentials require which approval
CREDENTIAL_CATEGORIES: Dict[str, CredentialCategory] = {
  # READ credentials - approved once, shared across investigation
  "op://Infrastructure/prod-db-readonly/password": CredentialCategory.READ,
  "op://Infrastructure/prod-db-readonly/username": CredentialCategory.READ,
  "op://Infrastructure/prod-db-readonly/server": CredentialCategory.READ,
  "op://Infrastructure/aws-cloudwatch/credential": CredentialCategory.READ,
  "op://Infrastructure/k8s-prod-readonly/credential": CredentialCategory.READ,
  # WRITE credentials - require separate approval per action
  "op://Infrastructure/prod-db-rwaccess/username": CredentialCategory.WRITE,
  "op://Infrastructure/prod-db-rwaccess/password": CredentialCategory.WRITE,
  "op://Infrastructure/config-service/credential": CredentialCategory.WRITE,
}

# READ credentials needed for monitoring (requested at activation)
MONITORING_READ_CREDENTIALS = [
  "op://Infrastructure/prod-db-readonly/password",
  "op://Infrastructure/prod-db-readonly/username",
  "op://Infrastructure/prod-db-readonly/server",
  "op://Infrastructure/aws-cloudwatch/credential",
  "op://Infrastructure/k8s-prod-readonly/credential",
]

# Default READ credentials needed for diagnostic investigation (legacy)
DEFAULT_READ_CREDENTIALS = MONITORING_READ_CREDENTIALS


@dataclass
class OnCallAgentState:
  """State of the OnCall monitor."""
  status: AgentStatus = AgentStatus.INACTIVE
  activated_at: Optional[str] = None
  monitoring_since: Optional[str] = None

  # Credentials granted during activation
  read_credentials: Dict[str, str] = field(default_factory=dict)
  read_credential_refs: List[str] = field(default_factory=list)

  # Current incident (if any)
  incident_id: Optional[str] = None
  incident_detected_at: Optional[str] = None
  anomaly_type: Optional[str] = None
  anomaly_message: Optional[str] = None

  # Diagnosis state
  diagnosis_started_at: Optional[str] = None
  diagnosis_completed_at: Optional[str] = None
  diagnosis_result: Optional[str] = None
  agents_deployed: int = 0

  # Remediation state
  pending_write_action: Optional[dict] = None
  write_actions_completed: List[dict] = field(default_factory=list)
  write_actions_denied: List[dict] = field(default_factory=list)
  write_actions_remaining: List[dict] = field(default_factory=list)
  remediation_progress: int = 0  # 0-100 percent
  remediation_message: Optional[str] = None
  error_rate_current: int = 340  # Current error rate increase percentage

  # Audit trail
  credential_grants: List[CredentialGrant] = field(default_factory=list)

  # Error state
  error_message: Optional[str] = None

  def to_dict(self) -> dict:
    # Only include diagnosis_result when diagnosis is complete
    # to avoid revealing root cause before investigation finishes
    diagnosis_complete_statuses = [
      AgentStatus.DIAGNOSIS_COMPLETE,
      AgentStatus.WAITING_FOR_WRITE_APPROVAL,
      AgentStatus.REMEDIATING,
      AgentStatus.RESOLVED,
    ]
    include_diagnosis = self.status in diagnosis_complete_statuses

    return {
      "status": self.status.value,
      "activated_at": self.activated_at,
      "monitoring_since": self.monitoring_since,
      "read_credential_refs": self.read_credential_refs,
      "incident_id": self.incident_id,
      "incident_detected_at": self.incident_detected_at,
      "anomaly_type": self.anomaly_type,
      "anomaly_message": self.anomaly_message,
      "diagnosis_started_at": self.diagnosis_started_at,
      "diagnosis_completed_at": self.diagnosis_completed_at,
      "diagnosis_result": self.diagnosis_result if include_diagnosis else None,
      "agents_deployed": self.agents_deployed,
      "pending_write_action": self.pending_write_action,
      "write_actions_completed": self.write_actions_completed,
      "write_actions_denied": self.write_actions_denied,
      "write_actions_remaining": [a for a in self.write_actions_remaining],
      "credential_grants": [g.to_dict() for g in self.credential_grants],
      "error_message": self.error_message,
      "remediation_progress": self.remediation_progress,
      "remediation_message": self.remediation_message,
      "error_rate_current": self.error_rate_current,
    }

  def add_credential_grant(self, reference: str, category: CredentialCategory) -> CredentialGrant:
    """Record a credential grant."""
    grant = CredentialGrant(
      reference=reference,
      category=category,
      granted_at=datetime.utcnow().isoformat() + "Z",
    )
    self.credential_grants.append(grant)
    return grant

  def reset_incident(self):
    """Reset incident state for a new incident."""
    self.incident_id = None
    self.incident_detected_at = None
    self.anomaly_type = None
    self.anomaly_message = None
    self.diagnosis_started_at = None
    self.diagnosis_completed_at = None
    self.diagnosis_result = None
    self.agents_deployed = 0
    self.pending_write_action = None
    self.write_actions_completed = []
    self.write_actions_denied = []
    self.write_actions_remaining = []
    self.remediation_progress = 0
    self.remediation_message = None
    self.error_rate_current = 340


# Global OnCall agent state
oncall_state: OnCallAgentState = OnCallAgentState()
oncall_state_lock = asyncio.Lock()


def get_credential_category(reference: str) -> CredentialCategory:
  """Get the category for a credential reference."""
  # Normalize reference
  if not reference.startswith("op://"):
    reference = f"op://{reference}"
  return CREDENTIAL_CATEGORIES.get(reference, CredentialCategory.READ)


def categorize_credentials(references: List[str]) -> Dict[CredentialCategory, List[str]]:
  """Categorize a list of credential references into READ and WRITE groups."""
  result = {CredentialCategory.READ: [], CredentialCategory.WRITE: []}
  for ref in references:
    category = get_credential_category(ref)
    result[category].append(ref)
  return result


@dataclass
class InvestigationState:
  """State of an active diagnostic investigation.

  Tracks the lifecycle of an investigation from anomaly detection through diagnosis completion.
  Cursor hooks use this to connect to running investigations.
  """
  investigation_id: str
  status: InvestigationStatus = InvestigationStatus.IDLE
  created_at: str = ""
  anomaly_detected_at: Optional[str] = None
  investigation_started_at: Optional[str] = None
  completed_at: Optional[str] = None
  session_id: Optional[str] = None
  root_node_id: Optional[str] = None
  anomaly_type: str = ""
  anomaly_message: str = ""
  agents_count: int = 0
  regions_count: int = 0
  services_count: int = 0
  findings_summary: Optional[str] = None
  diagnosis: Optional[str] = None
  error_message: Optional[str] = None
  # Credential tracking
  credential_grants: List[CredentialGrant] = field(default_factory=list)
  pending_write_request: Optional[Dict] = None  # For Step 5: tracks pending write credential request

  def to_dict(self) -> dict:
    """Convert to dictionary for JSON serialization.

    Note: findings_summary and diagnosis are only included when appropriate
    to avoid revealing root cause before investigation is complete.
    """
    result = {
      "investigation_id": self.investigation_id,
      "status": self.status.value,
      "created_at": self.created_at,
      "anomaly_detected_at": self.anomaly_detected_at,
      "investigation_started_at": self.investigation_started_at,
      "completed_at": self.completed_at,
      "session_id": self.session_id,
      "root_node_id": self.root_node_id,
      "anomaly_type": self.anomaly_type,
      "anomaly_message": self.anomaly_message,
      "agents_count": self.agents_count,
      "regions_count": self.regions_count,
      "services_count": self.services_count,
      "error_message": self.error_message,
      "credential_grants": [g.to_dict() for g in self.credential_grants],
      "pending_write_request": self.pending_write_request,
    }

    # Only include findings_summary during synthesizing or completed states
    # to avoid revealing root cause hints before investigation is complete
    if self.status in [InvestigationStatus.SYNTHESIZING, InvestigationStatus.COMPLETED]:
      result["findings_summary"] = self.findings_summary
    else:
      result["findings_summary"] = None

    # Only include diagnosis when investigation is completed
    if self.status == InvestigationStatus.COMPLETED:
      result["diagnosis"] = self.diagnosis
    else:
      result["diagnosis"] = None

    return result

  def add_credential_grant(self, reference: str, category: CredentialCategory) -> CredentialGrant:
    """Add a credential grant to the investigation."""
    grant = CredentialGrant(
      reference=reference,
      category=category,
      granted_at=datetime.utcnow().isoformat() + "Z",
    )
    self.credential_grants.append(grant)
    return grant

  def get_granted_credentials(self, category: Optional[CredentialCategory] = None) -> List[str]:
    """Get list of granted credential references, optionally filtered by category."""
    if category is None:
      return [g.reference for g in self.credential_grants]
    return [g.reference for g in self.credential_grants if g.category == category]

  def has_credential_grant(self, reference: str) -> bool:
    """Check if a credential has been granted."""
    return any(g.reference == reference for g in self.credential_grants)


# Global investigation state - tracks the currently active investigation (if any)
current_investigation: Optional[InvestigationState] = None
investigation_lock = asyncio.Lock()
# History of past investigations
investigation_history: list[InvestigationState] = []
MAX_INVESTIGATION_HISTORY = 10

# Pending anomaly state - tracks when anomaly is triggered but investigation hasn't started yet
pending_anomaly: Optional[dict] = None


# === monitor ===

class Monitor:
  """Watches for anomalies and spawns diagnostic investigations.

  In production, this would watch metrics from Prometheus/CloudWatch/etc.
  For demos, anomalies can be triggered manually via /incidents endpoint.

  The monitor:
  1. Runs persistently from app startup
  2. When anomaly detected, spawns a diagnostic investigation
  3. Tracks investigation state so Cursor hooks can connect
  4. Provides status for the demo flow
  """

  def __init__(self):
    self.running = False
    self.node: Optional[Node] = None
    self._monitor_task: Optional[asyncio.Task] = None
    self._investigation_task: Optional[asyncio.Task] = None

  async def start(self, node: Node):
    """Start the monitor."""
    self.node = node
    self.running = True
    logger.info("Monitor started - watching for anomalies")
    # Start the background monitoring loop (simulated for demo)
    self._monitor_task = asyncio.create_task(self._monitor_loop())

  async def stop(self):
    """Stop the monitor."""
    self.running = False
    if self._monitor_task:
      self._monitor_task.cancel()
      try:
        await self._monitor_task
      except asyncio.CancelledError:
        pass
    if self._investigation_task:
      self._investigation_task.cancel()
      try:
        await self._investigation_task
      except asyncio.CancelledError:
        pass
    logger.info("Monitor stopped")

  async def _monitor_loop(self):
    """Background loop that would check metrics in production.

    For demo purposes, this just runs and waits for manual triggers.
    In production, this would poll Prometheus, CloudWatch, etc.
    """
    while self.running:
      try:
        # In production: check metrics here
        # For demo: we use manual /trigger-anomaly endpoint
        await asyncio.sleep(30)  # Check every 30 seconds
      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error(f"Error in monitoring loop: {e}")
        await asyncio.sleep(5)

  async def trigger_anomaly(self, anomaly_type: str, severity: str, message: Optional[str] = None) -> dict:
    """Trigger anomaly detection and spawn a diagnostic investigation.

    This is called by /incidents endpoint for demo purposes.
    In production, this would be called by the monitoring loop when metrics exceed thresholds.

    Returns immediately with a pending status - the actual investigation starts after a delay.
    """
    # Generate anomaly message based on type
    if message:
      anomaly_message = message
    elif anomaly_type == "cascading_failure":
      anomaly_message = (
        "Detected cascading failure: Feature flag 'enable_new_replica_routing' was enabled 47 minutes ago. "
        "Since then, error rates have increased 340% across api-gateway, user-service, and order-service. "
        "Database connection pool exhaustion detected in us-east-1. Runaway queries accumulating."
      )
    elif anomaly_type == "high_error_rate":
      anomaly_message = f"Error rate exceeded threshold: 15% errors in the last 5 minutes (threshold: 1%)"
    elif anomaly_type == "latency_spike":
      anomaly_message = f"P99 latency spike detected: 4500ms (normal: 200ms)"
    else:
      anomaly_message = f"Anomaly detected: {anomaly_type} with severity {severity}"

    logger.info(f"Anomaly triggered: {anomaly_type} - will spawn investigation after detection delay")

    # Set pending anomaly state so UI knows something is happening
    global pending_anomaly
    pending_anomaly = {
      "anomaly_type": anomaly_type,
      "anomaly_message": anomaly_message,
      "triggered_at": datetime.utcnow().isoformat() + "Z",
    }

    # Spawn the delayed investigation asynchronously (returns immediately)
    self._investigation_task = asyncio.create_task(
      self._delayed_spawn_investigation(anomaly_type, anomaly_message)
    )

    # Return immediately with pending info (investigation not yet created)
    return {
      "status": "pending",
      "message": "Anomaly reported - monitor analyzing metrics...",
      "anomaly_type": anomaly_type,
    }

  async def _delayed_spawn_investigation(self, anomaly_type: str, anomaly_message: str):
    """Wait for detection delay, then create and spawn the investigation."""
    global current_investigation, investigation_history, pending_anomaly

    # Realistic delay to simulate the monitor analyzing metrics before confirming anomaly
    await asyncio.sleep(random.uniform(30.0, 40.0))

    # Clear pending anomaly state
    pending_anomaly = None

    async with investigation_lock:
      # Archive current investigation if exists
      if current_investigation and current_investigation.status != InvestigationStatus.IDLE:
        investigation_history.append(current_investigation)
        if len(investigation_history) > MAX_INVESTIGATION_HISTORY:
          investigation_history.pop(0)

      # Create new investigation state
      investigation_id = secrets.token_hex(8)
      current_investigation = InvestigationState(
        investigation_id=investigation_id,
        status=InvestigationStatus.DETECTING,
        created_at=datetime.utcnow().isoformat() + "Z",
        anomaly_detected_at=datetime.utcnow().isoformat() + "Z",
        anomaly_type=anomaly_type,
        anomaly_message=anomaly_message,
        regions_count=len(REGIONS),
        services_count=len(SERVICES),
        agents_count=len(REGIONS) * len(SERVICES) * len(SPECIALISTS),
      )

    logger.info(f"Investigation {investigation_id} created after detection delay - spawning diagnostic swarm")

    # Now spawn the diagnostic investigation
    await self._spawn_diagnostic_investigation(current_investigation)

  async def _spawn_diagnostic_investigation(self, investigation: InvestigationState):
    """Spawn diagnostic investigation - Phase 1: Setup and request READ credential approval.

    This method sets up the investigation and pauses for credential approval.
    The actual diagnosis runs after approval via _continue_investigation().
    """
    global current_investigation

    try:
      # Set SPAWNING status first so UI can show the spawning state
      async with investigation_lock:
        investigation.status = InvestigationStatus.SPAWNING
        investigation.investigation_started_at = datetime.utcnow().isoformat() + "Z"

      # Longer delay while in SPAWNING state (simulates swarm initialization across regions)
      await asyncio.sleep(random.uniform(15.0, 25.0))

      # Create session for the investigation
      session_id = secrets.token_hex(8)
      root_id = f"investigation-{session_id}"

      async with investigation_lock:
        investigation.session_id = session_id
        investigation.root_node_id = root_id

      # Create the problem description from anomaly
      problem = f"""Production Incident Alert

{investigation.anomaly_message}

Investigate the root cause and provide remediation recommendations.
Focus on:
1. Database health and connection pools
2. Service dependencies and cascading failures
3. Recent changes (feature flags, deployments)
4. Resource utilization patterns"""

      # Build approval prompt for READ credentials
      read_creds_display = "\n".join([f"  • {ref}" for ref in DEFAULT_READ_CREDENTIALS])
      approval_prompt = f"""🔐 **Credential Access Request**

The diagnostic investigation needs READ access to the following credentials to analyze the incident:

{read_creds_display}

These credentials provide **read-only** access for diagnostic purposes:
- Database queries to check connection pools and slow queries
- CloudWatch metrics to analyze resource utilization
- Kubernetes API to check pod health

**Approve** to grant read access and start the investigation.
**Deny** to cancel the investigation."""

      # Store in sessions - waiting for approval
      sessions[session_id] = {
        "id": session_id,
        "status": "waiting_for_approval",
        "phase": "waiting_for_read_approval",
        "problem": problem,
        "environment": "prod",
        "context": {
          "investigation_id": investigation.investigation_id,
          "anomaly_type": investigation.anomaly_type,
        },
        "created_at": investigation.created_at,
        "root_node_id": root_id,
        "distributed": True,
        "requested_credentials": DEFAULT_READ_CREDENTIALS,
        "credential_category": "read",
        "approval_prompt": approval_prompt,
      }

      # Reset graph and create root node
      await reset_graph()
      await add_node(root_id, "Incident Investigation", "root", meta={
        "problem": problem[:200] + "...",
        "environment": "prod",
        "session_id": session_id,
        "investigation_id": investigation.investigation_id,
        "anomaly_type": investigation.anomaly_type,
        "status": "waiting_for_read_approval",
      })
      await update_node_status(root_id, "pending")
      graph_state["status"] = "pending"

      await add_transcript_entry(root_id, "system",
        f"Anomaly detected: {investigation.anomaly_type}\n{investigation.anomaly_message}")
      await add_transcript_entry(root_id, "system",
        f"⏳ Waiting for READ credential approval...\n\nRequested credentials:\n{read_creds_display}")

      # Update investigation status - waiting for approval
      async with investigation_lock:
        investigation.status = InvestigationStatus.WAITING_FOR_READ_APPROVAL

      logger.info(f"Investigation {investigation.investigation_id} waiting for READ credential approval (session: {session_id})")

      # Investigation pauses here - _continue_investigation() will be called after approval

    except Exception as e:
      logger.error(f"Error spawning diagnostic investigation: {e}")
      import traceback
      logger.error(traceback.format_exc())

      async with investigation_lock:
        investigation.status = InvestigationStatus.ERROR
        investigation.error_message = str(e)
        investigation.completed_at = datetime.utcnow().isoformat() + "Z"

      if investigation.session_id and investigation.session_id in sessions:
        sessions[investigation.session_id]["status"] = "error"
        sessions[investigation.session_id]["message"] = str(e)

      graph_state["status"] = "completed"

  async def _continue_investigation(self, investigation: InvestigationState, session: dict):
    """Continue diagnostic investigation after READ credential approval - Phase 2: Run diagnosis.

    Called after user approves READ credentials. Runs the actual diagnosis with credentials.
    """
    global current_investigation

    session_id = investigation.session_id
    root_id = investigation.root_node_id

    try:
      # Add realistic delay before running phase (simulates credential validation)
      await asyncio.sleep(random.uniform(1.0, 2.5))

      async with investigation_lock:
        investigation.status = InvestigationStatus.RUNNING

      # Update session status
      session["status"] = "diagnosing"
      session["phase"] = "investigation_diagnosis"

      await update_node_status(root_id, "running")
      graph_state["status"] = "running"

      await add_transcript_entry(root_id, "system",
        f"✅ READ credentials approved. Starting parallel diagnosis across {len(REGIONS)} regions...")

      # Get the problem from session
      problem = session.get("problem", "")

      # Get credentials for distributed diagnosis
      credentials = session.get("credentials", {})

      # Record credential grants in investigation
      for ref in session.get("credentials_retrieved", []):
        category = get_credential_category(ref)
        investigation.add_credential_grant(ref, category)

      # Run parallel diagnosis across all agents WITH credentials
      results = await run_swarm_diagnosis(self.node, problem, session_id, root_id, credentials=credentials)

      # Add realistic delay before synthesis (simulates result aggregation)
      await asyncio.sleep(random.uniform(1.5, 3.0))

      async with investigation_lock:
        investigation.status = InvestigationStatus.SYNTHESIZING

      # Synthesize findings
      synthesis_node_id = f"synthesis-{session_id}"
      await add_node(synthesis_node_id, "Synthesis Agent", "synthesis", root_id, meta={
        "findings_count": len(results),
      })
      await update_node_status(synthesis_node_id, "running")

      synthesis_agent_name = f"synthesizer_{session_id}_{secrets.token_hex(4)}"
      agent = await Agent.start(
        node=self.node,
        name=synthesis_agent_name,
        instructions="""You are a senior SRE synthesizing diagnostic findings from a large-scale parallel investigation.

Multiple diagnostic agents have analyzed services across different regions. Your job is to:
1. Identify common patterns across regions/services
2. Determine the root cause of the incident
3. Prioritize remediation steps
4. Suggest monitoring improvements

Be concise but thorough. Focus on actionable insights.

IMPORTANT: After your analysis, if you identify actions that require WRITE access (such as killing queries or disabling feature flags), clearly list them as "Actions Requiring Approval" - these will need separate human approval.""",
        model=Model("claude-sonnet-4-5"),
      )

      # Build synthesis prompt
      findings_summary = []
      for result in results[:20]:
        service = result.get("service", "unknown")
        region = result.get("region", "unknown")
        service_results = result.get("results", {})

        issues = []
        for agent_type, agent_result in service_results.items():
          finding = agent_result.get("finding", "")
          if finding and ("error" in finding.lower() or "issue" in finding.lower() or "problem" in finding.lower()):
            issues.append(f"{agent_type}: {finding[:100]}")

        if issues:
          findings_summary.append(f"**{service} ({region})**: {'; '.join(issues)}")

      # Build credential usage summary
      creds_summary = ", ".join(session.get("credentials_retrieved", [])) or "none"

      synthesis_message = f"""Synthesize the diagnosis for this production incident.

**Anomaly Type**: {investigation.anomaly_type}
**Alert**: {investigation.anomaly_message}

**Credentials Used**: {creds_summary}

**Investigation Scale**:
- {len(REGIONS)} regions investigated
- {len(SERVICES)} services per region
- {len(SPECIALISTS)} diagnostic agents per service
- Total: {len(results)} services analyzed

**Key Findings**:
{chr(10).join(findings_summary) if findings_summary else "No critical issues detected."}

Provide:
1. Root cause analysis
2. Immediate remediation steps (what to do NOW with READ-only access)
3. Actions requiring WRITE approval:
   - Kill runaway database queries (requires: op://Infrastructure/prod-db-rwaccess/username and op://Infrastructure/prod-db-rwaccess/password)
   - Disable feature flag 'enable_new_replica_routing' (requires: op://Infrastructure/config-service/credential)
4. Long-term prevention measures"""

      diagnosis_text = ""
      try:
        async for response in agent.send_stream(synthesis_message, timeout=SYNTHESIS_TIMEOUT):
          snippet = response.snippet
          text = extract_text_from_snippet(snippet)
          if text:
            diagnosis_text += text
      except asyncio.TimeoutError:
        diagnosis_text += "\n\n[WARNING: Synthesis timed out, partial results shown]"

      await Agent.stop(self.node, synthesis_agent_name)

      # Store diagnosis in session
      session["diagnosis"] = diagnosis_text
      async with investigation_lock:
        investigation.diagnosis = diagnosis_text
        investigation.findings_summary = chr(10).join(findings_summary[:5]) if findings_summary else "No critical issues"

      await update_node_status(synthesis_node_id, "completed", {"diagnosis": diagnosis_text[:500]})
      await add_transcript_entry(synthesis_node_id, "model", diagnosis_text[:500] + "..." if len(diagnosis_text) > 500 else diagnosis_text)

      # Add realistic delay before checking write actions (simulates action planning)
      await asyncio.sleep(random.uniform(1.0, 2.0))

      # Check if there are write actions to approve
      write_actions = get_write_actions_for_anomaly(investigation.anomaly_type)
      if write_actions:
        # Store remaining write actions in session
        session["write_actions_remaining"] = [a.to_dict() for a in write_actions[1:]]  # All but first
        session["write_actions_completed"] = []
        session["write_actions_denied"] = []

        # Request approval for the first write action
        await self._request_write_approval(investigation, session, write_actions[0])

        logger.info(f"Investigation {investigation.investigation_id} waiting for WRITE approval (action: {write_actions[0].action_id})")
      else:
        # No write actions - complete the investigation
        await self._complete_investigation(investigation, session)

      logger.info(f"Diagnostic investigation {investigation.investigation_id} synthesis completed")

    except Exception as e:
      logger.error(f"Error in diagnostic investigation: {e}")
      import traceback
      logger.error(traceback.format_exc())

      async with investigation_lock:
        investigation.status = InvestigationStatus.ERROR
        investigation.error_message = str(e)
        investigation.completed_at = datetime.utcnow().isoformat() + "Z"

      if investigation.session_id and investigation.session_id in sessions:
        sessions[investigation.session_id]["status"] = "error"
        sessions[investigation.session_id]["message"] = str(e)

      graph_state["status"] = "completed"

  async def _request_write_approval(self, investigation: InvestigationState, session: dict, write_action: WriteAction):
    """Request approval for a write action.

    Sets up the state for write approval and pauses the investigation.
    The investigation will resume when the user approves or denies via /approve/{session_id}.
    """
    # Add realistic delay before requesting write approval (simulates action preparation)
    await asyncio.sleep(random.uniform(1.5, 3.0))

    root_id = investigation.root_node_id

    # Build approval prompt for the write action
    all_creds = write_action.get_all_credential_refs()
    creds_display = ", ".join(f"`{c}`" for c in all_creds)
    approval_prompt = f"""🔐 **WRITE Access Request**

The diagnostic investigation has identified a remediation action that requires WRITE access:

**Action**: {write_action.description}
**Credentials Required**: {creds_display}
**Action Type**: `{write_action.action_type}`

**Parameters**:
{json.dumps(write_action.params, indent=2)}

⚠️ **This is a WRITE operation that will modify production systems.**

**Approve** to execute this remediation action.
**Deny** to skip this action (manual steps will be included in the report)."""

    # Update investigation state
    async with investigation_lock:
      investigation.status = InvestigationStatus.WAITING_FOR_WRITE_APPROVAL
      investigation.pending_write_request = write_action.to_dict()

    # Update session state
    session["status"] = "waiting_for_approval"
    session["phase"] = "waiting_for_write_approval"
    session["credential_category"] = "write"
    session["requested_credentials"] = write_action.get_all_credential_refs()
    session["approval_prompt"] = approval_prompt
    session["pending_write_action"] = write_action.to_dict()

    # Update graph visualization
    await add_transcript_entry(root_id, "system",
      f"⏳ Waiting for WRITE approval: {write_action.description}")
    await update_node_status(root_id, "pending")
    graph_state["status"] = "pending"

  async def _complete_investigation(self, investigation: InvestigationState, session: dict):
    """Complete the investigation after all write actions are processed."""
    # Add realistic delay before completing (simulates final report generation)
    await asyncio.sleep(random.uniform(1.0, 2.5))

    root_id = investigation.root_node_id

    # Build final summary including denied actions
    denied_actions = session.get("write_actions_denied", [])
    completed_actions = session.get("write_actions_completed", [])

    final_notes = []
    if completed_actions:
      final_notes.append(f"✅ **Executed Remediation Actions** ({len(completed_actions)}):")
      for action in completed_actions:
        final_notes.append(f"  - {action['description']}")

    if denied_actions:
      final_notes.append(f"\n⚠️ **Manual Actions Required** ({len(denied_actions)}):")
      final_notes.append("The following actions were denied and should be performed manually:")
      for action in denied_actions:
        final_notes.append(f"  - {action['description']}")
        final_notes.append(f"    Credential: `{action['credential_ref']}`")
        if action['action_type'] == 'kill_runaway_queries':
          query_ids = action['params'].get('query_ids', [])
          final_notes.append(f"    Command: `KILL QUERY {', '.join(query_ids)}`")
        elif action['action_type'] == 'disable_feature_flag':
          flag_name = action['params'].get('flag_name', '')
          final_notes.append(f"    Command: `curl -X POST /flags/{flag_name}/disable`")

    final_summary = "\n".join(final_notes) if final_notes else ""

    # Update investigation state
    async with investigation_lock:
      investigation.status = InvestigationStatus.COMPLETED
      investigation.completed_at = datetime.utcnow().isoformat() + "Z"
      if final_summary:
        investigation.diagnosis = (investigation.diagnosis or "") + "\n\n---\n\n## Remediation Summary\n\n" + final_summary

    # Update session state
    session["status"] = "completed"
    session["phase"] = "diagnosis_complete"
    session["completed_at"] = datetime.utcnow().isoformat() + "Z"
    if final_summary:
      session["diagnosis"] = (session.get("diagnosis", "") or "") + "\n\n---\n\n## Remediation Summary\n\n" + final_summary
    session["remediation_summary"] = {
      "completed_actions": completed_actions,
      "denied_actions": denied_actions,
    }

    # Update graph visualization
    await update_node_status(root_id, "completed")
    graph_state["status"] = "completed"

    if final_summary:
      await add_transcript_entry(root_id, "system", final_summary)

    await add_transcript_entry(root_id, "system", "✅ Investigation completed")

    logger.info(f"Investigation {investigation.investigation_id} completed (executed: {len(completed_actions)}, denied: {len(denied_actions)})")

  async def _execute_write_action(self, investigation: InvestigationState, session: dict, write_action_dict: dict, credential_value: str) -> dict:
    """Execute an approved write action.

    Args:
      investigation: The current investigation state
      session: The session dict
      write_action_dict: The write action definition dict
      credential_value: The retrieved credential value

    Returns:
      The result of the write action execution
    """
    action_type = write_action_dict["action_type"]
    params = write_action_dict["params"]
    root_id = investigation.root_node_id

    await add_transcript_entry(root_id, "system",
      f"🔧 Executing: {write_action_dict['description']}")

    try:
      if action_type == "kill_runaway_queries":
        result = await kill_runaway_queries(params["query_ids"], credential_value)
      elif action_type == "disable_feature_flag":
        result = await disable_feature_flag(params["flag_name"], credential_value)
      else:
        result = json.dumps({"status": "error", "message": f"Unknown action type: {action_type}"})

      result_dict = json.loads(result)

      await add_transcript_entry(root_id, "system",
        f"✅ Action completed: {result_dict.get('message', 'Success')}")

      return result_dict

    except Exception as e:
      logger.error(f"Error executing write action {action_type}: {e}")
      await add_transcript_entry(root_id, "error",
        f"❌ Action failed: {str(e)}")
      return {"status": "error", "message": str(e)}

  async def _process_next_write_action(self, investigation: InvestigationState, session: dict):
    """Process the next write action in the queue, or complete the investigation."""
    remaining = session.get("write_actions_remaining", [])

    if remaining:
      # Get the next action
      next_action_dict = remaining.pop(0)
      session["write_actions_remaining"] = remaining

      # Convert back to WriteAction for _request_write_approval
      next_action = WriteAction(
        action_id=next_action_dict["action_id"],
        action_type=next_action_dict["action_type"],
        description=next_action_dict["description"],
        credential_ref=next_action_dict["credential_ref"],
        params=next_action_dict["params"],
        credential_refs=next_action_dict.get("credential_refs"),
      )

      await self._request_write_approval(investigation, session, next_action)
      logger.info(f"Investigation {investigation.investigation_id} waiting for WRITE approval (action: {next_action.action_id})")
    else:
      # No more actions - complete the investigation
      await self._complete_investigation(investigation, session)


# Global monitor instance
monitor = Monitor()


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
  # OnCall agent node is always added via get_graph_with_oncall_agent()


def get_graph_with_oncall_agent():
  """Get graph nodes with the OnCall Agent always included."""
  # Update oncall agent timestamp
  oncall_node = ONCALL_AGENT_NODE.copy()
  oncall_node["created_at"] = datetime.utcnow().isoformat()

  # Update oncall agent status based on new oncall_state
  if oncall_state.status == AgentStatus.INACTIVE:
    oncall_node["status"] = "inactive"
  elif oncall_state.status == AgentStatus.MONITORING:
    oncall_node["status"] = "monitoring"
  elif oncall_state.status in [AgentStatus.INCIDENT_DETECTED, AgentStatus.DIAGNOSING, AgentStatus.REMEDIATING]:
    oncall_node["status"] = "active"
  elif oncall_state.status in [AgentStatus.WAITING_FOR_ACTIVATION_APPROVAL, AgentStatus.WAITING_FOR_WRITE_APPROVAL]:
    oncall_node["status"] = "pending"
  elif oncall_state.status == AgentStatus.RESOLVED:
    oncall_node["status"] = "completed"
  else:
    oncall_node["status"] = "monitoring"

  nodes = [oncall_node] + graph_state["nodes"]
  edges = graph_state["edges"].copy()

  # Add Remediation Agent node when remediating or just resolved
  if oncall_state.status in [AgentStatus.REMEDIATING, AgentStatus.RESOLVED]:
    pending_action = oncall_state.pending_write_action
    completed_actions = oncall_state.write_actions_completed

    # Determine remediation agent status and description
    if oncall_state.status == AgentStatus.REMEDIATING:
      rem_status = "running"
      rem_description = pending_action.get("description", "Executing remediation...") if pending_action else "Executing remediation..."
    else:
      rem_status = "completed"
      if completed_actions:
        rem_description = completed_actions[-1].get("description", "Remediation completed")
      else:
        rem_description = "Remediation completed"

    remediation_node = {
      "id": "remediation-agent",
      "name": "Remediation Agent",
      "type": "remediation-agent",
      "status": rem_status,
      "parent": "oncall-agent",
      "description": rem_description,
      "created_at": datetime.utcnow().isoformat(),
    }
    nodes.append(remediation_node)

    # Connect remediation agent to oncall agent
    edges.append({"source": "oncall-agent", "target": "remediation-agent"})

  return nodes, edges

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


async def retrieve_credential_connect(reference: str, session: dict) -> tuple[bool, str]:
  """
  Retrieve a credential from 1Password using Connect API.
  Returns (success, message) tuple and stores credential in session.
  """
  # Normalize reference
  if not reference.startswith("op://"):
    reference = f"op://{reference}"

  last_error = None
  for attempt in range(CREDENTIAL_RETRY_ATTEMPTS):
    try:
      value = await resolve_connect_reference(reference)

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
      logger.warning(f"Connect credential retrieval attempt {attempt + 1} failed: {e}")

    # Wait before retry (except on last attempt)
    if attempt < CREDENTIAL_RETRY_ATTEMPTS - 1:
      await asyncio.sleep(CREDENTIAL_RETRY_DELAY * (attempt + 1))

  return False, f"Failed after {CREDENTIAL_RETRY_ATTEMPTS} attempts: {last_error}"


async def retrieve_credential(reference: str, session: dict) -> tuple[bool, str]:
  """
  Retrieve a credential from 1Password with retry logic.
  Returns (success, message) tuple and stores credential in session.

  Uses SDK mode, Connect mode, or mock mode based on ONEPASSWORD_MODE env var.
  """
  if ONEPASSWORD_MODE == "sdk":
    return await retrieve_credential_sdk(reference, session)
  elif ONEPASSWORD_MODE == "connect":
    return await retrieve_credential_connect(reference, session)
  else:
    return await retrieve_credential_mock(reference, session)


# === Mock Diagnostic Tools ===
# These tools return data reflecting the cascading failure scenario:
# - Feature flag 'enable_new_replica_routing' was enabled ~47 minutes ago
# - This caused connection pool exhaustion and runaway queries
# - Error rates increased 340% across api-gateway, user-service, order-service

async def query_db_connections(environment: str) -> str:
  """Query database connection statistics."""
  # Return mock data showing connection pool exhaustion from cascading failure
  if environment == "prod":
    return json.dumps({
      "environment": environment,
      "active_connections": 198,  # Near max - pool exhaustion!
      "max_connections": 200,
      "idle_connections": 2,  # Almost none idle
      "waiting_queries": 47,  # Many queries waiting
      "avg_query_time_ms": 4250,  # Very slow due to contention
      "connection_errors_last_hour": 156,  # Many errors
      "pool_exhaustion_events": 23,
      "status": "CRITICAL",
      "message": "Connection pool near exhaustion. Queries backing up.",
      "correlation": {
        "started_at": "~47 minutes ago",
        "correlated_event": "Feature flag 'enable_new_replica_routing' enabled",
        "affected_services": ["api-gateway", "user-service", "order-service"]
      }
    })
  else:
    return json.dumps({
      "environment": environment,
      "active_connections": 42,
      "max_connections": 100,
      "idle_connections": 15,
      "waiting_queries": 2,
      "avg_query_time_ms": 28,
      "connection_errors_last_hour": 0,
      "status": "HEALTHY"
    })


async def query_slow_queries(environment: str, threshold_ms: int = 1000) -> str:
  """Query slow database queries above threshold."""
  # Return runaway queries caused by the replica routing misconfiguration
  return json.dumps({
    "environment": environment,
    "threshold_ms": threshold_ms,
    "status": "CRITICAL",
    "message": "Multiple runaway queries detected. These started ~47 minutes ago when 'enable_new_replica_routing' was enabled.",
    "slow_queries": [
      {
        "query_id": "q-5420-orders",
        "query": "SELECT * FROM orders WHERE created_at > ? ORDER BY id DESC LIMIT 1000",
        "avg_duration_ms": 5420,
        "calls_last_hour": 156,
        "table": "orders",
        "connections_held": 3,
        "status": "RUNAWAY",
        "started_at": "47 minutes ago",
        "recommendation": "KILL - holding connections, blocking other queries"
      },
      {
        "query_id": "q-2340-inventory",
        "query": "UPDATE inventory SET quantity = quantity - ? WHERE product_id = ?",
        "avg_duration_ms": 2340,
        "calls_last_hour": 89,
        "table": "inventory",
        "connections_held": 2,
        "status": "RUNAWAY",
        "started_at": "45 minutes ago",
        "recommendation": "KILL - write query stuck, causing lock contention"
      },
      {
        "query_id": "q-1850-users-orders",
        "query": "SELECT u.*, o.* FROM users u JOIN orders o ON u.id = o.user_id WHERE u.status = ?",
        "avg_duration_ms": 1850,
        "calls_last_hour": 234,
        "table": "users, orders",
        "connections_held": 2,
        "status": "RUNAWAY",
        "started_at": "43 minutes ago",
        "recommendation": "KILL - cross-table join consuming resources"
      }
    ],
    "total_connections_held_by_slow_queries": 7,
    "correlation": "All runaway queries started after feature flag change"
  })


async def get_cloudwatch_metrics(service: str, metric: str, period_minutes: int = 60) -> str:
  """Get CloudWatch metrics for a service."""
  # Return metrics showing the spike that started 47 minutes ago
  # First 4 datapoints are "before", last 4 are "after" feature flag change

  # Metrics showing clear inflection point at feature flag change
  cascading_failure_values = {
    "CPUUtilization": [45.2, 48.1, 46.5, 47.3, 89.6, 94.4, 97.1, 98.3],  # Spike after flag
    "MemoryUtilization": [52.5, 54.3, 53.9, 55.2, 82.1, 89.8, 94.4, 96.2],  # Memory pressure
    "ErrorRate": [0.1, 0.2, 0.1, 0.15, 3.4, 4.2, 4.8, 4.9],  # 340% error increase
    "RequestLatency": [120, 135, 128, 142, 2450, 3200, 4100, 4500],  # Latency spike
    "NetworkIn": [1024000, 1056000, 980000, 1050000, 3450000, 4120000, 4890000, 5150000],
    "NetworkOut": [512000, 528000, 490000, 520000, 1720000, 2050000, 2380000, 2520000],
    "DatabaseConnections": [45, 48, 46, 47, 156, 178, 192, 198],  # Pool exhaustion
  }

  values = cascading_failure_values.get(metric, [50, 52, 51, 53, 85, 89, 92, 94])
  unit_map = {
    "CPUUtilization": "Percent",
    "MemoryUtilization": "Percent",
    "ErrorRate": "Percent",
    "RequestLatency": "Milliseconds",
    "NetworkIn": "Bytes",
    "NetworkOut": "Bytes",
    "DatabaseConnections": "Count",
  }

  return json.dumps({
    "service": service,
    "metric": metric,
    "period_minutes": period_minutes,
    "datapoints": [
      {"timestamp": f"2024-01-15T{10+i}:00:00Z", "value": v, "annotation": "BEFORE flag change" if i < 4 else "AFTER flag change"}
      for i, v in enumerate(values)
    ],
    "unit": unit_map.get(metric, "None"),
    "statistics": {
      "min": min(values),
      "max": max(values),
      "avg": sum(values) / len(values),
      "before_avg": sum(values[:4]) / 4,
      "after_avg": sum(values[4:]) / 4,
      "change_percent": round((sum(values[4:]) / 4 - sum(values[:4]) / 4) / (sum(values[:4]) / 4) * 100, 1)
    },
    "anomaly_detected": True,
    "anomaly_start": "2024-01-15T14:00:00Z",
    "correlation": "Metrics spiked when 'enable_new_replica_routing' was enabled"
  })


async def check_instance_health(instance_id: str) -> str:
  """Check EC2/RDS instance health status."""
  # Return health showing stress from cascading failure
  return json.dumps({
    "instance_id": instance_id,
    "instance_type": "m5.xlarge",
    "status": "running",
    "health_checks": {
      "system_status": "ok",
      "instance_status": "impaired"  # Degraded due to load
    },
    "metrics": {
      "cpu_utilization": 97.5,  # Critical
      "memory_utilization": 94.2,  # Critical
      "disk_io_wait": 45.3,  # High I/O wait
      "network_packets_dropped": 1247,  # Packet drops
      "connection_count": 198,  # Near limit
      "connection_limit": 200
    },
    "recent_events": [
      {"timestamp": "2024-01-15T08:00:00Z", "event": "Instance started"},
      {"timestamp": "2024-01-15T14:00:00Z", "event": "Feature flag 'enable_new_replica_routing' enabled"},
      {"timestamp": "2024-01-15T14:03:00Z", "event": "High CPU alert triggered"},
      {"timestamp": "2024-01-15T14:05:00Z", "event": "Memory pressure warning"},
      {"timestamp": "2024-01-15T14:10:00Z", "event": "Connection pool exhaustion detected"},
      {"timestamp": "2024-01-15T14:15:00Z", "event": "Instance status changed to 'impaired'"}
    ],
    "status_summary": "DEGRADED - Instance under heavy load since feature flag change",
    "recommendation": "Rollback feature flag 'enable_new_replica_routing' to reduce load"
  })


async def check_kubernetes_pods(namespace: str, label_selector: str = "") -> str:
  """Check Kubernetes pod status and health."""
  # Return pods showing cascading failures from the feature flag change
  return json.dumps({
    "namespace": namespace,
    "label_selector": label_selector or "app=api",
    "pods": [
      {
        "name": "api-gateway-7d4f8b6c9-abc12",
        "status": "Running",
        "restarts": 12,  # Many restarts since flag change
        "ready": "1/1",
        "age": "2d",
        "cpu_usage": "950m",  # Near limit
        "cpu_limit": "1000m",
        "memory_usage": "1800Mi",  # Near limit
        "memory_limit": "2Gi",
        "last_restart_reason": "OOMKilled",
        "last_restart_time": "3 minutes ago",
        "events": ["OOMKilled 12 times in last 47 minutes"]
      },
      {
        "name": "user-service-5b8e9c7d2-def34",
        "status": "Running",
        "restarts": 8,
        "ready": "1/1",
        "age": "2d",
        "cpu_usage": "890m",
        "cpu_limit": "1000m",
        "memory_usage": "1650Mi",
        "memory_limit": "2Gi",
        "last_restart_reason": "OOMKilled",
        "last_restart_time": "7 minutes ago",
        "events": ["Database connection timeout errors"]
      },
      {
        "name": "order-service-3a6f4e8b1-ghi56",
        "status": "CrashLoopBackOff",
        "restarts": 23,
        "ready": "0/1",
        "age": "47m",  # Started failing when flag was enabled
        "cpu_usage": "0m",
        "memory_usage": "0Mi",
        "last_restart_reason": "Error",
        "last_restart_time": "1 minute ago",
        "events": ["CrashLoopBackOff: Container keeps crashing", "Error: database connection pool exhausted"]
      }
    ],
    "summary": {
      "total": 3,
      "running": 2,
      "failed": 1,
      "pending": 0,
      "total_restarts_last_hour": 43,
      "status": "DEGRADED"
    },
    "correlation": {
      "failure_started": "47 minutes ago",
      "correlated_event": "Feature flag 'enable_new_replica_routing' enabled",
      "most_affected": ["order-service", "api-gateway", "user-service"]
    }
  })


async def get_application_logs(service: str, level: str = "ERROR", limit: int = 10) -> str:
  """Get recent application logs filtered by level."""
  # Return logs showing the cascading failure timeline
  return json.dumps({
    "service": service,
    "level": level,
    "limit": limit,
    "status": "CRITICAL",
    "error_rate": "340% above baseline",
    "logs": [
      {
        "timestamp": "2024-01-15T14:47:23Z",
        "level": "ERROR",
        "message": "Database connection timeout after 30000ms - pool exhausted",
        "source": "db-pool",
        "trace_id": "abc123",
        "context": {"available_connections": 0, "max_connections": 200}
      },
      {
        "timestamp": "2024-01-15T14:46:18Z",
        "level": "ERROR",
        "message": "Failed to acquire connection from pool: pool exhausted, 47 queries waiting",
        "source": "db-pool",
        "trace_id": "def456",
        "context": {"waiting_queries": 47}
      },
      {
        "timestamp": "2024-01-15T14:45:55Z",
        "level": "ERROR",
        "message": "Request timeout: /api/orders took 45023ms",
        "source": "http-handler",
        "trace_id": "ghi789",
        "context": {"endpoint": "/api/orders", "method": "GET"}
      },
      {
        "timestamp": "2024-01-15T14:44:30Z",
        "level": "ERROR",
        "message": "Memory pressure detected: heap usage at 94%",
        "source": "memory-monitor",
        "trace_id": "jkl012"
      },
      {
        "timestamp": "2024-01-15T14:00:05Z",
        "level": "WARN",
        "message": "Feature flag 'enable_new_replica_routing' changed to ENABLED",
        "source": "config-service",
        "trace_id": "flag001",
        "context": {"flag": "enable_new_replica_routing", "previous": False, "new": True}
      },
      {
        "timestamp": "2024-01-15T14:00:30Z",
        "level": "ERROR",
        "message": "New replica routing causing connection storm to read replicas",
        "source": "db-router",
        "trace_id": "route001",
        "context": {"connections_to_primary": 45, "connections_to_replica": 312}
      }
    ],
    "timeline_summary": {
      "incident_start": "2024-01-15T14:00:00Z (47 minutes ago)",
      "trigger": "Feature flag 'enable_new_replica_routing' enabled",
      "progression": [
        "14:00 - Flag enabled, routing change activated",
        "14:01 - Connection storm to read replicas begins",
        "14:03 - Error rate starts climbing",
        "14:10 - Connection pool exhaustion detected",
        "14:15 - Services begin crash-looping",
        "14:47 - Current: 340% error rate increase, pool exhausted"
      ]
    }
  })


# === Write Action Tools (require separate approval) ===

async def kill_runaway_queries(query_ids: list, write_token: str) -> str:
  """Kill runaway database queries. Requires write token.

  This is a WRITE operation that terminates long-running queries.
  Used during incidents to free up database connection pool resources.
  """
  # Mock implementation - in production this would connect to the database
  return json.dumps({
    "action": "kill_runaway_queries",
    "killed": query_ids,
    "count": len(query_ids),
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "status": "success",
    "message": f"Successfully terminated {len(query_ids)} runaway queries",
    "freed_connections": len(query_ids) * 3,  # Mock: each query was holding ~3 connections
  })


async def disable_feature_flag(flag_name: str, api_key: str) -> str:
  """Disable a feature flag. Requires config service API key.

  This is a WRITE operation that modifies the feature flag state.
  Used during incidents to rollback problematic feature releases.
  """
  # Mock implementation - in production this would call the config service API
  return json.dumps({
    "action": "disable_feature_flag",
    "flag": flag_name,
    "previous_state": "enabled",
    "new_state": "disabled",
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "status": "success",
    "message": f"Feature flag '{flag_name}' has been disabled",
    "affected_services": ["api-gateway", "user-service", "order-service"],
  })


# === Write Action Definitions ===

@dataclass
class WriteAction:
  """Definition of a write action that requires approval."""
  action_id: str
  action_type: str  # "kill_runaway_queries" or "disable_feature_flag"
  description: str
  credential_ref: str  # Primary credential ref (for backwards compatibility)
  params: dict
  credential_refs: List[str] = None  # Optional list of all credential refs needed
  tool_func: callable = None  # The function to call when approved

  def get_all_credential_refs(self) -> List[str]:
    """Get all credential refs needed for this action."""
    if self.credential_refs:
      return self.credential_refs
    return [self.credential_ref]

  def to_dict(self) -> dict:
    return {
      "action_id": self.action_id,
      "action_type": self.action_type,
      "description": self.description,
      "credential_ref": self.credential_ref,
      "credential_refs": self.get_all_credential_refs(),
      "params": self.params,
    }


def get_write_actions_for_anomaly(anomaly_type: str) -> List[WriteAction]:
  """Get the write actions recommended for a given anomaly type."""
  if anomaly_type == "cascading_failure":
    return [
      WriteAction(
        action_id="kill_queries",
        action_type="kill_runaway_queries",
        description="Kill 3 runaway database queries consuming >80% of connection pool",
        credential_ref="op://Infrastructure/prod-db-rwaccess/username",
        params={"query_ids": ["q-5420-orders", "q-2340-inventory", "q-1850-users-orders"]},
        credential_refs=["op://Infrastructure/prod-db-rwaccess/username", "op://Infrastructure/prod-db-rwaccess/password"],
        tool_func=kill_runaway_queries,
      ),
      WriteAction(
        action_id="disable_flag",
        action_type="disable_feature_flag",
        description="Disable feature flag 'enable_new_replica_routing' (enabled 47 minutes ago, correlated with incident start)",
        credential_ref="op://Infrastructure/config-service/credential",
        params={"flag_name": "enable_new_replica_routing"},
        tool_func=disable_feature_flag,
      ),
    ]
  elif anomaly_type == "high_error_rate":
    return [
      WriteAction(
        action_id="disable_flag",
        action_type="disable_feature_flag",
        description="Disable recent feature flag changes to isolate cause",
        credential_ref="op://Infrastructure/config-service/credential",
        params={"flag_name": "enable_new_api_version"},
        tool_func=disable_feature_flag,
      ),
    ]
  else:
    return []  # No write actions for other anomaly types


# === Agent Instructions ===

ANALYSIS_INSTRUCTIONS = """You are an expert SRE (Site Reliability Engineer) analyzing an infrastructure incident.

SCENARIO CONTEXT:
This is likely a cascading failure caused by a recent configuration change. Look for:
- Feature flag changes that correlate with incident start time
- Connection pool exhaustion patterns
- Runaway queries accumulating
- Error rate spikes across multiple services

Your task is to analyze the problem and identify what credentials would be needed to diagnose it.

AVAILABLE CREDENTIALS:

**READ Credentials** (for diagnostic queries - approved once, shared across investigation):
- op://Infrastructure/prod-db-readonly/password - Production database password (read-only queries)
- op://Infrastructure/prod-db-readonly/username - Production database username
- op://Infrastructure/prod-db-readonly/server - Production database server
- op://Infrastructure/aws-cloudwatch/credential - CloudWatch credential (metrics)
- op://Infrastructure/k8s-prod-readonly/credential - Kubernetes production credential (pod status)

**WRITE Credentials** (for remediation actions - require separate approval per action):
- op://Infrastructure/prod-db-rwaccess/username and /password - Database write access (kill runaway queries)
- op://Infrastructure/config-service/credential - Config service credential (disable feature flags)

Provide your analysis in this format:

## Incident Classification
[Type and severity - look for cascading failure patterns]

## Potential Root Causes
1. [Most likely] - Feature flag or configuration change
2. [Second likely] - Database connection pool issues
3. [Third likely] - Service dependency cascade

## Timeline Correlation
- When did the incident start?
- What changes occurred around that time?
- Which services were affected first?

## Required Credentials
List the specific credentials needed:
- op://... : [why needed]

Be concise and specific. Focus on finding the root cause correlation."""


DIAGNOSIS_INSTRUCTIONS = """You are an expert SRE (Site Reliability Engineer) completing a diagnosis.

You have been given access to the requested credentials (stored securely, not visible to you).
Based on the initial analysis, provide your final diagnosis and recommendations.

FOCUS ON CASCADING FAILURE PATTERNS:
1. Identify the trigger event (feature flag, deployment, config change)
2. Trace the cascade: trigger → primary failure → secondary effects
3. Prioritize remediation: rollback trigger first, then clean up secondary issues

Provide:
1. Root cause with timeline correlation
2. Immediate remediation: rollback the trigger (feature flag/config)
3. Secondary cleanup: kill stuck queries, restart affected services
4. Prevention: how to safely roll out such changes in the future

Be specific and actionable. Include the exact feature flag or config to rollback."""


# === Specialized Agent Instructions ===

DB_DIAGNOSTIC_INSTRUCTIONS = """You are a database diagnostic specialist investigating a cascading failure.

SCENARIO: A feature flag change may have caused connection pool exhaustion and runaway queries.

DIAGNOSTIC APPROACH:
1. Check connection pool status - look for exhaustion (active near max, few idle)
2. Query for slow/runaway queries - these started when the incident began
3. Look for correlation with feature flag 'enable_new_replica_routing'
4. Identify queries that should be killed to free connections

WHAT TO REPORT:
- Connection pool utilization (is it exhausted?)
- Runaway queries (query IDs, duration, connections held)
- Timeline correlation (did issues start ~47 minutes ago?)
- Recommended actions (which queries to kill)

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


CLOUD_DIAGNOSTIC_INSTRUCTIONS = """You are a cloud infrastructure diagnostic specialist investigating a cascading failure.

SCENARIO: A feature flag change caused a cascade affecting multiple services. Look for resource pressure.

DIAGNOSTIC APPROACH:
1. Check CloudWatch metrics - look for inflection points ~47 minutes ago
2. Compare before/after metrics to identify the change impact
3. Look for CPU, memory, error rate spikes correlating with feature flag
4. Check instance health for degradation

WHAT TO REPORT:
- Metric changes (before vs after the incident start)
- Correlation with feature flag 'enable_new_replica_routing' change
- Resource pressure indicators (CPU, memory near limits)
- Impact on instance health status

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


K8S_DIAGNOSTIC_INSTRUCTIONS = """You are a Kubernetes diagnostic specialist investigating a cascading failure.

SCENARIO: A feature flag change triggered pod failures across multiple services.

DIAGNOSTIC APPROACH:
1. Check pod status - look for CrashLoopBackOff, OOMKilled, restarts
2. Count restarts in the last 47 minutes (since incident start)
3. Identify which services are most affected
4. Look for correlation with feature flag change in events

WHAT TO REPORT:
- Pod health status (running, failing, crash-looping)
- Restart counts and reasons (OOMKilled indicates memory pressure)
- Most affected services (api-gateway, user-service, order-service)
- Timeline correlation with 'enable_new_replica_routing' change

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


# === FastAPI App with Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
  """Manage app lifecycle - start/stop the monitor."""
  # Startup: Get the node and start monitor
  # Note: Node is not available here, so we'll start the agent on first request
  # that provides a node. For now, just log startup.
  logger.info("OnCall service starting up...")
  logger.info("Monitor will start on first node-dependent request")

  yield

  # Shutdown: Stop the monitor
  logger.info("OnCall service shutting down...")
  await monitor.stop()


app = FastAPI(
  title="OnCall",
  description="Incident response with autonomous diagnostic agents and long-running monitoring",
  version=VERSION,
  lifespan=lifespan,
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
    # Always include the OnCall Agent node (and remediation agent if applicable)
    nodes, extra_edges = get_graph_with_oncall_agent()

    # When resolved, only return the oncall-agent node (clear investigation nodes)
    if oncall_state.status == AgentStatus.RESOLVED:
      # Filter to only keep oncall-agent node
      oncall_only = [n for n in nodes if n.get("type") == "oncall-agent"]
      return JSONResponse(content={
        "nodes": oncall_only,
        "edges": [],
        "status": "resolved",
      })

    # Get edges, adding edge from oncall-agent to investigation root if exists
    edges = list(graph_state["edges"]) + extra_edges

    # Find the root node (investigation root) to connect to oncall-agent
    root_node_id = None
    if current_investigation and current_investigation.root_node_id:
      root_node_id = current_investigation.root_node_id
    else:
      # Check for any root node in the graph (from _run_diagnosis)
      for node in graph_state["nodes"]:
        if node.get("type") == "root":
          root_node_id = node["id"]
          break

    if root_node_id:
      # Add edge from oncall-agent to the investigation root
      oncall_to_root_edge = {
        "source": "oncall-agent",
        "target": root_node_id
      }
      if oncall_to_root_edge not in edges:
        edges.insert(0, oncall_to_root_edge)

    return JSONResponse(content={
      "nodes": nodes,
      "edges": edges,
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


# === monitor Endpoints ===

# === New OnCall Agent Endpoints ===

@app.get("/oncall/status")
async def get_oncall_status():
  """Get the current status of the OnCall agent.

  This is the primary endpoint for the new activation-based flow.
  """
  async with oncall_state_lock:
    state = oncall_state

    # Build response based on current status
    response = {
      "status": state.status.value,
      "agent": state.to_dict(),
    }

    # Add approval info if waiting for approval
    if state.status == AgentStatus.WAITING_FOR_ACTIVATION_APPROVAL:
      response["waiting_for_approval"] = True
      response["approval"] = {
        "type": "activation",
        "credential_category": "read",
        "requested_credentials": MONITORING_READ_CREDENTIALS,
        "prompt": "The OnCall Agent needs READ access to monitor your infrastructure.",
      }
    elif state.status == AgentStatus.WAITING_FOR_WRITE_APPROVAL:
      response["waiting_for_approval"] = True
      response["approval"] = {
        "type": "remediation",
        "credential_category": "write",
        "pending_action": state.pending_write_action,
        "requested_credentials": state.pending_write_action.get("credential_refs", [state.pending_write_action["credential_ref"]]) if state.pending_write_action else [],
        "prompt": f"The agent wants to: {state.pending_write_action['description']}" if state.pending_write_action else "",
      }
    else:
      response["waiting_for_approval"] = False

    return JSONResponse(content=response)


@app.post("/oncall/activate")
async def activate_oncall_agent(node: NodeDep):
  """Request activation of the OnCall agent.

  This initiates the activation flow by requesting READ credentials.
  The agent will start monitoring after credentials are approved.
  """
  async with oncall_state_lock:
    if oncall_state.status != AgentStatus.INACTIVE:
      return JSONResponse(content={
        "error": "Agent is already active or activating",
        "status": oncall_state.status.value,
      }, status_code=400)

    # Transition to waiting for activation approval
    oncall_state.status = AgentStatus.WAITING_FOR_ACTIVATION_APPROVAL

  logger.info("OnCall agent activation requested - waiting for READ credential approval")

  return JSONResponse(content={
    "message": "Activation requested - approve READ credentials to start monitoring",
    "status": "waiting_for_activation_approval",
    "requested_credentials": MONITORING_READ_CREDENTIALS,
  })


@app.post("/oncall/approve")
async def approve_oncall_credentials(request: ApproveRequest, node: NodeDep):
  """Approve or deny credentials for the OnCall agent.

  Used for both activation (READ) and remediation (WRITE) approvals.
  """
  async with oncall_state_lock:
    state = oncall_state

    if state.status == AgentStatus.WAITING_FOR_ACTIVATION_APPROVAL:
      if request.approved:
        # Retrieve READ credentials
        state.status = AgentStatus.ACTIVATING

        # Retrieve all monitoring credentials
        for ref in MONITORING_READ_CREDENTIALS:
          success, result = await retrieve_credential(ref, {})
          if success:
            state.read_credentials[ref] = result
            state.read_credential_refs.append(ref)
            state.add_credential_grant(ref, CredentialCategory.READ)

        # Delay for activation
        await asyncio.sleep(random.uniform(2.0, 4.0))

        # Transition to monitoring
        state.status = AgentStatus.MONITORING
        state.activated_at = datetime.utcnow().isoformat() + "Z"
        state.monitoring_since = datetime.utcnow().isoformat() + "Z"

        # Start the monitor
        if not monitor.running:
          await monitor.start(node)

        logger.info("OnCall agent activated - now monitoring infrastructure")

        return JSONResponse(content={
          "message": "OnCall Agent activated - now monitoring infrastructure",
          "status": "monitoring",
          "credentials_granted": len(state.read_credential_refs),
        })
      else:
        # Denied - go back to inactive
        state.status = AgentStatus.INACTIVE
        logger.info("OnCall agent activation denied")
        return JSONResponse(content={
          "message": "Activation denied - agent remains inactive",
          "status": "inactive",
        })

    elif state.status == AgentStatus.WAITING_FOR_WRITE_APPROVAL:
      pending_action = state.pending_write_action
      if not pending_action:
        return JSONResponse(content={"error": "No pending write action"}, status_code=400)

      if request.approved:
        # Retrieve all WRITE credentials for this action
        credential_refs = pending_action.get("credential_refs", [pending_action["credential_ref"]])
        all_success = True
        for credential_ref in credential_refs:
          success, credential_value = await retrieve_credential(credential_ref, {})
          if success:
            state.add_credential_grant(credential_ref, CredentialCategory.WRITE)
          else:
            all_success = False
            break

        if all_success:

          # Execute the write action with visible progress
          state.status = AgentStatus.REMEDIATING
          state.remediation_progress = 0
          state.remediation_message = "Initializing remediation..."

          action_type = pending_action["action_type"]

          # Simulate gradual remediation with progress updates (30-45 seconds total)
          total_steps = 10
          step_duration = random.uniform(3.0, 4.5)  # 3-4.5 seconds per step

          progress_messages = [
            "Connecting to production database...",
            "Authenticating with write credentials...",
            "Analyzing current system state...",
            "Preparing remediation action...",
            "Executing remediation...",
            "Verifying changes...",
            "Monitoring error rates...",
            "Error rates decreasing...",
            "Validating system stability...",
            "Finalizing remediation...",
          ]

          error_rate_steps = [340, 280, 220, 160, 120, 80, 50, 25, 10, 0]

          for i in range(total_steps):
            state.remediation_progress = (i + 1) * 10
            state.remediation_message = progress_messages[i] if i < len(progress_messages) else "Processing..."
            state.error_rate_current = error_rate_steps[i] if i < len(error_rate_steps) else 0
            await asyncio.sleep(step_duration)

          # Simulate action execution
          if action_type == "kill_runaway_queries":
            result = await kill_runaway_queries(pending_action["params"]["query_ids"], credential_value)
          elif action_type == "disable_feature_flag":
            result = await disable_feature_flag(pending_action["params"]["flag_name"], credential_value)
          else:
            result = json.dumps({"status": "success", "message": "Action completed"})

          # Record completion
          completed_action = pending_action.copy()
          completed_action["result"] = json.loads(result) if isinstance(result, str) else result
          state.write_actions_completed.append(completed_action)
          state.pending_write_action = None

          logger.info(f"Write action approved and executed: {pending_action['action_id']}")

          # Check for more write actions
          if state.write_actions_remaining:
            next_action = state.write_actions_remaining.pop(0)
            state.pending_write_action = next_action
            state.status = AgentStatus.WAITING_FOR_WRITE_APPROVAL
            return JSONResponse(content={
              "message": f"Action completed. Next action pending approval.",
              "status": "waiting_for_write_approval",
              "completed_action": pending_action["action_id"],
              "next_action": next_action,
            })
          else:
            # All done - resolved
            state.status = AgentStatus.RESOLVED
            return JSONResponse(content={
              "message": "All remediation actions completed. Incident resolved.",
              "status": "resolved",
              "completed_actions": len(state.write_actions_completed),
              "denied_actions": len(state.write_actions_denied),
            })
        else:
          return JSONResponse(content={"error": f"Failed to retrieve credential: {credential_value}"}, status_code=500)

      else:
        # Denied - record and move to next action
        state.write_actions_denied.append(pending_action)
        state.pending_write_action = None

        logger.info(f"Write action denied: {pending_action['action_id']}")

        # Check for more write actions
        if state.write_actions_remaining:
          next_action = state.write_actions_remaining.pop(0)
          state.pending_write_action = next_action
          state.status = AgentStatus.WAITING_FOR_WRITE_APPROVAL
          return JSONResponse(content={
            "message": f"Action denied. Next action pending approval.",
            "status": "waiting_for_write_approval",
            "denied_action": pending_action["action_id"],
            "next_action": next_action,
          })
        else:
          # All done - resolved (with some actions denied)
          state.status = AgentStatus.RESOLVED
          return JSONResponse(content={
            "message": "Incident resolved. Some actions were denied and require manual intervention.",
            "status": "resolved",
            "completed_actions": len(state.write_actions_completed),
            "denied_actions": len(state.write_actions_denied),
          })

    else:
      return JSONResponse(content={
        "error": f"Agent is not waiting for approval (status: {state.status.value})",
      }, status_code=400)


@app.post("/oncall/simulate-anomaly")
async def simulate_anomaly(node: NodeDep):
  """Simulate an anomaly detection for demo purposes.

  Only works when agent is in MONITORING state.
  """
  async with oncall_state_lock:
    if oncall_state.status != AgentStatus.MONITORING:
      return JSONResponse(content={
        "error": f"Agent must be monitoring to detect anomalies (current: {oncall_state.status.value})",
      }, status_code=400)

    # Transition to incident detected
    oncall_state.status = AgentStatus.INCIDENT_DETECTED
    oncall_state.incident_id = secrets.token_hex(8)
    oncall_state.incident_detected_at = datetime.utcnow().isoformat() + "Z"
    oncall_state.anomaly_type = "cascading_failure"
    oncall_state.anomaly_message = (
      "Detected cascading failure: Feature flag 'enable_new_replica_routing' was enabled 47 minutes ago. "
      "Since then, error rates have increased 340% across api-gateway, user-service, and order-service. "
      "Database connection pool exhaustion detected in us-east-1. Runaway queries accumulating."
    )

  logger.info(f"Anomaly simulated: {oncall_state.incident_id}")

  # Start async diagnosis
  asyncio.create_task(_run_diagnosis(node))

  return JSONResponse(content={
    "message": "Anomaly detected - diagnostic agents deploying",
    "status": "incident_detected",
    "incident_id": oncall_state.incident_id,
  })


async def _run_diagnosis(node: Node):
  """Run the diagnosis asynchronously after anomaly detection.

  Uses credentials granted during activation to run real diagnostic agents.
  """
  global oncall_state

  # Delay for incident detection phase (longer for demo pacing)
  await asyncio.sleep(random.uniform(8.0, 12.0))

  # Get credentials from activation
  async with oncall_state_lock:
    credentials = oncall_state.read_credentials.copy()
    oncall_state.status = AgentStatus.DIAGNOSING
    oncall_state.diagnosis_started_at = datetime.utcnow().isoformat() + "Z"
    oncall_state.agents_deployed = len(REGIONS) * len(SERVICES) * len(SPECIALISTS)

  # Create session for diagnosis
  session_id = secrets.token_hex(8)
  root_id = f"diagnosis-{session_id}"

  # Reset graph and create root node for this diagnosis
  await reset_graph()
  await add_node(root_id, "Incident Investigation", "root", meta={
    "incident_id": oncall_state.incident_id,
    "anomaly_type": oncall_state.anomaly_type,
  })
  await update_node_status(root_id, "running")

  # Build problem description
  problem = f"""Production Incident Alert

{oncall_state.anomaly_message}

Investigate the root cause and provide remediation recommendations.
Focus on:
1. Database health and connection pools
2. Service dependencies and cascading failures
3. Recent changes (feature flags, deployments)
4. Resource utilization patterns"""

  await add_transcript_entry(root_id, "system", f"Starting diagnosis: {oncall_state.anomaly_type}")

  # Run actual swarm diagnosis with credentials from activation
  try:
    results = await run_swarm_diagnosis(node, problem, session_id, root_id, credentials=credentials)

    # Synthesize clean diagnosis text (don't include raw agent outputs)
    diagnosis_text = (
      "Root cause identified: Feature flag 'enable_new_replica_routing' is causing cascading failures. "
      "The new routing logic has a bug that causes connection pool exhaustion under load. "
      "Additionally, runaway queries are accumulating due to missing timeouts."
    )

  except Exception as e:
    logger.error(f"Error running swarm diagnosis: {e}")
    diagnosis_text = (
      "Root cause identified: Feature flag 'enable_new_replica_routing' is causing cascading failures. "
      "The new routing logic has a bug that causes connection pool exhaustion under load. "
      "Additionally, runaway queries are accumulating due to missing timeouts."
    )

  # Diagnosis complete - prepare write actions
  write_actions = get_write_actions_for_anomaly("cascading_failure")

  async with oncall_state_lock:
    oncall_state.status = AgentStatus.DIAGNOSIS_COMPLETE
    oncall_state.diagnosis_completed_at = datetime.utcnow().isoformat() + "Z"
    oncall_state.diagnosis_result = diagnosis_text

  await update_node_status(root_id, "completed")
  await add_transcript_entry(root_id, "system", "Diagnosis complete - root cause identified")

  # Brief delay then request write approval (longer for demo pacing)
  await asyncio.sleep(random.uniform(5.0, 8.0))

  async with oncall_state_lock:
    if write_actions:
      oncall_state.pending_write_action = write_actions[0].to_dict()
      oncall_state.write_actions_remaining = [a.to_dict() for a in write_actions[1:]]
      oncall_state.status = AgentStatus.WAITING_FOR_WRITE_APPROVAL

  logger.info("Diagnosis complete - waiting for write approval")


# _add_diagnostic_nodes_to_graph and _complete_diagnostic_nodes removed
# Graph nodes are now created by run_swarm_diagnosis which runs real agents


@app.post("/oncall/reset")
async def reset_oncall_agent():
  """Reset the OnCall agent to inactive state.

  Used for demo purposes to restart the flow.
  """
  global oncall_state

  async with oncall_state_lock:
    oncall_state = OnCallAgentState()

  # Reset graph
  await reset_graph()

  logger.info("OnCall agent reset to inactive")

  return JSONResponse(content={
    "message": "OnCall agent reset to inactive",
    "status": "inactive",
  })


# === Legacy Endpoints (kept for backwards compatibility) ===

@app.post("/incidents")
async def create_incident(request: TriggerAnomalyRequest, node: NodeDep):
  """Trigger an incident investigation.

  This endpoint is used to start a diagnostic investigation, either manually
  for demos or automatically when the Monitor detects an anomaly.

  Returns immediately with investigation info - the investigation runs asynchronously.
  Use /investigation/status to track progress.
  """
  # Ensure monitor is started with the node
  if not monitor.running:
    await monitor.start(node)

  # Trigger the investigation (returns immediately, investigation starts after delay)
  result = await monitor.trigger_anomaly(
    anomaly_type=request.anomaly_type,
    severity=request.severity,
    message=request.message,
  )

  return JSONResponse(content={
    "message": "Anomaly reported - monitor analyzing metrics",
    "status": result["status"],
    "anomaly_type": result["anomaly_type"],
  })


@app.get("/investigation/status")
async def get_investigation_status():
  """Get current investigation state.

  Returns the state of the currently active diagnostic investigation (if any).
  Cursor hooks use this to connect to running investigations.
  """
  async with investigation_lock:
    if current_investigation is None:
      if not current_investigation:
        # Check if there's a pending anomaly (detection in progress)
        if pending_anomaly:
          return JSONResponse(content={
            "active": False,
            "waiting_for_approval": False,
            "investigation": None,
            "pending_anomaly": pending_anomaly,
            "message": "monitor analyzing anomaly...",
          })
        return JSONResponse(content={
          "active": False,
          "waiting_for_approval": False,
          "investigation": None,
          "pending_anomaly": None,
          "message": "No active investigation. Use POST /incidents to start one.",
        })

    investigation = current_investigation
    is_active = investigation.status in [
      InvestigationStatus.DETECTING,
      InvestigationStatus.SPAWNING,
      InvestigationStatus.RUNNING,
      InvestigationStatus.SYNTHESIZING,
    ]
    is_waiting = investigation.status in [
      InvestigationStatus.WAITING_FOR_READ_APPROVAL,
      InvestigationStatus.WAITING_FOR_WRITE_APPROVAL,
    ]

    # Build response with approval instructions if waiting
    response = {
      "active": is_active,
      "waiting_for_approval": is_waiting,
      "investigation": investigation.to_dict(),
      "message": f"Investigation {investigation.investigation_id} is {investigation.status.value}",
    }

    if is_waiting and investigation.session_id:
      session = sessions.get(investigation.session_id, {})
      credential_category = session.get("credential_category", "read")
      response["approval"] = {
        "session_id": investigation.session_id,
        "credential_category": credential_category,
        "prompt": session.get("approval_prompt", ""),
        "requested_credentials": session.get("requested_credentials", []),
        "instructions": f"POST /approve/{investigation.session_id} with {{'approved': true}} to approve or {{'approved': false}} to deny",
      }
      # Add write-specific information if waiting for write approval
      if credential_category == "write":
        response["approval"]["pending_write_action"] = session.get("pending_write_action", {})
        response["approval"]["write_actions_remaining"] = len(session.get("write_actions_remaining", []))
        response["approval"]["write_actions_completed"] = session.get("write_actions_completed", [])
        response["approval"]["write_actions_denied"] = session.get("write_actions_denied", [])

    return JSONResponse(content=response)


@app.get("/investigation/history")
async def get_investigation_history():
  """Get history of past investigations."""
  async with investigation_lock:
    history = [s.to_dict() for s in investigation_history]
    current = current_investigation.to_dict() if current_investigation else None

  return JSONResponse(content={
    "current": current,
    "history": history,
    "total_investigations": len(history) + (1 if current else 0),
  })


@app.get("/investigation/findings")
async def get_investigation_findings():
  """Get detailed findings from the current or most recent investigation.

  This endpoint is used by Cursor hooks to provide context about
  what the diagnostic agents have discovered.
  """
  async with investigation_lock:
    if current_investigation is None:
      return JSONResponse(content={
        "has_findings": False,
        "message": "No investigation has run yet. Use POST /incidents to start one.",
      })

    investigation = current_investigation

  # Get the session data if available
  session_data = {}
  if investigation.session_id and investigation.session_id in sessions:
    session = sessions[investigation.session_id]
    session_data = {
      "diagnosis_results": session.get("diagnosis_results", []),
      "diagnosis": session.get("diagnosis", ""),
    }

  # Get graph data for agent details
  async with graph_lock:
    nodes = graph_state.get("nodes", [])
    activity = graph_state.get("activity", [])[-20:]  # Last 20 activity items

  # Count agents by status
  agent_nodes = [n for n in nodes if n.get("type") in ["diagnostic-agent", "sub-agent"]]
  status_counts = {}
  for node in agent_nodes:
    status = node.get("status", "unknown")
    status_counts[status] = status_counts.get(status, 0) + 1

  # Extract key findings from completed agents
  key_findings = []
  for node in agent_nodes:
    if node.get("status") == "completed" and node.get("meta"):
      finding = node["meta"].get("finding", "")
      if finding and ("error" in finding.lower() or "issue" in finding.lower() or "problem" in finding.lower()):
        key_findings.append({
          "agent": node.get("name", "unknown"),
          "finding": finding[:200],
        })

  return JSONResponse(content={
    "has_findings": True,
    "investigation": investigation.to_dict(),
    "session": {
      "id": investigation.session_id,
      "diagnosis": session_data.get("diagnosis", investigation.diagnosis or ""),
    },
    "agents": {
      "total": len(agent_nodes),
      "by_status": status_counts,
    },
    "key_findings": key_findings[:10],  # Top 10 findings
    "recent_activity": activity,
  })


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
  elif ONEPASSWORD_MODE == "connect":
    # Check 1Password Connect server
    try:
      response = await http_client.get(
        f"{OP_CONNECT_HOST}/health",
        headers={"Authorization": f"Bearer {OP_CONNECT_TOKEN}"}
      )
      if response.status_code == 200:
        onepass_status = "healthy (connect)"
      else:
        onepass_status = f"unhealthy (HTTP {response.status_code})"
    except httpx.RequestError as e:
      onepass_status = f"unreachable ({type(e).__name__})"
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


# === Distributed Diagnosis ===

# Agent types for distributed diagnosis - each service gets these 5 agents
# Instructions are tailored for cascading failure scenario investigation
SPECIALISTS = [
  ("database", "Check connection pool exhaustion and runaway queries. Look for correlation with feature flag 'enable_new_replica_routing' enabled ~47 minutes ago. Report query IDs that should be killed."),
  ("cache", "Check if cache pressure increased after feature flag change. Look for eviction spikes, memory pressure, and cache miss rate increases starting ~47 minutes ago."),
  ("network", "Check for connection storms or routing anomalies. The feature flag 'enable_new_replica_routing' may have caused connection spikes to read replicas. Look for packet drops and latency increases."),
  ("resources", "Check CPU and memory pressure that started ~47 minutes ago. Look for correlation with feature flag change. Report if resources are near limits and contributing to cascading failures."),
  ("logs", "Scan for errors starting ~47 minutes ago when 'enable_new_replica_routing' was enabled. Look for database timeouts, connection pool exhaustion, and the cascade progression."),
]

# Regions and services for distributed diagnosis
REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]
SERVICES = [
  "api-gateway", "user-service", "order-service", "payment-service",
  "inventory-service", "notification-service", "auth-service",
  "analytics-service", "search-service", "recommendation-service"
]


class DiagnosticWorker:
  """Worker that runs on remote runners to execute diagnostic agents for services.

  Timeout Philosophy: Allow agents to complete complex work
  ---------------------------------------------------------
  Agents are actors (Ockam workers in Rust/Tokio). With parallel execution,
  runner time ≈ max(agent times), not sum. So generous agent timeouts don't
  cause runner timeouts - the original problem was sequential execution.

  Timeouts are set to allow:
  - Capable models (claude-sonnet, opus) that may take 30-90s per call
  - Multiple autonomous iterations (thinking → acting → thinking...)
  - Complex diagnostic work, not just simple single-turn queries

  Timeout Hierarchy:
  1. LLM request timeout (300s) - Generous for capable models
  2. Throttle queue timeout (300s) - Match request timeout
  3. Agent max_execution_time (600s) - Framework default; allows 10+ iterations
  4. Runner timeout (900s) - Generous; achievable with parallel execution
  """

  async def run_single_agent(
    self,
    agent_type: str,
    instructions: str,
    service_name: str,
    region: str,
    model: "Model",
  ) -> dict:
    """Run a single diagnostic agent. Designed for parallel execution.

    Args:
      agent_type: Type of agent (database, cache, network, etc.)
      instructions: Agent-specific diagnostic instructions
      service_name: Name of service being diagnosed
      region: Region where service runs
      model: Shared Model instance with throttling

    Returns:
      Dict with agent_type, status, and finding
    """
    from autonomy import Agent
    import secrets as secrets_module
    import asyncio

    agent_name = f"diag_{agent_type}_{service_name}_{secrets_module.token_hex(4)}"

    try:
      # Start agent with generous timeouts to allow completion of complex work
      agent = await Agent.start(
        node=self.node,
        instructions=f"""You are a {agent_type} diagnostic agent for the {service_name} service in {region}.
{instructions}

Provide a brief assessment (2-3 sentences) of the {agent_type} health for this service.
Report any issues found or confirm healthy status.""",
        model=model,
        # Use framework defaults for max_iterations (1000) and max_execution_time (600s)
        # This allows agents to do complex multi-step work with capable models
      )

      try:
        message = f"Diagnose {agent_type} health for {service_name} in {region}. Check for any issues related to the production latency spike."
        # Explicit timeout must exceed agent's max_execution_time (600s default)
        # The send() timeout controls Ockam message transport layer (default 30s)
        # Agent's max_execution_time controls internal state machine execution
        # Set to 660s (11 minutes) to allow agent to complete + buffer time
        responses = await agent.send(message, timeout=660)
        finding = responses[-1].content.text if responses else "No response"
        return {
          "agent_type": agent_type,
          "status": "completed",
          "finding": finding[:500]
        }
      except asyncio.TimeoutError:
        return {
          "agent_type": agent_type,
          "status": "timeout",
          "finding": "Agent timed out"
        }
      except Exception as agent_err:
        return {
          "agent_type": agent_type,
          "status": "error",
          "finding": str(agent_err)[:200]
        }
      finally:
        try:
          await Agent.stop(self.node, agent_name)
        except Exception:
          pass

    except asyncio.TimeoutError:
      return {
        "agent_type": agent_type,
        "status": "timeout",
        "finding": "Agent failed to start"
      }
    except Exception as start_err:
      return {
        "agent_type": agent_type,
        "status": "error",
        "finding": f"Failed to start agent: {str(start_err)[:100]}"
      }

  async def run_service_diagnosis(self, service_info: dict, model: "Model") -> dict:
    """Run multiple diagnostic agents for a single service IN PARALLEL.

    Args:
      service_info: Dict with service, region, node_id, session_id
      model: Shared Model instance (single throttle queue for all agents)

    Key insight: Agents are actors (Ockam workers in Rust/Tokio runtime), not Python threads.
    The Model's throttle queue handles LLM request backpressure, so we don't need a semaphore.
    All agents can fire requests simultaneously - the throttle queue manages concurrency.
    """
    import asyncio

    # Agent types for distributed diagnosis
    agent_configs = [
      ("database", "Analyze database connections, query latency, connection pool health, and slow queries."),
      ("cache", "Analyze cache hit rates, eviction patterns, memory usage, and cache miss patterns."),
      ("network", "Check network latency, DNS resolution, connection errors, and packet loss."),
      ("resources", "Check CPU usage, memory consumption, disk I/O, and container resource limits."),
      ("logs", "Scan application logs for errors, warning patterns, anomalies, and stack traces."),
    ]

    service_name = service_info["service"]
    region = service_info["region"]
    node_id = service_info["node_id"]

    # Run ALL agents for this service in PARALLEL
    # Agents are actors - they don't block Python's event loop
    # The throttle queue (throttle_max_requests_in_progress) handles LLM concurrency
    agent_tasks = [
      self.run_single_agent(agent_type, instructions, service_name, region, model)
      for agent_type, instructions in agent_configs
    ]
    agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

    # Convert results to expected format
    results = {}
    for i, result in enumerate(agent_results):
      agent_type = agent_configs[i][0]
      if isinstance(result, Exception):
        results[agent_type] = {
          "status": "error",
          "finding": f"Unexpected error: {str(result)[:150]}"
        }
      else:
        results[result["agent_type"]] = {
          "status": result["status"],
          "finding": result["finding"]
        }

    return {
      "service": service_name,
      "region": region,
      "node_id": node_id,
      "results": results,
    }

  async def handle_message(self, context, message):
    """Handle batch of services to diagnose.

    Agents are actors (Ockam workers in Rust/Tokio runtime), so we don't need
    a semaphore to limit concurrency. All 30 agents fire in parallel, and the
    Model's throttle queue coordinates LLM requests to the gateway.

    Timeout philosophy: Allow agents to complete complex work
    - Generous timeouts accommodate capable models (sonnet, opus)
    - Agents may take multiple autonomous steps
    - With parallel execution, runner time ≈ max(agent times)
    - 900s runner timeout is achievable because agents run in parallel
    """
    import json as json_module
    import asyncio
    from autonomy import Model

    request = json_module.loads(message)
    services = request.get("services", [])
    runner_id = request.get("runner_id", "unknown")
    credentials = request.get("credentials", {})

    if credentials:
      logger.info(f"Runner {runner_id} received {len(credentials)} credentials")

    # Create ONE shared Model instance for ALL agents in this batch.
    # No semaphore needed - agents are actors, throttle queue handles backpressure.
    model = Model(
      "claude-sonnet-4-5",
      throttle=True,
      # All 30 agents fire concurrently - no client-side queuing
      throttle_max_requests_in_progress=30,
      # Generous timeouts to allow capable models to complete complex work
      throttle_max_seconds_to_wait_in_queue=300.0,
      request_timeout=300.0,
      # Retries for transient failures
      throttle_max_retry_attempts=3,
      throttle_initial_seconds_between_retry_attempts=2.0,
      throttle_max_seconds_between_retry_attempts=30.0,
    )

    # No semaphore needed - agents are actors, throttle queue handles backpressure
    # All services process in parallel, all agents within services process in parallel
    tasks = [self.run_service_diagnosis(service_info, model) for service_info in services]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and convert to proper results
    valid_results = []
    for i, result in enumerate(results):
      if isinstance(result, Exception):
        # Create error result for failed services
        service_info = services[i]
        valid_results.append({
          "service": service_info.get("service", "unknown"),
          "region": service_info.get("region", "unknown"),
          "node_id": service_info.get("node_id", "unknown"),
          "results": {
            "database": {"status": "error", "finding": str(result)[:200]},
            "cache": {"status": "error", "finding": str(result)[:200]},
            "network": {"status": "error", "finding": str(result)[:200]},
            "resources": {"status": "error", "finding": str(result)[:200]},
            "logs": {"status": "error", "finding": str(result)[:200]},
          },
        })
      else:
        valid_results.append(result)

    # Return all results at once
    await context.reply(json_module.dumps({
      "runner_id": runner_id,
      "results": valid_results,
    }))


async def run_swarm_diagnosis(node, problem: str, session_id: str, root_id: str, credentials: dict = None):
  """Run parallel diagnosis across all regions and services.

  Spawns 150+ diagnostic agents (3 regions × 10 services × 5 agent types) all running
  in parallel on the main pod. Agents are actors, so they don't block -
  the runtime schedules them efficiently.

  Args:
    node: The Autonomy node
    problem: The problem description
    session_id: Session identifier
    root_id: Root node ID for graph visualization
    credentials: Optional credentials dict for diagnostic tools
  """
  logger.info(f"Starting parallel diagnosis for session {session_id}")

  # Create region and service nodes in visualization
  targets = []
  for region in REGIONS:
    region_id = f"region-{region}-{session_id}"
    await add_node(region_id, region, "region", root_id)
    await update_node_status(region_id, "running")

    for service in SERVICES:
      service_id = f"service-{region}-{service}-{session_id}"
      await add_node(service_id, service, "service", region_id)
      targets.append({
        "service": service,
        "region": region,
        "node_id": service_id,
        "session_id": session_id,
      })

  total_agents = len(targets) * len(SPECIALISTS)
  await add_transcript_entry(root_id, "system",
    f"Created {len(targets)} investigation targets across {len(REGIONS)} regions ({total_agents} agents)")

  # Create all agent nodes upfront for visualization
  for target in targets:
    service_node_id = target["node_id"]
    await update_node_status(service_node_id, "running")
    for agent_type, _ in SPECIALISTS:
      agent_node_id = f"agent-{target['region']}-{target['service']}-{agent_type}-{session_id}"
      await add_node(agent_node_id, agent_type, "diagnostic-agent", service_node_id)
      await update_node_status(agent_node_id, "running")

  # Create shared model for all agents (throttle queue handles concurrency)
  from autonomy import Model
  model = Model(
    "claude-sonnet-4-5",
    throttle=True,
    throttle_max_requests_in_progress=30,
    throttle_max_seconds_to_wait_in_queue=300.0,
    request_timeout=300.0,
    throttle_max_retry_attempts=3,
    throttle_initial_seconds_between_retry_attempts=2.0,
    throttle_max_seconds_between_retry_attempts=30.0,
  )

  # Create worker and run all targets in parallel
  worker = DiagnosticWorker()
  worker.node = node

  async def run_target(target):
    """Run diagnosis for a single service target."""
    try:
      result = await worker.run_service_diagnosis(target, model)

      # Update visualization
      service_node_id = target["node_id"]
      await update_node_status(service_node_id, "completed")

      for agent_type, agent_result in result.get("results", {}).items():
        agent_node_id = f"agent-{target['region']}-{target['service']}-{agent_type}-{session_id}"
        agent_status = "completed" if agent_result.get("status") == "completed" else "error"
        await update_node_status(agent_node_id, agent_status, agent_result)

      return result
    except Exception as e:
      logger.error(f"Error diagnosing {target['service']} in {target['region']}: {e}")
      service_node_id = target["node_id"]
      await update_node_status(service_node_id, "error")
      for agent_type, _ in SPECIALISTS:
        agent_node_id = f"agent-{target['region']}-{target['service']}-{agent_type}-{session_id}"
        await update_node_status(agent_node_id, "error", {"status": "error", "finding": str(e)[:200]})
      return {
        "service": target["service"],
        "region": target["region"],
        "node_id": target["node_id"],
        "results": {at: {"status": "error", "finding": str(e)[:200]} for at, _ in SPECIALISTS},
      }

  # Run ALL targets in parallel - agents are actors, they don't block
  results = await asyncio.gather(*[run_target(t) for t in targets], return_exceptions=True)

  # Filter out exceptions
  valid_results = []
  for i, result in enumerate(results):
    if isinstance(result, Exception):
      logger.error(f"Target {i} failed: {result}")
      target = targets[i]
      valid_results.append({
        "service": target["service"],
        "region": target["region"],
        "node_id": target["node_id"],
        "results": {at: {"status": "error", "finding": str(result)[:200]} for at, _ in SPECIALISTS},
      })
    else:
      valid_results.append(result)

  # Update region nodes to completed
  for region in REGIONS:
    region_id = f"region-{region}-{session_id}"
    await update_node_status(region_id, "completed")

  await add_transcript_entry(root_id, "system",
    f"Parallel diagnosis complete: {len(valid_results)} services analyzed")

  return valid_results


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
  """Approve or deny credential access.

  Handles two types of approvals:
  1. Investigation READ approval - grants read-only credentials and continues investigation
  2. Regular diagnose session approval - original flow for /diagnose endpoint
  """
  global current_investigation

  if session_id not in sessions:
    return JSONResponse(content={"error": "Session not found"}, status_code=404)

  session = sessions[session_id]

  if session.get("status") != "waiting_for_approval":
    return JSONResponse(
      content={"error": f"Session not waiting for approval (status: {session.get('status')})"},
      status_code=400
    )

  # Check if this is an investigation approval
  is_investigation = session.get("context", {}).get("investigation_id") is not None
  credential_category = session.get("credential_category", "read")

  if not request.approved:
    # Handle WRITE denial differently - don't cancel, just skip this action and continue
    if is_investigation and credential_category == "write" and current_investigation:
      return await _handle_investigation_write_denial(session_id, session)

    # READ denial cancels the investigation
    session["status"] = "denied"
    session["phase"] = "credentials_denied"

    # Update graph with denial
    root_id = session.get("root_node_id")
    if root_id:
      await update_node_status(root_id, "error")
      category_label = credential_category.upper()
      await add_transcript_entry(root_id, "system", f"❌ {category_label} credential access denied by user")
      graph_state["status"] = "completed"

    # Update investigation state if this is an investigation
    if is_investigation and current_investigation:
      async with investigation_lock:
        current_investigation.status = InvestigationStatus.ERROR
        current_investigation.error_message = f"{credential_category.upper()} credentials denied"
        current_investigation.completed_at = datetime.utcnow().isoformat() + "Z"

    return JSONResponse(content={
      "session_id": session_id,
      "status": "denied",
      "credential_category": credential_category,
      "message": f"{credential_category.upper()} credential access denied, investigation cancelled"
    })

  # Approved - retrieve credentials
  session["status"] = "retrieving_credentials"
  session["phase"] = "credential_retrieval"

  # For investigation approvals, we handle things differently
  if is_investigation and credential_category == "read":
    return await _handle_investigation_read_approval(session_id, session, node)

  # Handle WRITE approval for investigations
  if is_investigation and credential_category == "write":
    return await _handle_investigation_write_approval(session_id, session, node)

  # Original flow for regular diagnose sessions
  async def stream_diagnosis():
    synthesis_agent_name = None
    diagnosis_start_time = datetime.utcnow()

    # Get graph node IDs from session
    root_id = session.get("root_node_id", f"investigation-{session_id}")

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

      # Get credentials for distributed diagnosis
      credentials = session.get("credentials", {})
      creds_summary = ", ".join(session.get("credentials_retrieved", [])) or "none"

      yield make_event("diagnosis_started",
        session_id=session_id,
        credentials_retrieved=successful_creds,
        credentials_failed=len(failed_creds),
        regions=REGIONS,
        services=SERVICES,
        agents_per_service=len(SPECIALISTS),
        total_targets=len(REGIONS) * len(SERVICES),
        total_agents=len(REGIONS) * len(SERVICES) * len(SPECIALISTS),
        progress=20
      )

      await add_transcript_entry(root_id, "system",
        f"Starting distributed diagnosis: {len(REGIONS)} regions × {len(SERVICES)} services × {len(SPECIALISTS)} agents = {len(REGIONS) * len(SERVICES) * len(SPECIALISTS)} total agents")

      # Run distributed diagnosis across all regions/services WITH credentials
      results = await run_distributed_diagnosis(
        node,
        session.get("problem", ""),
        session_id,
        root_id,
        credentials=credentials
      )

      # Store results in session
      session["diagnosis_results"] = results

      yield make_event("collection_complete",
        session_id=session_id,
        results_count=len(results),
        progress=70
      )

      await add_transcript_entry(root_id, "system", f"Collected results from {len(results)} services")

      # Now run synthesizer agent to combine findings
      yield make_event("synthesis_started",
        session_id=session_id,
        findings_count=len(results),
        progress=75
      )

      # Create synthesis agent node in graph
      synthesis_node_id = f"synthesis-{session_id}"
      await add_node(synthesis_node_id, "Synthesis Agent", "synthesis", root_id, meta={
        "findings_count": len(results),
      })
      await update_node_status(synthesis_node_id, "running")
      await add_transcript_entry(synthesis_node_id, "system", f"Synthesizing findings from {len(results)} services...")

      # Create synthesis agent to combine all findings
      synthesis_agent_name = f"synthesizer_{session_id}_{secrets.token_hex(4)}"
      agent = await Agent.start(
        node=node,
        name=synthesis_agent_name,
        instructions="""You are a senior SRE synthesizing diagnostic findings from a large-scale parallel investigation.

Multiple diagnostic agents have analyzed services across different regions. Your job is to:
1. Identify common patterns across regions/services
2. Determine the root cause of the incident
3. Prioritize remediation steps
4. Suggest monitoring improvements

Be concise but thorough. Focus on actionable insights.""",
        model=Model("claude-sonnet-4-5"),
      )

      # Build synthesis prompt with aggregated findings
      findings_summary = []
      for result in results[:20]:  # Limit to first 20 for synthesis
        service = result.get("service", "unknown")
        region = result.get("region", "unknown")
        service_results = result.get("results", {})

        issues = []
        for agent_type, agent_result in service_results.items():
          if agent_result.get("status") != "completed":
            issues.append(f"{agent_type}: {agent_result.get('status', 'unknown')}")
          finding = agent_result.get("finding", "")
          if finding and ("error" in finding.lower() or "issue" in finding.lower() or "problem" in finding.lower()):
            issues.append(f"{agent_type}: {finding[:100]}")

        if issues:
          findings_summary.append(f"**{service} ({region})**: {'; '.join(issues)}")

      diagnosis_message = f"""Synthesize the diagnosis for this production incident.

**Problem**: {session.get('problem')}
**Environment**: {session.get('environment')}

**Initial Analysis**:
{session.get('analysis', 'No analysis available')}

**Credentials Retrieved**: {creds_summary}

**Investigation Scale**:
- {len(REGIONS)} regions investigated
- {len(SERVICES)} services per region
- {len(SPECIALISTS)} diagnostic agents per service
- Total: {len(results)} services analyzed with {len(results) * len(SPECIALISTS)} agent findings

**Key Findings Summary**:
{chr(10).join(findings_summary) if findings_summary else "No critical issues detected across services."}

Based on this large-scale investigation, provide:
1. Root cause analysis
2. Immediate remediation steps
3. Long-term prevention measures
4. Recommended monitoring improvements"""

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

      total_duration = (datetime.utcnow() - diagnosis_start_time).total_seconds()
      await add_transcript_entry(root_id, "system", f"Diagnosis complete in {round(total_duration, 1)}s with {len(results) * len(SPECIALISTS)} agents")
      graph_state["status"] = "completed"

      yield make_event("diagnosis_complete",
        session_id=session_id,
        status="completed",
        credentials_retrieved=session.get("credentials_retrieved", []),
        services_analyzed=len(results),
        total_agents=len(results) * len(SPECIALISTS),
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


async def _handle_investigation_read_approval(session_id: str, session: dict, node: Node):
  """Handle READ credential approval for Monitor investigations.

  Retrieves credentials, updates investigation state, and continues the investigation.
  Returns a streaming response with progress events.
  """
  global current_investigation

  async def stream_investigation_approval():
    root_id = session.get("root_node_id", f"investigation-{session_id}")

    try:
      yield make_event("approval_accepted",
        session_id=session_id,
        status="approved",
        credential_category="read",
        progress=0
      )

      await add_transcript_entry(root_id, "system", "✅ READ credentials approved")

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
          category="read",
          progress=int((i + 1) / total_creds * 20) if total_creds > 0 else 20
        )
        logger.info(f"Credential retrieval: {result}")

      # Log warning if some credentials failed
      if failed_creds:
        logger.warning(f"Failed to retrieve {len(failed_creds)} credentials: {failed_creds}")

      await add_transcript_entry(root_id, "system",
        f"Retrieved {successful_creds}/{total_creds} READ credentials")

      yield make_event("credentials_complete",
        session_id=session_id,
        credentials_retrieved=successful_creds,
        credentials_failed=len(failed_creds),
        category="read",
        progress=20
      )

      # Now continue the investigation
      if current_investigation and current_investigation.session_id == session_id:
        yield make_event("investigation_continuing",
          session_id=session_id,
          investigation_id=current_investigation.investigation_id,
          message="Starting parallel diagnosis...",
          progress=25
        )

        # Continue the investigation in the background
        # The investigation will run and update graph/session state
        asyncio.create_task(
          monitor._continue_investigation(current_investigation, session)
        )

        yield make_event("investigation_started",
          session_id=session_id,
          investigation_id=current_investigation.investigation_id,
          regions=REGIONS,
          services=SERVICES,
          agents_per_service=len(SPECIALISTS),
          total_agents=len(REGIONS) * len(SERVICES) * len(SPECIALISTS),
          message="Diagnostic investigation is now running. Check /investigations/current for status.",
          progress=30
        )
      else:
        yield make_event("warning",
          session_id=session_id,
          message="No active investigation found to continue",
          progress=100
        )

    except Exception as e:
      logger.error(f"Error in investigation approval: {str(e)}")
      import traceback
      logger.error(f"Traceback: {traceback.format_exc()}")

      session["status"] = "error"
      session["phase"] = "approval_failed"
      session["message"] = str(e)

      await update_node_status(root_id, "error")
      await add_transcript_entry(root_id, "error", f"Credential retrieval failed: {str(e)}")

      yield make_event("error",
        session_id=session_id,
        message=str(e)
      )

  return StreamingResponse(stream_investigation_approval(), media_type="application/x-ndjson")


async def _handle_investigation_write_denial(session_id: str, session: dict):
  """Handle WRITE credential denial for Monitor investigations.

  When a write action is denied:
  1. Add the action to denied_actions list (will be included as manual steps in report)
  2. Move to the next write action, or complete if none remaining

  Returns a JSON response (not streaming).
  """
  global current_investigation

  # Add realistic delay for processing denial (simulates logging and state update)
  await asyncio.sleep(random.uniform(1.0, 2.0))

  root_id = session.get("root_node_id", f"investigation-{session_id}")
  pending_action = session.get("pending_write_action", {})

  try:
    # Clear pending write request
    if current_investigation and current_investigation.session_id == session_id:
      async with investigation_lock:
        current_investigation.pending_write_request = None

    # Add to denied actions list
    denied_actions = session.get("write_actions_denied", [])
    denied_actions.append(pending_action)
    session["write_actions_denied"] = denied_actions

    # Log the denial
    await add_transcript_entry(root_id, "system",
      f"❌ WRITE access denied: {pending_action.get('description', 'Unknown action')}\n"
      f"   Manual steps will be included in the final report.")

    logger.info(f"Write action denied: {pending_action.get('action_id', 'unknown')}")

    # Process next write action or complete
    if current_investigation and current_investigation.session_id == session_id:
      await monitor._process_next_write_action(current_investigation, session)

    return JSONResponse(content={
      "session_id": session_id,
      "status": "write_denied",
      "credential_category": "write",
      "action_denied": pending_action.get("action_id"),
      "message": "Write action denied. Manual steps will be included in report. Processing next action...",
      "remaining_actions": len(session.get("write_actions_remaining", [])),
    })

  except Exception as e:
    logger.error(f"Error handling write denial: {str(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")

    return JSONResponse(content={
      "session_id": session_id,
      "status": "error",
      "message": str(e)
    }, status_code=500)


async def _handle_investigation_write_approval(session_id: str, session: dict, node: Node):
  """Handle WRITE credential approval for Monitor investigations.

  When a write action is approved:
  1. Retrieve the credential
  2. Execute the write action
  3. Move to the next write action, or complete if none remaining

  Returns a streaming response with progress events.
  """
  global current_investigation

  async def stream_write_approval():
    root_id = session.get("root_node_id", f"investigation-{session_id}")
    pending_action = session.get("pending_write_action", {})

    try:
      # Add realistic delay for processing approval (simulates credential validation)
      await asyncio.sleep(random.uniform(1.0, 2.5))

      yield make_event("approval_accepted",
        session_id=session_id,
        status="approved",
        credential_category="write",
        action_id=pending_action.get("action_id"),
        progress=0
      )

      await add_transcript_entry(root_id, "system",
        f"✅ WRITE access approved: {pending_action.get('description', 'Unknown action')}")

      # Retrieve all write credentials for this action
      requested = session.get("requested_credentials", [])
      if not requested:
        requested = pending_action.get("credential_refs", [pending_action.get("credential_ref")])

      all_credentials_retrieved = True
      retrieved_values = {}

      for i, credential_ref in enumerate(requested):
        progress_base = 20 + (i * 15 // len(requested))

        yield make_event("credential_retrieving",
          reference=credential_ref,
          category="write",
          progress=progress_base
        )

        success, result = await retrieve_credential(credential_ref, session)

        if not success:
          yield make_event("credential_failed",
            reference=credential_ref,
            message=result,
            progress=progress_base + 5
          )
          await add_transcript_entry(root_id, "error",
            f"❌ Failed to retrieve credential: {result}")

          all_credentials_retrieved = False
          break

        retrieved_values[credential_ref] = session.get("credentials", {}).get(credential_ref, "mock-credential-value")

        yield make_event("credential_retrieved",
          reference=credential_ref,
          success=True,
          category="write",
          progress=progress_base + 10
        )

        # Record credential grant
        if current_investigation and current_investigation.session_id == session_id:
          async with investigation_lock:
            current_investigation.add_credential_grant(credential_ref, CredentialCategory.WRITE)

      if not all_credentials_retrieved:
        # Treat as denial and move on
        denied_actions = session.get("write_actions_denied", [])
        denied_actions.append(pending_action)
        session["write_actions_denied"] = denied_actions

        if current_investigation and current_investigation.session_id == session_id:
          await monitor._process_next_write_action(current_investigation, session)

        yield make_event("action_skipped",
          action_id=pending_action.get("action_id"),
          reason="credential_retrieval_failed",
          progress=100
        )
        return

      # Get the first credential value from retrieved values (for backwards compatibility)
      credential_value = list(retrieved_values.values())[0] if retrieved_values else "mock-credential-value"

      yield make_event("action_executing",
        action_id=pending_action.get("action_id"),
        action_type=pending_action.get("action_type"),
        progress=60
      )

      # Add realistic delay before executing action (simulates preparation)
      await asyncio.sleep(random.uniform(1.5, 3.0))

      # Execute the write action
      if current_investigation and current_investigation.session_id == session_id:
        action_result = await monitor._execute_write_action(
          current_investigation,
          session,
          pending_action,
          credential_value
        )

        yield make_event("action_completed",
          action_id=pending_action.get("action_id"),
          result=action_result,
          progress=80
        )

        # Add to completed actions
        completed_actions = session.get("write_actions_completed", [])
        completed_action = pending_action.copy()
        completed_action["result"] = action_result
        completed_actions.append(completed_action)
        session["write_actions_completed"] = completed_actions

        # Clear pending write request
        async with investigation_lock:
          current_investigation.pending_write_request = None

        # Process next write action or complete
        await monitor._process_next_write_action(current_investigation, session)

        yield make_event("write_approval_complete",
          session_id=session_id,
          action_id=pending_action.get("action_id"),
          status="executed",
          remaining_actions=len(session.get("write_actions_remaining", [])),
          progress=100
        )
      else:
        yield make_event("warning",
          session_id=session_id,
          message="No active investigation found",
          progress=100
        )

    except Exception as e:
      logger.error(f"Error in write approval: {str(e)}")
      import traceback
      logger.error(f"Traceback: {traceback.format_exc()}")

      session["status"] = "error"
      session["phase"] = "write_approval_failed"
      session["message"] = str(e)

      await update_node_status(root_id, "error")
      await add_transcript_entry(root_id, "error", f"Write action failed: {str(e)}")

      yield make_event("error",
        session_id=session_id,
        message=str(e)
      )

  return StreamingResponse(stream_write_approval(), media_type="application/x-ndjson")


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
    credentials_retrieved=session.get("credentials_retrieved", []),
    requested_credentials=session.get("requested_credentials", [])
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


# === Distributed Diagnosis Endpoints ===

@app.post("/diagnose/swarm")
async def diagnose_swarm_endpoint(request: DiagnoseRequest, node: NodeDep):
  """Start a swarm diagnosis (150+ agents) WITHOUT credentials.

  This endpoint runs swarm diagnosis directly without the credential approval flow.
  For the full flow with analysis, approval, and credentials, use POST /diagnose.

  Scale:
  - 3 regions × 10 services = 30 investigation targets
  - 5 diagnostic agents per service = 150+ agents
  - All agents run in parallel on the main pod
  """
  session_id = secrets.token_hex(8)
  root_id = f"investigation-{session_id}"

  sessions[session_id] = {
    "id": session_id,
    "status": "diagnosing",
    "phase": "distributed_diagnosis",
    "problem": request.problem,
    "environment": request.environment,
    "context": request.context or {},
    "created_at": datetime.utcnow().isoformat(),
    "root_node_id": root_id,
    "distributed": True,
  }

  # Reset graph and create root node
  await reset_graph()
  await add_node(root_id, "Incident Investigation", "root", meta={
    "problem": request.problem,
    "environment": request.environment,
    "session_id": session_id,
    "distributed": True,
  })
  await update_node_status(root_id, "running")
  graph_state["status"] = "running"

  async def stream_swarm_diagnosis():
    synthesis_agent_name = None
    diagnosis_start_time = datetime.utcnow()

    try:
      yield make_event("diagnosis_started",
        session_id=session_id,
        status="diagnosing",
        regions=REGIONS,
        services=SERVICES,
        agents_per_service=len(SPECIALISTS),
        total_targets=len(REGIONS) * len(SERVICES),
        total_agents=len(REGIONS) * len(SERVICES) * len(SPECIALISTS),
        progress=0
      )

      await add_transcript_entry(root_id, "system",
        f"Starting swarm diagnosis: {len(REGIONS)} regions × {len(SERVICES)} services × {len(SPECIALISTS)} agents = {len(REGIONS) * len(SERVICES) * len(SPECIALISTS)} total agents")

      # Run swarm diagnosis locally
      results = await run_swarm_diagnosis(node, request.problem, session_id, root_id)

      yield make_event("collection_complete",
        session_id=session_id,
        results_count=len(results),
        progress=70
      )

      await add_transcript_entry(root_id, "system", f"Collected results from {len(results)} services")

      # Create synthesis agent to combine all findings
      yield make_event("synthesis_started",
        session_id=session_id,
        findings_count=len(results),
        progress=75
      )

      synthesis_node_id = f"synthesis-{session_id}"
      await add_node(synthesis_node_id, "Synthesis Agent", "synthesis", root_id, meta={
        "findings_count": len(results),
      })
      await update_node_status(synthesis_node_id, "running")
      await add_transcript_entry(synthesis_node_id, "system", f"Synthesizing findings from {len(results)} services...")

      synthesis_agent_name = f"synthesizer_{session_id}_{secrets.token_hex(4)}"
      agent = await Agent.start(
        node=node,
        name=synthesis_agent_name,
        instructions="""You are a senior SRE synthesizing diagnostic findings from a large-scale parallel investigation.

Multiple diagnostic agents have analyzed services across different regions. Your job is to:
1. Identify common patterns across regions/services
2. Determine the root cause of the incident
3. Prioritize remediation steps
4. Suggest monitoring improvements

Be concise but thorough. Focus on actionable insights.""",
        model=Model("claude-sonnet-4-5"),
      )

      # Build synthesis prompt with aggregated findings
      findings_summary = []
      for result in results[:20]:  # Limit to first 20 for synthesis
        service = result.get("service", "unknown")
        region = result.get("region", "unknown")
        service_results = result.get("results", {})

        issues = []
        for agent_type, agent_result in service_results.items():
          if agent_result.get("status") != "completed":
            issues.append(f"{agent_type}: {agent_result.get('status', 'unknown')}")
          finding = agent_result.get("finding", "")
          if finding and "error" in finding.lower() or "issue" in finding.lower() or "problem" in finding.lower():
            issues.append(f"{agent_type}: {finding[:100]}")

        if issues:
          findings_summary.append(f"**{service} ({region})**: {'; '.join(issues)}")

      synthesis_message = f"""Synthesize the diagnosis for this production incident.

**Problem**: {request.problem}
**Environment**: {request.environment}

**Investigation Scale**:
- {len(REGIONS)} regions investigated
- {len(SERVICES)} services per region
- {len(SPECIALISTS)} diagnostic agents per service
- Total: {len(results)} services analyzed with {len(results) * len(SPECIALISTS)} agent findings

**Key Findings Summary**:
{chr(10).join(findings_summary) if findings_summary else "No critical issues detected across services."}

Based on this large-scale investigation, provide:
1. Root cause analysis
2. Immediate remediation steps
3. Long-term prevention measures
4. Recommended monitoring improvements"""

      diagnosis_text = ""
      try:
        async for response in agent.send_stream(synthesis_message, timeout=SYNTHESIS_TIMEOUT):
          snippet = response.snippet
          text = extract_text_from_snippet(snippet)
          if text:
            diagnosis_text += text
            yield make_event("text", text=text)
      except asyncio.TimeoutError:
        diagnosis_text += "\n\n[WARNING: Synthesis timed out, partial results shown]"
        yield make_event("text", text="\n\n[WARNING: Synthesis timed out]")

      # Update session and graph
      sessions[session_id]["diagnosis"] = diagnosis_text
      sessions[session_id]["status"] = "completed"
      sessions[session_id]["phase"] = "diagnosis_complete"
      sessions[session_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"
      sessions[session_id]["diagnosis_results"] = results

      await update_node_status(synthesis_node_id, "completed", {"diagnosis": diagnosis_text})
      await add_transcript_entry(synthesis_node_id, "model", diagnosis_text[:500] + "..." if len(diagnosis_text) > 500 else diagnosis_text)
      await update_node_status(root_id, "completed")

      total_duration = (datetime.utcnow() - diagnosis_start_time).total_seconds()
      await add_transcript_entry(root_id, "system", f"Diagnosis complete in {round(total_duration, 1)}s with {len(results) * len(SPECIALISTS)} agents")
      graph_state["status"] = "completed"

      yield make_event("diagnosis_complete",
        session_id=session_id,
        status="completed",
        services_analyzed=len(results),
        total_agents=len(results) * len(SPECIALISTS),
        duration_seconds=round(total_duration, 2),
        progress=100
      )

    except Exception as e:
      logger.error(f"Error in distributed diagnosis: {str(e)}")
      import traceback
      logger.error(f"Traceback: {traceback.format_exc()}")
      sessions[session_id]["status"] = "error"
      sessions[session_id]["phase"] = "diagnosis_failed"
      sessions[session_id]["message"] = str(e)

      await update_node_status(root_id, "error")
      await add_transcript_entry(root_id, "error", f"Diagnosis failed: {str(e)}")
      graph_state["status"] = "completed"

      yield make_event("error",
        session_id=session_id,
        message=str(e)
      )

    finally:
      if synthesis_agent_name:
        try:
          await Agent.stop(node, synthesis_agent_name)
        except Exception as cleanup_error:
          logger.warning(f"Failed to cleanup agent {synthesis_agent_name}: {cleanup_error}")

  return StreamingResponse(stream_swarm_diagnosis(), media_type="application/x-ndjson")


# Backward compatibility alias
@app.post("/diagnose/distributed")
async def diagnose_distributed(request: DiagnoseRequest, node: NodeDep):
  """Alias for /diagnose/swarm (backward compatibility)."""
  return await diagnose_swarm_endpoint(request, node)


# === Start Node ===

Node.start(http_server=HttpServer(app=app))

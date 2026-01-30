# SRE Incident Diagnosis System - Architecture Document

## Overview

This system enables developers to diagnose infrastructure problems directly from Cursor IDE by spinning up autonomous diagnostic agents in Autonomy Computer, with secure credential retrieval from 1Password via human-in-the-loop approval.

## User Journey

```
1. Developer gets paged about a production issue (e.g., PagerDuty alert)
2. Opens Cursor, types: "Diagnose the database connection failures in prod"
3. Cursor Agent calls Autonomy's /diagnose API
4. Autonomy Orchestrator analyzes the problem, determines needed credentials
5. Orchestrator requests approval: "I need read-only access to: prod-db, aws-cloudwatch"
6. Developer sees approval request in Cursor, types "approve"
7. Autonomy retrieves credentials from 1Password (credentials never touch LLM)
8. Diagnostic agents run in parallel across runner pods
9. Findings stream back to Cursor in real-time
10. Developer receives a comprehensive diagnosis report with root cause analysis
```

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CURSOR IDE                                    │
│                                                                            │
│  ┌──────────────┐    ┌─────────────────────┐    ┌───────────────────────┐  │
│  │  Developer   │───▶│   Cursor Agent      │───▶│    Cursor Hooks       │  │
│  │  (on-call)   │    │   (Claude/GPT)      │    │  • sessionStart       │  │
│  └──────────────┘    │                     │    │  • 1Password env      │  │
│        ▲             │  "Diagnose DB       │    │  • audit logging      │  │
│        │             │   issues in prod"   │    └───────────────────────┘  │
│        │             └──────────┬──────────┘                               │
│        │                        │                                          │
│    Approval                HTTP POST /diagnose                             │
│    Request                      │                                          │
│        │                        ▼                                          │
└────────┼────────────────────────┼──────────────────────────────────────────┘
         │                        │
         │    ┌───────────────────┴───────────────────┐
         │    │         AUTONOMY COMPUTER             │
         │    │              (Zone)                   │
         │    │                                       │
         │    │  ┌─────────────────────────────────┐  │
         │    │  │         MAIN POD (public)       │  │
         │    │  │                                 │  │
         │    │  │  ┌───────────────────────────┐  │  │
         │    │  │  │     FastAPI Server        │  │  │
         │    │  │  │  POST /diagnose           │  │  │
         │    │  │  │  POST /approve/{session}  │◀─┼──┼──── Approval response
         │    │  │  │  GET /status/{session}    │  │  │
         │    │  │  │  WS /stream/{session}     │──┼──┼───▶ Real-time updates
         │    │  │  └───────────┬───────────────┘  │  │
         │    │  │              │                  │  │
         │    │  │  ┌───────────▼───────────────┐  │  │
         │    │  │  │   Orchestrator Agent      │  │  │
         │    │  │  │  • Analyzes problem       │  │  │
         └────┼──┼──┼──• ask_user_for_input ────┼──┘  │
              │  │  │  • Dispatches workers     │     │
              │  │  │  • Synthesizes report     │     │
              │  │  └───────────┬───────────────┘     │
              │  │              │                     │
              │  │  ┌───────────▼───────────────┐     │
              │  │  │   1Password MCP Server    │     │
              │  │  │  • Retrieves credentials  │     │
              │  │  │  • op://vault/item refs   │     │
              │  │  └───────────────────────────┘     │
              │  └─────────────────────────────────┘  │
              │                                       │
              │  ┌─────────────────────────────────┐  │
              │  │    RUNNER PODS (clones: N)      │  │
              │  │                                 │  │
              │  │  ┌─────────┐ ┌─────────┐ ┌────┐ │  │
              │  │  │ DB      │ │ Cloud   │ │Log │ │  │
              │  │  │ Agent   │ │ Agent   │ │Agt │ │  │
              │  │  └────┬────┘ └────┬────┘ └─┬──┘ │  │
              │  │       │           │        │    │  │
              │  │  ┌────▼────────── ▼────────▼──┐ │  │
              │  │  │   MCP Tools (per pod)      │ │  │
              │  │  │  • postgres, mysql         │ │  │
              │  │  │  • aws, gcp, k8s           │ │  │
              │  │  │  • datadog, cloudwatch     │ │  │
              │  │  └────────────────────────────┘ │  │
              │  └─────────────────────────────────┘  │
              └───────────────────────────────────────┘
```

## The Credential Approval Flow (Critical Security Path)

This is the most important security mechanism - **credentials NEVER enter the LLM conversation**.

### Flow Phases

```
PHASE 1: Analysis
─────────────────
Developer → Cursor: "Diagnose database connection failures in prod"
Cursor Agent → Autonomy: POST /diagnose { problem: "...", env: "prod" }
Orchestrator: Analyzes problem, determines needed tools/credentials

PHASE 2: Approval Request
─────────────────────────
Orchestrator calls ask_user_for_input:
  "To diagnose this issue, I need access to:
   • Production Database (read-only) - op://Infrastructure/prod-db
   • AWS CloudWatch Logs - op://Infrastructure/aws-readonly

   Reply 'approve' to grant access, or 'deny' to cancel."

Autonomy → Cursor: { phase: "waiting_for_input", prompt: "..." }
Cursor Agent → Developer: Shows approval request

PHASE 3: Credential Retrieval (on approval)
───────────────────────────────────────────
Developer → Cursor: "approve"
Cursor → Autonomy: POST /approve/session-123 { approved: true }

Autonomy (internal, NOT via LLM):
  1. 1Password MCP Server retrieves actual credentials
  2. Credentials injected directly into tool environment variables
  3. Tools execute with credentials (credentials never in conversation)

PHASE 4: Parallel Diagnosis
───────────────────────────
Orchestrator dispatches workers to runner pods:
  - DB Agent: Queries pg_stat_activity, checks connections, slow queries
  - Cloud Agent: Pulls CloudWatch metrics, checks EC2/RDS status
  - Log Agent: Searches for error patterns, correlates events
  - K8s Agent: Checks pod status, resource limits, network policies

PHASE 5: Synthesis & Report
───────────────────────────
Results stream back → Orchestrator synthesizes findings
→ Root cause analysis generated
→ Report sent to Cursor → Developer
```

### Credential Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    CREDENTIAL FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Agent requests: "I need access to prod-db"                  │
│     └─▶ This is a REFERENCE (op://vault/prod-db), not the cred  │
│                                                                 │
│  2. Human approves the reference                                │
│     └─▶ Approval recorded in session state                      │
│                                                                 │
│  3. 1Password MCP server retrieves actual credential            │
│     └─▶ Credential goes to ENVIRONMENT VARIABLE                 │
│     └─▶ NOT returned to the LLM conversation                    │
│                                                                 │
│  4. Database MCP tool reads credential from environment         │
│     └─▶ Tool executes query with real credential                │
│     └─▶ Only RESULTS returned to LLM                            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  WHAT THE LLM SEES:                                             │
│  ✓ "User approved access to prod-db"                            │
│  ✓ "Query returned 150 active connections"                      │
│  ✗ Never sees: "password=s3cr3t123"                             │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Cursor Hooks

Cursor hooks provide the bridge between the developer's IDE and the Autonomy system.

**Purpose:**
- Validate 1Password environment is properly configured
- Inject Autonomy cluster/zone info into session
- Audit all tool calls for security compliance
- Clean up sessions on completion

**Key Hooks:**
- `sessionStart`: Initialize SRE session, validate 1Password
- `beforeMCPExecution`: Audit MCP tool calls
- `stop`: Clean up Autonomy sessions

### 2. Autonomy Main Pod

The main pod is the public entry point that handles:

**FastAPI Server:**
- `POST /diagnose` - Start a diagnosis session
- `POST /approve/{session_id}` - Handle credential approval
- `GET /status/{session_id}` - Get diagnosis status
- `WS /stream/{session_id}` - Stream real-time updates

**Orchestrator Agent:**
- Analyzes the reported problem
- Determines which diagnostic agents are needed
- Requests credential approval via `ask_user_for_input`
- Dispatches workers to runner pods
- Synthesizes findings into final report

**1Password MCP Server:**
- Runs as a sidecar container
- Retrieves credentials on approval
- Injects credentials into environment (not conversation)

### 3. Autonomy Runner Pods

Multiple cloned pods for parallel diagnosis:

**Diagnostic Agents:**
- **Database Agent**: Connection analysis, slow queries, locks, replication
- **Cloud Agent**: AWS/GCP/Azure resource status, metrics, quotas
- **Log Agent**: Error pattern matching, event correlation
- **Kubernetes Agent**: Pod health, resource limits, network issues

**MCP Tools (per pod):**
- Database connectors (postgres, mysql, mongodb, redis)
- Cloud SDKs (aws, gcp, azure)
- Kubernetes tools (kubectl)
- Log aggregators (datadog, cloudwatch, splunk)

## Data Flow Sequence Diagram

```
Developer          Cursor Agent         Autonomy            1Password         Diagnostic
    │                   │                  │                    │               Agents
    │                   │                  │                    │                  │
    │ "Diagnose DB"     │                  │                    │                  │
    │──────────────────▶│                  │                    │                  │
    │                   │                  │                    │                  │
    │                   │ POST /diagnose   │                    │                  │
    │                   │─────────────────▶│                    │                  │
    │                   │                  │                    │                  │
    │                   │                  │ Analyze problem    │                  │
    │                   │                  │ Determine creds    │                  │
    │                   │                  │                    │                  │
    │                   │ ask_user_input   │                    │                  │
    │                   │◀─────────────────│                    │                  │
    │                   │ "Need prod-db,   │                    │                  │
    │                   │  aws-readonly"   │                    │                  │
    │                   │                  │                    │                  │
    │ "Approve access?" │                  │                    │                  │
    │◀──────────────────│                  │                    │                  │
    │                   │                  │                    │                  │
    │ "approve"         │                  │                    │                  │
    │──────────────────▶│                  │                    │                  │
    │                   │                  │                    │                  │
    │                   │ POST /approve    │                    │                  │
    │                   │─────────────────▶│                    │                  │
    │                   │                  │                    │                  │
    │                   │                  │ get_secret         │                  │
    │                   │                  │───────────────────▶│                  │
    │                   │                  │                    │                  │
    │                   │                  │ creds → env vars   │                  │
    │                   │                  │◀───────────────────│                  │
    │                   │                  │                    │                  │
    │                   │                  │ Dispatch workers   │                  │
    │                   │                  │──────────────────────────────────────▶│
    │                   │                  │                    │                  │
    │                   │  Stream updates  │                    │   Query DBs      │
    │                   │◀─────────────────│                    │   Check AWS      │
    │ "Checking DB..."  │                  │                    │   Parse logs     │
    │◀──────────────────│                  │                    │                  │
    │                   │                  │                    │                  │
    │                   │                  │ Findings           │                  │
    │                   │◀─────────────────│◀──────────────────────────────────────│
    │                   │                  │                    │                  │
    │ "Root cause:      │                  │                    │                  │
    │  Connection pool  │                  │                    │                  │
    │  exhausted..."    │                  │                    │                  │
    │◀──────────────────│                  │                    │                  │
```

## Key Design Decisions

### Why Autonomy (not just Cursor MCP)?

| Aspect | Cursor MCP Only | Autonomy |
|--------|-----------------|----------|
| Parallel agents | Limited by local resources | Unlimited via clones |
| Long-running diagnosis | Timeout issues | Handles gracefully |
| State management | Per-session only | Persistent memory |
| Tool isolation | Same process | Container isolation |
| Scaling | Single machine | Distributed across pods |
| Cost | Uses your API keys | Managed infrastructure |

### Why Human-in-the-Loop for Credentials?

- **Principle of Least Privilege**: Only explicitly requested credentials are accessible
- **Audit Trail**: Every approval is logged with timestamp and context
- **Context Awareness**: Developer sees what's being accessed and why
- **Revocability**: Developer can deny at any point
- **Compliance**: Meets SOC2/ISO27001 requirements for access control

### Why 1Password (not environment variables)?

- **Dynamic Retrieval**: Just-in-time access, not pre-loaded secrets
- **Rotation**: Credentials can rotate without redeployment
- **Scoping**: Different credentials per session/environment
- **Audit**: 1Password logs all secret access
- **Revocation**: Instant credential revocation if needed

## Diagnostic Capabilities

### Database Diagnostics
- Connection pool analysis (active, idle, waiting)
- Slow query identification
- Lock contention detection
- Replication lag monitoring
- Table bloat analysis
- Index usage statistics

### Cloud Diagnostics (AWS/GCP/Azure)
- Service health status
- Resource utilization metrics
- Quota/limit analysis
- Network connectivity tests
- IAM permission verification
- Cost anomaly detection

### Kubernetes Diagnostics
- Pod health and restart counts
- Resource limit analysis (CPU/memory)
- Network policy verification
- Service mesh health
- Ingress/load balancer status
- PersistentVolume issues

### Log Analysis
- Error pattern matching
- Event correlation across services
- Anomaly detection
- Stack trace analysis
- Request tracing

## Security Considerations

### Credential Security
1. Credentials never enter LLM context
2. Short-lived credential sessions
3. Read-only access by default
4. Audit logging for all access

### Network Security
1. All traffic over HTTPS/TLS
2. Autonomy zone runs in isolated network
3. MCP tools access only approved endpoints
4. No credential storage in Autonomy

### Access Control
1. Developer must explicitly approve each credential request
2. Credentials scoped to specific resources
3. Session-based access (expires automatically)
4. Deny-by-default policy

## Open Questions / Dependencies

### 1Password Integration
- **Question**: Exact container image for 1Password MCP server?
- **Note**: User mentioned "a server that can be run in autonomy as a container to talk to 1password"
- **Action**: Confirm image reference and API documentation

### Cursor ↔ Autonomy Session Binding
- **Question**: How does Cursor track which Autonomy session to approve?
- **Options**:
  - Return session_id in initial response, Cursor tracks it
  - Use stable identifier from Cursor hooks (conversation_id)
- **Recommendation**: Use session_id returned from /diagnose

### Credential Injection Mechanism
- **Question**: How exactly do credentials flow from 1Password to MCP tools?
- **Options**:
  - Environment variables via `--pass-environment`
  - Temporary files with restricted permissions
  - Unix domain sockets
- **Recommendation**: Environment variables (simplest, well-supported)

### Streaming Protocol
- **Question**: Best protocol for real-time updates to Cursor?
- **Options**:
  - WebSocket (bidirectional, complex)
  - Server-Sent Events (simpler, unidirectional)
  - Polling (simplest, higher latency)
- **Recommendation**: Start with SSE, upgrade to WebSocket if needed

## References

### Autonomy Documentation
- [Creating Autonomy Apps](https://autonomy.computer/docs/_for-coding-agents/create-a-new-autonomy-app.md)
- [Custom APIs](https://autonomy.computer/docs/_for-coding-agents/create-custom-apis.md)
- [Tools](https://autonomy.computer/docs/_for-coding-agents/tools.md)
- [Workers](https://autonomy.computer/docs/_for-coding-agents/workers.md)

### Cursor Documentation
- [Cursor Hooks](https://cursor.com/docs/agent/hooks.md)

### 1Password Documentation
- [1Password Cursor Hooks](https://developer.1password.com/docs/cursor-hooks/)
- [1Password Environments](https://developer.1password.com/docs/environments)

### Related Projects
- [Autonomous SRE Agent](https://github.com/madhurprash/Autonomous_SRE_Agent) - AWS Bedrock-based SRE agent (cloned to (external reference) for reference)

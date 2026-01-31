# SRE Incident Diagnosis System - Architecture Document

## Overview

This system enables developers to diagnose infrastructure problems by running autonomous diagnostic agents in Autonomy Computer, with secure credential retrieval from 1Password via human-in-the-loop approval.

## User Journey

```
1. Developer gets paged about a production issue (e.g., PagerDuty alert)
2. Opens Cursor, types: "Diagnose the database connection failures in prod"
3. Cursor Agent calls Autonomy's /diagnose API
4. Analysis Agent examines the problem, identifies needed credentials
5. System shows approval request: "I need access to: prod-db, aws-cloudwatch"
6. Developer approves via POST /approve/{session_id}
7. Autonomy retrieves credentials from 1Password (credentials never touch LLM)
8. Specialist agents run in parallel: Database, Cloud, Kubernetes
9. Synthesis Agent combines findings into comprehensive diagnosis
10. Developer receives root cause analysis and remediation steps
```

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CURSOR IDE                                    │
│                                                                            │
│  ┌──────────────┐    ┌─────────────────────┐    ┌───────────────────────┐  │
│  │  Developer   │───▶│   Cursor Agent      │───▶│    Cursor Hooks       │  │
│  │  (on-call)   │    │   (Claude)          │    │  • session-start      │  │
│  └──────────────┘    │                     │    │  • pre-tool-call      │  │
│                      │                     │    │  • post-tool-call     │  │
│        ▲             │  "Diagnose DB       │    └───────────────────────┘  │
│        │             │   issues in prod"   │                               │
│        │             └──────────┬──────────┘                               │
│        │                        │                                          │
│    Streaming                HTTP POST /diagnose                            │
│    Response                     │                                          │
│        │                        ▼                                          │
└────────┼────────────────────────┼──────────────────────────────────────────┘
         │                        │
         │    ┌───────────────────┴───────────────────┐
         │    │         AUTONOMY COMPUTER             │
         │    │              (Zone: srediag)          │
         │    │                                       │
         │    │  ┌─────────────────────────────────┐  │
         │    │  │         MAIN POD (public)       │  │
         │    │  │                                 │  │
         │    │  │  ┌───────────────────────────┐  │  │
         │    │  │  │     main container        │  │  │
         │    │  │  │                           │  │  │
         │    │  │  │  ┌─────────────────────┐  │  │  │
         │    │  │  │  │    FastAPI Server   │  │  │  │
         │    │  │  │  │  POST /diagnose     │  │  │  │
         │    │  │  │  │  POST /approve/{id} │◀─┼──┼──┼── Approval
         │    │  │  │  │  GET /status/{id}   │  │  │  │
         │    │  │  │  │  GET /sessions      │──┼──┼──┼─▶ NDJSON Stream
         │    │  │  │  └────────┬────────────┘  │  │  │
         │    │  │  │           │               │  │  │
         │    │  │  │  ┌────────▼────────────┐  │  │  │
         │    │  │  │  │   Agent Orchestration│  │  │  │
         │    │  │  │  │  • Analysis Agent   │  │  │  │
         │    │  │  │  │  • Specialist Agents│  │  │  │
         │    │  │  │  │  • Synthesis Agent  │  │  │  │
         │    │  │  │  └────────┬────────────┘  │  │  │
         │    │  │  │           │               │  │  │
         │    │  │  │  ┌────────▼────────────┐  │  │  │
         │    │  │  │  │  Diagnostic Tools   │  │  │  │
         │    │  │  │  │  (Python functions) │  │  │  │
         │    │  │  │  └─────────────────────┘  │  │  │
         │    │  │  └───────────────────────────┘  │  │
         │    │  │                                 │  │
         │    │  │  ┌───────────────────────────┐  │  │
         │    │  │  │   onepass container       │  │  │
         │    │  │  │   (Mock 1Password Server) │  │  │
         │    │  │  │   HTTP REST API :8080     │  │  │
         │    │  │  └───────────────────────────┘  │  │
         │    │  │                                 │  │
         │    │  └─────────────────────────────────┘  │
         │    │                                       │
         │    └───────────────────────────────────────┘
         │
         └──────────── Streaming diagnosis results
```

## Two-Phase Diagnosis Flow

The system uses a two-phase approach with separate agents for analysis and diagnosis.

### Why Two Phases?

The Autonomy framework's `ask_user_for_input` resume flow has a known issue where tool_result messages get appended to the end of conversation history instead of immediately after the tool_use message, violating Claude's API requirements. The two-phase approach (separate agents) works around this limitation.

### Phase 1: Analysis

```
Developer → POST /diagnose
              │
              ▼
       ┌─────────────────┐
       │  Analysis Agent │  (no tools)
       │                 │
       │  • Examines problem description
       │  • Identifies potential root causes
       │  • Determines needed credentials
       │  • Outputs op:// references
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │  Regex Extract  │  Extract op:// refs from response
       └────────┬────────┘
                │
                ▼
       Status: waiting_for_approval
       Response: List of requested credentials
```

### Phase 2: Diagnosis (after approval)

```
Developer → POST /approve/{session_id}
              │
              ▼
       ┌─────────────────────┐
       │  Credential         │
       │  Retrieval          │  HTTP calls to onepass container
       │  (Mock 1Password)   │  Credentials stored in session
       └────────┬────────────┘
                │
                ▼
       ┌─────────────────────────────────────────┐
       │     Specialist Agents (parallel)        │
       │                                         │
       │  ┌──────────┐ ┌──────────┐ ┌─────────┐  │
       │  │ Database │ │  Cloud   │ │   K8s   │  │
       │  │ Specialist│ │Specialist│ │Specialist│ │
       │  └────┬─────┘ └────┬─────┘ └────┬────┘  │
       │       │            │            │       │
       │       ▼            ▼            ▼       │
       │    Tools:       Tools:       Tools:     │
       │  • query_db_  • get_cloud  • check_k8s │
       │    connections  watch_      _pods      │
       │  • query_slow   metrics   • get_app_   │
       │    _queries   • check_      logs       │
       │                 instance_              │
       │                 health                 │
       └─────────────────┬───────────────────────┘
                         │ asyncio.gather()
                         ▼
       ┌─────────────────────┐
       │   Synthesis Agent   │  (no tools)
       │                     │
       │  • Combines all specialist findings
       │  • Identifies root cause
       │  • Recommends remediation
       │  • Suggests monitoring improvements
       └─────────────────────┘
                │
                ▼
       Status: completed
       Response: Full diagnosis report
```

## Credential Security Model

**Critical**: Credentials NEVER enter the LLM conversation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CREDENTIAL FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Analysis Agent outputs: "I need access to prod-db"          │
│     └─▶ This is a REFERENCE (op://vault/prod-db), not the cred  │
│                                                                 │
│  2. Human approves the reference via /approve endpoint          │
│     └─▶ Approval recorded in session state                      │
│                                                                 │
│  3. System retrieves credential from 1Password HTTP API         │
│     └─▶ Credential stored in session["credentials"] dict        │
│     └─▶ NOT returned to any LLM conversation                    │
│                                                                 │
│  4. Diagnostic tools could access credentials from session      │
│     └─▶ Tool executes with real credential                      │
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

### 1. FastAPI Server (main container)

The main entry point handling HTTP requests and agent orchestration.

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML UI |
| `/health` | GET | Health check with 1Password status |
| `/diagnose` | POST | Start Phase 1 analysis |
| `/approve/{session_id}` | POST | Approve credentials, start Phase 2 |
| `/status/{session_id}` | GET | Get session status |
| `/sessions` | GET | List all sessions |
| `/sessions/{session_id}` | DELETE | Delete a session |

**Streaming Protocol:**
- Uses NDJSON (Newline-delimited JSON) via `StreamingResponse`
- Media type: `application/x-ndjson`
- Each line is a complete JSON object with a `type` field

### 2. Agent Types

| Agent | Phase | Tools | Purpose |
|-------|-------|-------|---------|
| Analysis Agent | 1 | None | Analyze problem, identify credentials |
| Database Specialist | 2 | `query_db_connections`, `query_slow_queries` | Database diagnostics |
| Cloud Specialist | 2 | `get_cloudwatch_metrics`, `check_instance_health` | AWS/cloud diagnostics |
| Kubernetes Specialist | 2 | `check_kubernetes_pods`, `get_application_logs` | K8s diagnostics |
| Synthesis Agent | 2 | None | Combine findings, generate report |

### 3. Mock 1Password Server (onepass container)

A FastAPI server simulating 1Password Connect API.

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/secrets/{reference}` | GET | Retrieve secret by op:// reference |
| `/vaults` | GET | List available vaults |
| `/vaults/{vault_id}/items` | GET | List items in vault |

**Available Mock Credentials:**
```
op://Infrastructure/prod-db/password
op://Infrastructure/prod-db/username
op://Infrastructure/prod-db/host
op://Infrastructure/staging-db/password
op://Infrastructure/staging-db/username
op://Infrastructure/aws-cloudwatch/access-key
op://Infrastructure/aws-cloudwatch/secret-key
op://Infrastructure/aws-ec2/access-key
op://Infrastructure/aws-ec2/secret-key
op://Infrastructure/k8s-prod/token
op://Infrastructure/datadog/api-key
op://Infrastructure/grafana/admin-password
op://Services/payment-api/api-key
op://Services/email-service/smtp-password
```

### 4. Diagnostic Tools (Python Functions)

| Tool | Parameters | Returns |
|------|------------|---------|
| `query_db_connections` | `environment` | Connection pool stats |
| `query_slow_queries` | `environment`, `threshold_ms` | Slow query list |
| `get_cloudwatch_metrics` | `service`, `metric`, `period_minutes` | Metric datapoints |
| `check_instance_health` | `instance_id` | Instance status, health checks |
| `check_kubernetes_pods` | `namespace`, `label_selector` | Pod list with status |
| `get_application_logs` | `service`, `level`, `limit` | Recent log entries |

### 5. Cursor Hooks

Located in `.cursor/` directory for IDE integration.

**Hooks Configuration (`.cursor/hooks.json`):**
| Hook | Trigger | Purpose |
|------|---------|---------|
| `session-start` | New conversation | Sets `SRE_DIAGNOSE_URL` env var and provides context |
| `pre-tool-call` | Before terminal commands | Intercepts `sre-diagnose` commands and routes to API |
| `post-tool-call` | After tool execution | Logs activity for auditing |

**Purpose:**
- Provide context to Cursor Agent about SRE capabilities
- Intercept `sre-diagnose` commands and guide to API usage
- Audit tool calls for security compliance

## Specialist Agent Selection

The system automatically determines which specialists to run based on keywords:

| Agent | Trigger Keywords |
|-------|------------------|
| Database | database, db, connection pool, query, sql, postgres, mysql, timeout, connection, transaction, deadlock |
| Cloud | aws, ec2, cloudwatch, instance, cpu, memory usage, scaling, load balancer, rds |
| Kubernetes | kubernetes, k8s, pod, container, deployment, replica, crashloop, oom, health check, liveness, readiness |

**Default:** If no keywords match, runs Database + Cloud specialists.

## Session State

Sessions are stored in-memory (for MVP) with this structure:

```python
{
  "id": "abc123",
  "status": "analyzing|waiting_for_approval|retrieving_credentials|diagnosing|completed|error|denied",
  "phase": "analysis|awaiting_approval|credential_retrieval|diagnosis|diagnosis_complete|...",
  "problem": "Original problem description",
  "environment": "prod|staging|dev",
  "context": {},
  "created_at": "2024-01-15T10:30:00Z",
  "analysis": "Full analysis text from Phase 1",
  "requested_credentials": ["op://...", "op://..."],
  "credentials": {"op://...": "actual_value"},  # Never sent to LLM
  "credentials_retrieved": ["op://...", "op://..."],
  "specialist_findings": [...],
  "diagnosis": "Final diagnosis text"
}
```

## Streaming Event Types

| Event Type | Phase | Description |
|------------|-------|-------------|
| `session_start` | 1 | Analysis started |
| `text` | 1, 2 | Streaming text from agent |
| `approval_required` | 1 | Analysis complete, awaiting approval |
| `approval_accepted` | 2 | Approval received |
| `credential_retrieved` | 2 | Individual credential retrieved |
| `diagnosis_started` | 2 | Beginning specialist dispatch |
| `specialists_selected` | 2 | Which specialists will run |
| `specialist_complete` | 2 | Individual specialist finished |
| `synthesis_started` | 2 | Beginning final synthesis |
| `diagnosis_complete` | 2 | All done |
| `error` | Any | Error occurred |

## Key Design Decisions

### Why Two-Phase with Separate Agents?

The Autonomy framework's `ask_user_for_input` tool has a bug where resume messages violate Claude's API message ordering requirements. Using separate agents for analysis and diagnosis avoids this issue while maintaining the human-in-the-loop security model.

### Why Parallel Specialists with asyncio.gather?

Running specialists in parallel significantly reduces total diagnosis time. Each specialist is independent and can query its domain-specific tools concurrently. The synthesis agent then combines all findings.

### Why NDJSON Streaming (not WebSocket)?

- Simpler to implement and debug
- Works with standard HTTP clients (curl)
- Unidirectional streaming is sufficient
- Each line is independently parseable
- Better compatibility with load balancers

### Why Mock 1Password HTTP Server?

- Allows development without real 1Password setup
- Tests the credential flow end-to-end
- HTTP REST is simpler than MCP integration
- Easy to replace with real 1Password Connect later

### Why Credentials in Session Dict (not Environment Variables)?

- More flexible for dynamic credential sets
- Easier to track which credentials were retrieved
- Supports multiple credentials per session
- Still secure: never sent to LLM conversation

## Zone Configuration

```yaml
# autonomy.yaml
name: srediag
pods:
  - name: main-pod
    public: true
    containers:
      - name: main
        image: main
        env:
          # 1Password mode: "mock" (default) or "sdk" (production)
          - ONEPASSWORD_MODE: "mock"
          # Uncomment for production:
          # - ONEPASSWORD_MODE: "sdk"
          # - OP_SERVICE_ACCOUNT_TOKEN: secrets.OP_SERVICE_ACCOUNT_TOKEN
      - name: onepass
        image: mock-1password
```

**Key Points:**
- Single pod architecture (no runner pods in current implementation)
- `onepass` container accessible via `localhost:8080` from main container (mock mode)
- Public endpoint exposed for external access
- Supports dual 1Password modes: `mock` (development) and `sdk` (production)

## Security Considerations

### Credential Security
1. Credentials never enter LLM context
2. Stored only in server-side session dict
3. Human approval required for each session
4. Mock server simulates secure retrieval

### Network Security
1. All traffic over HTTPS (Autonomy handles TLS)
2. Mock 1Password only accessible within pod (localhost)
3. No credential storage persistence (in-memory only)

### Access Control
1. Developer must explicitly approve each credential request
2. Analysis agent cannot trigger credential retrieval
3. Session-based access (manual cleanup for now)

## Implementation Status

### Completed
- [x] **Phase 1**: Foundation (MVP) - Basic orchestrator, streaming, dashboard
- [x] **Phase 2**: Mock 1Password & Credential Flow - Secure credential retrieval
- [x] **Phase 3**: Diagnostic Tools & Specialized Agents - Parallel diagnosis
- [x] **Phase 4**: Polish & Production Readiness - Error handling, timeouts, progress tracking
- [x] **Phase 5**: Real 1Password Integration - SDK support for production use
- [x] **Cursor Hooks**: IDE integration with session-start, pre-tool-call, post-tool-call

### Future Enhancements
- [ ] **Distributed Processing**: Runner pods for true parallel execution
- [ ] **Real Diagnostic Tools**: Actual database, AWS, Kubernetes integrations
- [ ] **Persistent Sessions**: Database-backed session storage

## References

### Autonomy Documentation
- [Creating Autonomy Apps](https://autonomy.computer/docs/_for-coding-agents/create-a-new-autonomy-app)
- [Custom APIs](https://autonomy.computer/docs/_for-coding-agents/create-custom-apis)
- [Tools](https://autonomy.computer/docs/_for-coding-agents/tools)

### 1Password Documentation
- [1Password SDKs](https://developer.1password.com/docs/sdks/)
- [Service Accounts](https://developer.1password.com/docs/service-accounts/)
- [Load Secrets with SDKs](https://developer.1password.com/docs/sdks/load-secrets/)
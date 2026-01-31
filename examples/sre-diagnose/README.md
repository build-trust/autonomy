# SRE Incident Diagnosis

An Autonomy app that enables developers to diagnose infrastructure problems using autonomous diagnostic agents with secure credential retrieval via human-in-the-loop approval.

**Version**: 0.4.0

## Overview

This app demonstrates:

- **Two-Phase Diagnosis Flow**: Analysis → Approval → Diagnosis
- **Specialized Diagnostic Agents**: Database, Cloud, and Kubernetes specialists
- **Mock Diagnostic Tools**: Realistic tool responses for testing
- **Human-in-the-Loop Approval**: Requests permission before accessing credentials
- **Secure Credential Flow**: Credentials retrieved from mock 1Password, never exposed to LLM
- **Real-time Streaming**: Live updates with progress indicators
- **Cursor Hooks Integration**: Deep integration with Cursor IDE
- **Production-Ready Features**: Timeouts, retry logic, session management

## Quick Start

### 1. Deploy the Zone

```bash
cd autonomy/examples/sre-diagnose
autonomy zone deploy
```

The app will be available at:
- **Dashboard**: https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer
- **API**: https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/health

### 2. Test the API

```bash
# Health check
curl https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/health

# Start diagnosis (Phase 1: Analysis)
curl -X POST https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/diagnose \
  -H "Content-Type: application/json" \
  -d '{"problem": "Database connection pool exhausted", "environment": "prod"}'

# Check session status
curl https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/sessions

# Approve and run diagnosis (Phase 2)
curl -X POST https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/approve/{session_id} \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DIAGNOSIS FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: ANALYSIS                                                          │
│  ┌─────────────┐                                                            │
│  │  Analysis   │ → Identifies problem type, root causes, needed credentials │
│  │   Agent     │ → Extracts op:// credential references from response       │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │  APPROVAL   │ ← User approves/denies credential access                   │
│  │  REQUIRED   │                                                            │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  PHASE 2: DIAGNOSIS                                                         │
│  ┌─────────────┐                                                            │
│  │  Credential │ → Retrieves approved credentials from mock 1Password       │
│  │  Retrieval  │                                                            │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────┐                        │
│  │         SPECIALIST AGENTS (run in parallel)     │                        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │                        │
│  │  │ Database │  │  Cloud   │  │  Kubernetes  │   │                        │
│  │  │Specialist│  │Specialist│  │  Specialist  │   │                        │
│  │  └──────────┘  └──────────┘  └──────────────┘   │                        │
│  │       │             │              │            │                        │
│  │       ▼             ▼              ▼            │                        │
│  │  ┌──────────────────────────────────────────┐   │                        │
│  │  │           DIAGNOSTIC TOOLS               │   │                        │
│  │  │  • query_db_connections                  │   │                        │
│  │  │  • query_slow_queries                    │   │                        │
│  │  │  • get_cloudwatch_metrics                │   │                        │
│  │  │  • check_instance_health                 │   │                        │
│  │  │  • check_kubernetes_pods                 │   │                        │
│  │  │  • get_application_logs                  │   │                        │
│  │  └──────────────────────────────────────────┘   │                        │
│  └─────────────────────────────────────────────────┘                        │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │  Synthesis  │ → Combines all specialist findings into final diagnosis    │
│  │   Agent     │ → Provides root cause, remediation, and monitoring advice  │
│  └─────────────┘                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Specialist Agents

The system automatically selects which specialist agents to run based on the problem description:

| Agent | Triggers | Tools Used |
|-------|----------|------------|
| **Database Specialist** | "database", "db", "connection pool", "query", "sql", "timeout" | `query_db_connections`, `query_slow_queries` |
| **Cloud Specialist** | "aws", "ec2", "cloudwatch", "cpu", "memory", "instance" | `get_cloudwatch_metrics`, `check_instance_health` |
| **Kubernetes Specialist** | "kubernetes", "k8s", "pod", "container", "crashloop", "oom" | `check_kubernetes_pods`, `get_application_logs` |

## Mock Diagnostic Tools

All tools return realistic mock data for testing:

### Database Tools
- **query_db_connections**: Connection pool statistics (active, idle, waiting)
- **query_slow_queries**: Slow query analysis with execution times

### Cloud Tools
- **get_cloudwatch_metrics**: CPU, memory, network, disk metrics
- **check_instance_health**: EC2/RDS instance status and health checks

### Kubernetes Tools
- **check_kubernetes_pods**: Pod status, restarts, resource usage
- **get_application_logs**: Recent error logs with trace IDs

## Available Credentials (Mock 1Password)

The mock 1Password server provides these credentials:

| Reference | Description |
|-----------|-------------|
| `op://Infrastructure/prod-db/password` | Production database password |
| `op://Infrastructure/prod-db/username` | Production database username |
| `op://Infrastructure/prod-db/host` | Production database host |
| `op://Infrastructure/staging-db/password` | Staging database password |
| `op://Infrastructure/staging-db/username` | Staging database username |
| `op://Infrastructure/aws-cloudwatch/access-key` | CloudWatch access key |
| `op://Infrastructure/aws-cloudwatch/secret-key` | CloudWatch secret key |
| `op://Infrastructure/k8s-prod/token` | Kubernetes production token |
| `op://Infrastructure/datadog/api-key` | Datadog API key |

## API Reference

### POST /diagnose

Start a new diagnostic session (Phase 1: Analysis).

**Request:**
```json
{
  "problem": "Database connection pool exhausted",
  "environment": "prod",
  "context": {"service": "api-gateway"}
}
```

**Streaming Response (NDJSON):**
```json
{"type": "session_start", "session_id": "abc123", "status": "analyzing"}
{"type": "text", "text": "## Incident Classification\n..."}
{"type": "approval_required", "session_id": "abc123", "status": "waiting_for_approval", "requested_credentials": ["op://..."]}
```

### POST /approve/{session_id}

Approve credential access and run Phase 2: Diagnosis.

**Request:**
```json
{"approved": true}
```

**Streaming Response:**
```json
{"type": "approval_accepted", "session_id": "abc123", "status": "approved"}
{"type": "credential_retrieved", "reference": "op://...", "success": true}
{"type": "diagnosis_started", "session_id": "abc123", "credentials_retrieved": 5}
{"type": "specialists_selected", "session_id": "abc123", "specialists": ["database", "cloud"]}
{"type": "specialist_complete", "agent": "database_specialist", "status": "completed"}
{"type": "specialist_complete", "agent": "cloud_specialist", "status": "completed"}
{"type": "synthesis_started", "session_id": "abc123"}
{"type": "text", "text": "# Final Diagnosis..."}
{"type": "diagnosis_complete", "session_id": "abc123", "status": "completed"}
```

### GET /status/{session_id}

Get session status with analysis and credentials retrieved.

### GET /sessions

List all diagnostic sessions.

### DELETE /sessions/{session_id}

Delete a session.

### GET /health

Health check including mock 1Password status.

## Cursor IDE Integration

Copy the `.cursor` folder to your project for IDE integration:

```bash
cp -r .cursor /path/to/your/project/
```

Then use natural language to trigger diagnosis:
```
"Can you diagnose why our database connections are failing in production?"
```

## Project Structure

```
sre-diagnose/
├── autonomy.yaml              # Zone configuration (main + onepass containers)
├── README.md                  # This file
├── .cursor/                   # Cursor hooks integration
│   ├── hooks.json
│   └── hooks/*.sh
├── images/
│   ├── main/
│   │   ├── Dockerfile
│   │   ├── main.py           # FastAPI + agents + diagnostic tools
│   │   └── index.html        # Dashboard
│   └── mock-1password/
│       ├── Dockerfile
│       └── server.py         # Mock 1Password Connect server
└── planning/
    ├── ARCHITECTURE.md
    └── IMPLEMENTATION_PLAN.md
```

## Implementation Status

- [x] **Phase 1**: Foundation (MVP) - Basic orchestrator, streaming, dashboard
- [x] **Phase 2**: Mock 1Password & Credential Flow - Secure credential retrieval
- [x] **Phase 3**: Diagnostic Tools & Specialized Agents - Parallel diagnosis
- [x] **Phase 4**: Polish & Production Readiness - Error handling, timeouts, progress tracking
- [ ] **Phase 5**: Distributed Processing (Optional)

## Configuration

Key configuration constants in `main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `ANALYSIS_TIMEOUT` | 120s | Timeout for analysis phase |
| `SPECIALIST_TIMEOUT` | 90s | Timeout for each specialist agent |
| `SYNTHESIS_TIMEOUT` | 120s | Timeout for synthesis phase |
| `CREDENTIAL_RETRY_ATTEMPTS` | 3 | Retries for credential retrieval |
| `SESSION_EXPIRY_HOURS` | 24h | Session auto-cleanup after |
| `MAX_SESSIONS` | 100 | Maximum concurrent sessions |

## Troubleshooting

### Common Issues

**Session stuck in "analyzing" state**
- Check if the analysis agent timed out (default 120s)
- View logs at the Autonomy logs endpoint
- The session will show status "error" if the agent failed

**Credentials not being retrieved**
- Verify the mock 1Password server is running: `curl http://localhost:8080/health`
- Check that the credential reference exists in the mock server
- Retrieval retries 3 times with exponential backoff

**Specialist agents timing out**
- Each specialist has a 90-second timeout
- Partial results are returned if timeout occurs
- Check the `duration_seconds` field in responses

**Dashboard not updating**
- Ensure auto-refresh is enabled (checkbox in Active Sessions)
- Manual refresh: click the Refresh button
- Check browser console for API errors

### Debug Endpoints

```bash
# View all sessions
curl https://<zone-url>/sessions

# Get session details
curl https://<zone-url>/status/{session_id}

# Check credentials stored (without values)
curl https://<zone-url>/debug/sessions/{session_id}/credentials

# Clean up expired sessions
curl -X POST https://<zone-url>/sessions/cleanup
```

### Viewing Logs

Access the Autonomy logs portal for detailed agent execution logs:
```
https://<zone-url>/logs
```

## References

- [Architecture Document](planning/ARCHITECTURE.md)
- [Implementation Plan](planning/IMPLEMENTATION_PLAN.md)
- [1Password Connect API Reference](planning/1PASSWORD_CONNECT_REFERENCE.md)
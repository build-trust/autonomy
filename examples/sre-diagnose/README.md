# SRE Incident Diagnosis

An Autonomy app that enables developers to diagnose infrastructure problems directly from their IDE by spinning up autonomous diagnostic agents with secure credential retrieval via human-in-the-loop approval.

## Overview

This app demonstrates:

- **Orchestrator Agent**: Analyzes incident descriptions and coordinates diagnostic workers
- **Human-in-the-Loop Approval**: Requests permission before accessing credentials
- **Secure Credential Flow**: Credentials never enter the LLM conversation
- **Distributed Diagnostics**: Parallel diagnosis across database, cloud, and Kubernetes systems
- **Real-time Streaming**: Live updates as diagnosis progresses

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CURSOR IDE                               │
│  Developer triggers: "Diagnose database connection failures"    │
└─────────────────────────────┬───────────────────────────────────┘
                              │ POST /diagnose
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AUTONOMY ZONE                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    MAIN POD                                │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │  │
│  │  │  FastAPI    │───▶│ Orchestrator │───▶│  1Password   │  │  │
│  │  │  Server     │    │    Agent     │    │  MCP Server  │  │  │
│  │  └─────────────┘    └──────────────┘    └──────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               RUNNER PODS (parallel)                       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │ DB Agent │  │Cloud Agt │  │ K8s Agt  │  │ Log Agt  │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

1. **Deploy the zone**:
   ```bash
   cd autonomy/examples/sre-diagnose
   autonomy zone deploy
   ```

2. **Start the app**:
   ```bash
   autonomy --rm
   ```

3. **Access the API**:
   - HTTP API: `http://localhost:32100`
   - Logs: `http://localhost:32101`

## API Endpoints

### POST /diagnose

Start a new diagnostic session.

**Request:**
```json
{
  "problem": "Database connection failures in production",
  "environment": "prod",
  "context": {
    "service": "api-gateway",
    "region": "us-west-2"
  }
}
```

**Response:**
```json
{
  "session_id": "abc123",
  "status": "analyzing",
  "message": "Diagnosis started"
}
```

### POST /approve/{session_id}

Approve or deny credential access request.

**Request:**
```json
{
  "approved": true
}
```

### GET /status/{session_id}

Get the current status of a diagnostic session.

**Response:**
```json
{
  "session_id": "abc123",
  "status": "diagnosing",
  "phase": "running_diagnostics",
  "progress": 0.65,
  "findings": []
}
```

## Credential Security

This app demonstrates a secure credential flow where:

1. **Agent requests credential references** (e.g., `op://Infrastructure/prod-db`)
2. **Human approves** the reference (not the actual credential)
3. **1Password retrieves** the actual credential
4. **Credential injected** into tool environment variables
5. **Tools execute** with credentials (credentials never in LLM conversation)
6. **Only results** returned to the LLM

## Development

### Project Structure

```
sre-diagnose/
├── autonomy.yaml          # Zone configuration
├── README.md              # This file
└── images/
    └── main/
        ├── Dockerfile     # Container image
        ├── main.py        # FastAPI server + agents
        └── index.html     # Status dashboard
```

### Testing

```bash
# Health check
curl http://localhost:32100/health

# Start diagnosis
curl -X POST http://localhost:32100/diagnose \
  -H "Content-Type: application/json" \
  -d '{"problem": "Database connection timeout", "environment": "prod"}'

# Check status
curl http://localhost:32100/status/{session_id}
```

## References

- [Autonomy Documentation](https://autonomy.computer/docs)
- [Architecture Document](../../.scratch/sre-incident-diagnosis/ARCHITECTURE.md)
- [Implementation Plan](../../.scratch/sre-incident-diagnosis/IMPLEMENTATION_PLAN.md)
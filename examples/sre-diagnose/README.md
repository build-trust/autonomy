# SRE Incident Diagnosis

An Autonomy app that enables developers to diagnose infrastructure problems directly from Cursor IDE using autonomous diagnostic agents with secure credential retrieval via human-in-the-loop approval.

## Overview

This app demonstrates:

- **Orchestrator Agent**: Analyzes incident descriptions and coordinates diagnosis
- **Human-in-the-Loop Approval**: Requests permission before accessing credentials
- **Secure Credential Flow**: Credentials never enter the LLM conversation
- **Cursor Hooks Integration**: Deep integration with Cursor IDE for seamless diagnosis
- **Real-time Streaming**: Live updates as diagnosis progresses

## Quick Start

### 1. Deploy the Zone

```bash
cd autonomy/examples/sre-diagnose
autonomy zone deploy
```

The app will be available at:
- **Dashboard**: https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer
- **API**: https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/health

### 2. Use from Cursor IDE

Copy the `.cursor` folder to your project or home directory:

```bash
# Project-level (for a specific project)
cp -r .cursor /path/to/your/project/

# Or user-level (global, for all projects)
cp -r .cursor ~/.cursor
```

Restart Cursor to load the hooks.

### 3. Trigger Diagnosis

In Cursor, you can now ask the agent to diagnose infrastructure issues:

```
"Can you diagnose why our database connections are failing in production?"
```

The Cursor hook will intercept the diagnosis request and guide the agent to use the SRE Diagnose API.

## Cursor Hooks Integration

This app includes Cursor hooks that provide deep integration with the diagnosis workflow.

### Hook Configuration

The `.cursor/hooks.json` file defines three hooks:

| Hook | Purpose |
|------|---------|
| `sessionStart` | Sets up SRE Diagnose environment variables and context |
| `beforeShellExecution` | Intercepts `sre-diagnose` commands and routes to the API |
| `stop` | Logs session activity for auditing |

### Available Commands

Once hooks are installed, you can use these commands in Cursor:

```bash
# Diagnose an infrastructure issue
sre-diagnose "Database connection failures in production"

# Alternative command
diagnose-infra "High latency on API gateway"
```

The hook will intercept these commands and guide the agent to use the SRE Diagnose API.

### Environment Variables

The `sessionStart` hook sets these environment variables:

| Variable | Description |
|----------|-------------|
| `SRE_DIAGNOSE_URL` | URL of the SRE Diagnose API |
| `SRE_SESSION_ID` | Cursor conversation ID for correlation |

You can override `SRE_DIAGNOSE_URL` in your environment to point to a different instance.

### Audit Logs

Session activity is logged to `/tmp/sre-diagnose/`:

- `sessions.log` - Session start events
- `commands.log` - Intercepted commands
- `audit.log` - All hook events

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CURSOR IDE                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Cursor Hooks (.cursor/hooks.json)                        │  │
│  │  • sessionStart: Set up SRE context                       │  │
│  │  • beforeShellExecution: Intercept diagnose commands      │  │
│  │  • stop: Audit logging                                    │  │
│  └─────────────────────────┬─────────────────────────────────┘  │
│                            │                                    │
│  Developer: "Diagnose database connection failures"             │
└────────────────────────────┬────────────────────────────────────┘
                             │ POST /diagnose
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AUTONOMY ZONE                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    MAIN POD                               │  │
│  │  ┌─────────────┐    ┌──────────────┐                      │  │
│  │  │  FastAPI    │───▶│ Orchestrator │                      │  │
│  │  │  Server     │    │    Agent     │                      │  │
│  │  │             │    │              │                      │  │
│  │  │  /diagnose  │    │ ask_user_for │                      │  │
│  │  │  /approve   │◀───│ _input       │                      │  │
│  │  │  /status    │    │              │                      │  │
│  │  └─────────────┘    └──────────────┘                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## API Reference

### POST /diagnose

Start a new diagnostic session with streaming response.

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

**Streaming Response (NDJSON):**
```json
{"type": "session_start", "session_id": "abc123", "status": "analyzing"}
{"type": "text", "text": "## Incident Classification\n..."}
{"type": "approval_required", "session_id": "abc123", "status": "waiting_for_approval"}
```

### POST /approve/{session_id}

Approve or deny credential access request.

**Request:**
```json
{
  "approved": true,
  "message": "Approved for read-only database access"
}
```

**Streaming Response (if approved):**
```json
{"type": "approval_accepted", "session_id": "abc123", "status": "approved"}
{"type": "text", "text": "Proceeding with diagnosis..."}
{"type": "diagnosis_complete", "session_id": "abc123", "status": "completed"}
```

### GET /status/{session_id}

Get the current status of a diagnostic session.

**Response:**
```json
{
  "session_id": "abc123",
  "status": "waiting_for_approval",
  "phase": "waiting_for_approval",
  "created_at": "2024-01-15T10:30:00Z",
  "analysis": "## Incident Classification\n...",
  "approval_prompt": "To proceed with diagnosis, I need access to..."
}
```

### GET /sessions

List all diagnostic sessions.

### DELETE /sessions/{session_id}

Delete a session and clean up resources.

### GET /health

Health check endpoint.

## Workflow

1. **User describes problem** in Cursor or via API
2. **Orchestrator Agent analyzes** the incident and identifies:
   - Type of incident (database, network, cloud, etc.)
   - Potential root causes
   - Required credentials for investigation
3. **Agent requests approval** using `ask_user_for_input`
4. **User approves/denies** via API or dashboard
5. **If approved**, agent proceeds with deeper diagnosis
6. **Results returned** to the user

## Testing

```bash
# Health check
curl https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/health

# Start diagnosis (with streaming)
curl -X POST https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/diagnose \
  -H "Content-Type: application/json" \
  -d '{"problem": "Database connection timeout", "environment": "prod"}'

# Check status
curl https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/status/{session_id}

# Approve credential access
curl -X POST https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/approve/{session_id} \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```

## Project Structure

```
sre-diagnose/
├── autonomy.yaml              # Zone configuration
├── README.md                  # This file
├── .cursor/
│   ├── hooks.json             # Cursor hooks configuration
│   └── hooks/
│       ├── session-init.sh    # Session initialization hook
│       ├── sre-diagnose.sh    # Diagnose command interceptor
│       └── audit.sh           # Audit logging hook
└── images/
    └── main/
        ├── Dockerfile         # Container image
        ├── main.py            # FastAPI server + agents
        └── index.html         # Status dashboard
```

## Troubleshooting

### Hooks not working

1. Ensure hooks are in the correct location:
   - Project: `<project>/.cursor/hooks.json`
   - User: `~/.cursor/hooks.json`

2. Check scripts are executable:
   ```bash
   chmod +x .cursor/hooks/*.sh
   ```

3. Restart Cursor to reload hooks

4. Check the Hooks output channel in Cursor (View → Output → Hooks)

### API not responding

1. Check zone status:
   ```bash
   autonomy zone status
   ```

2. Check health endpoint:
   ```bash
   curl https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer/health
   ```

3. Redeploy if needed:
   ```bash
   autonomy zone deploy
   ```

## References

- [Autonomy Documentation](https://autonomy.computer/docs)
- [Cursor Hooks Documentation](https://cursor.com/docs/agent/hooks.md)
- [Architecture Document](planning/ARCHITECTURE.md)
- [Implementation Plan](planning/IMPLEMENTATION_PLAN.md)
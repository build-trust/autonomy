---
name: oncall
description: Control the OnCall Agent demo - activate monitoring, check investigation status, approve/deny credentials, simulate anomalies, and manage incident response. Use when the user mentions oncall, incidents, monitoring, production issues, anomalies, or wants to interact with the demo.
---

# OnCall Agent Demo

This skill helps you interact with the OnCall Agent - an AI-powered incident response system that monitors infrastructure, detects anomalies, and orchestrates remediation with secure credential access via 1Password.

## When to Use

- User wants to activate or start the OnCall agent
- User asks about production status, incidents, or what's happening
- User wants to check investigation progress or root cause
- User needs to approve or deny credential requests
- User wants to simulate an anomaly for demo purposes
- User wants to reset the demo

## Environment

The OnCall demo is deployed at:
- **Dashboard URL**: https://a9eb812238f753132652ae09963a05e9-oncall.cluster.autonomy.computer/
- **API Base**: https://a9eb812238f753132652ae09963a05e9-oncall.cluster.autonomy.computer

Set `ONCALL_URL` environment variable to override.

## Demo Flow

1. **Inactive** → Activate the agent
2. **Activation Approval** → Approve READ credentials (5 credentials from 1Password)
3. **Monitoring** → Agent watches infrastructure; simulate anomaly for demo
4. **Incident Detected** → 150 diagnostic agents deploy across 3 regions
5. **Investigating** → Agents analyze services in parallel
6. **Diagnosis Complete** → Root cause identified
7. **Write Approval #1** → "Kill 3 runaway queries" - typically DENIED
8. **Write Approval #2** → "Disable feature flag" - typically APPROVED
9. **Remediating** → 30-45 second remediation with progress
10. **Resolved** → Incident closed, audit trail available
11. **Reset** → Return to Inactive state

## Scripts

Use these scripts to interact with the OnCall API:

### Check Status
Run `scripts/status.sh` to get the current agent status and any pending approvals.

### Activate Agent
Run `scripts/activate.sh` to start the OnCall agent (requests READ credentials).

### Approve Credentials
Run `scripts/approve.sh` to approve pending credential requests.

### Deny Credentials
Run `scripts/deny.sh` to deny pending credential requests.

### Simulate Anomaly
Run `scripts/simulate.sh` to trigger a cascading failure anomaly (demo only).

### Reset Demo
Run `scripts/reset.sh` to reset the demo to the inactive state.

## API Reference

### GET /oncall/status
Returns current agent state including:
- `status`: Current state (inactive, monitoring, diagnosing, etc.)
- `waiting_for_approval`: Boolean indicating if approval is needed
- `approval`: Details about pending credential requests
- `agent`: Agent-specific data (diagnosis results, remediation progress)

### POST /oncall/activate
Starts the OnCall agent. Returns status `waiting_for_activation_approval`.

### POST /oncall/approve
Body: `{"approved": true}` or `{"approved": false}`
Approves or denies pending credential requests.

### POST /oncall/simulate-anomaly
Triggers a cascading failure anomaly for demo purposes. Only works when status is `monitoring`.

### POST /oncall/reset
Resets the demo to inactive state.

### GET /graph
Returns the agent visualization graph with nodes and edges.

## Interpreting Status

| Status | Meaning |
|--------|---------|
| `inactive` | Agent not running, ready to activate |
| `waiting_for_activation_approval` | Needs READ credential approval |
| `activating` | Retrieving credentials |
| `monitoring` | Actively watching infrastructure |
| `incident_detected` | Anomaly found, deploying agents |
| `diagnosing` | Investigation in progress |
| `diagnosis_complete` | Root cause identified |
| `waiting_for_write_approval` | Needs WRITE credential approval for remediation |
| `remediating` | Executing approved fix |
| `resolved` | Incident closed successfully |

## Tips

- Always check status before taking actions
- The dashboard provides real-time visualization of agents
- Watch the graph fill with gold dots (deploying) turning green (complete)
- The first WRITE approval (kill queries) is designed to be denied
- The second WRITE approval (disable feature flag) is designed to be approved
- Remediation takes 30-45 seconds with visible progress
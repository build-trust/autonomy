#!/bin/bash
# session-init.sh - Initialize SRE Diagnose session context
#
# This hook runs when a new Cursor conversation starts.
# It sets up environment variables and context for SRE diagnosis.

set -euo pipefail

# Read JSON input from stdin
json_input=$(cat)

# Extract conversation info
conversation_id=$(echo "$json_input" | jq -r '.conversation_id // empty')
workspace_roots=$(echo "$json_input" | jq -r '.workspace_roots[0] // empty')

# Log session start
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p /tmp/sre-diagnose
echo "[$timestamp] Session started: $conversation_id in $workspace_roots" >> /tmp/sre-diagnose/sessions.log

# SRE Diagnose endpoint (configure via environment or use default)
SRE_DIAGNOSE_URL="${SRE_DIAGNOSE_URL:-https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer}"

# Output JSON response with environment variables and additional context
cat << EOF
{
  "environmentVariables": {
    "SRE_DIAGNOSE_URL": "$SRE_DIAGNOSE_URL",
    "SRE_SESSION_ID": "$conversation_id"
  },
  "context": "You have access to SRE Incident Diagnosis via the Autonomy platform at $SRE_DIAGNOSE_URL.\n\nTo diagnose infrastructure issues:\n1. POST /diagnose - Start a diagnosis with {\"problem\": \"description\", \"environment\": \"prod\"}\n2. GET /status/{session_id} - Check diagnosis status\n3. POST /approve/{session_id} - Approve credential access with {\"approved\": true}\n\nThe diagnosis uses a two-phase approach: first analyzing the problem, then requesting approval before accessing any credentials."
}
EOF

#!/bin/bash
# session-init.sh - Initialize SRE Diagnose session context
#
# This hook runs when a new Cursor conversation starts.
# It sets up environment variables and context for SRE diagnosis.

# Read JSON input from stdin
json_input=$(cat)

# Extract conversation info
conversation_id=$(echo "$json_input" | jq -r '.conversation_id // empty')
workspace_roots=$(echo "$json_input" | jq -r '.workspace_roots[0] // empty')

# Log session start
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p /tmp/sre-diagnose
echo "[$timestamp] Session started: $conversation_id" >> /tmp/sre-diagnose/sessions.log

# SRE Diagnose endpoint (configure via environment or use default)
SRE_DIAGNOSE_URL="${SRE_DIAGNOSE_URL:-https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer}"

# Output JSON response with environment variables and additional context
cat << EOF
{
  "env": {
    "SRE_DIAGNOSE_URL": "$SRE_DIAGNOSE_URL",
    "SRE_SESSION_ID": "$conversation_id"
  },
  "additional_context": "You have access to SRE Incident Diagnosis via the Autonomy platform. To diagnose infrastructure issues, you can use the 'sre-diagnose' command or call the SRE Diagnose API at $SRE_DIAGNOSE_URL. The API provides: POST /diagnose to start diagnosis, GET /status/{session_id} to check status, POST /approve/{session_id} to approve/deny credential access.",
  "continue": true
}
EOF

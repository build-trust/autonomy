#!/bin/bash
# sre-diagnose.sh - Intercept SRE diagnosis commands and route to Autonomy
#
# This hook intercepts terminal commands matching SRE diagnosis patterns
# and provides guidance to use the SRE Diagnose API instead.

set -euo pipefail

# Read JSON input from stdin
json_input=$(cat)

# Extract tool call details
tool_name=$(echo "$json_input" | jq -r '.tool_name // empty')
tool_input=$(echo "$json_input" | jq -r '.tool_input // empty')
command=$(echo "$tool_input" | jq -r '.command // empty')

# Log the command
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p /tmp/sre-diagnose
echo "[$timestamp] Tool: $tool_name, Command: $command" >> /tmp/sre-diagnose/commands.log

# SRE Diagnose endpoint
SRE_DIAGNOSE_URL="${SRE_DIAGNOSE_URL:-https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer}"

# Check if this is a diagnose command
if [[ "$command" =~ sre-diagnose[[:space:]]+(.*) ]] || [[ "$command" =~ diagnose-infra[[:space:]]+(.*) ]]; then
    # Extract the problem description from the command
    problem="${BASH_REMATCH[1]:-}"

    if [[ -z "$problem" ]]; then
        # No problem specified
        cat << EOF
{
  "decision": "block",
  "reason": "Missing problem description. Usage: sre-diagnose <problem description>"
}
EOF
        exit 0
    fi

    # Block the shell command and provide API guidance
    cat << EOF
{
  "decision": "block",
  "reason": "Use the SRE Diagnose API instead of shell commands for infrastructure diagnosis.",
  "instructions": "To diagnose '$problem', make these API calls:\n\n1. Start diagnosis:\n   curl -X POST '$SRE_DIAGNOSE_URL/diagnose' -H 'Content-Type: application/json' -d '{\"problem\": \"$problem\", \"environment\": \"prod\"}'\n\n2. The response will stream analysis results and request approval for credentials.\n\n3. Approve credential access:\n   curl -X POST '$SRE_DIAGNOSE_URL/approve/{session_id}' -H 'Content-Type: application/json' -d '{\"approved\": true}'\n\n4. View dashboard: $SRE_DIAGNOSE_URL"
}
EOF
    exit 0
fi

# For other commands, allow them to proceed
cat << EOF
{
  "decision": "allow"
}
EOF

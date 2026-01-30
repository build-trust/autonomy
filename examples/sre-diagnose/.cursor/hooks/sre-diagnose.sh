#!/bin/bash
# sre-diagnose.sh - Intercept SRE diagnosis commands and route to Autonomy
#
# This hook intercepts commands matching 'sre-diagnose' or 'diagnose-infra'
# and routes them to the SRE Diagnose Autonomy app.

set -e

# Read JSON input from stdin
json_input=$(cat)

# Extract command details
command=$(echo "$json_input" | jq -r '.command // empty')
cwd=$(echo "$json_input" | jq -r '.cwd // empty')
conversation_id=$(echo "$json_input" | jq -r '.conversation_id // empty')

# Log the command
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p /tmp/sre-diagnose
echo "[$timestamp] Command intercepted: $command" >> /tmp/sre-diagnose/commands.log

# SRE Diagnose endpoint
SRE_DIAGNOSE_URL="${SRE_DIAGNOSE_URL:-https://a9eb812238f753132652ae09963a05e9-srediag.cluster.autonomy.computer}"

# Check if this is a diagnose command
if [[ "$command" =~ sre-diagnose[[:space:]]+(.*) ]] || [[ "$command" =~ diagnose-infra[[:space:]]+(.*) ]]; then
    # Extract the problem description from the command
    problem="${BASH_REMATCH[1]}"

    if [[ -z "$problem" ]]; then
        # No problem specified, ask user for clarification
        cat << EOF
{
  "continue": true,
  "permission": "deny",
  "user_message": "Please specify a problem to diagnose. Usage: sre-diagnose <problem description>",
  "agent_message": "The sre-diagnose command requires a problem description. Please ask the user what infrastructure issue they want to diagnose, then call: sre-diagnose <problem description>"
}
EOF
        exit 0
    fi

    # Block the shell command and provide guidance to use the API instead
    cat << EOF
{
  "continue": true,
  "permission": "deny",
  "user_message": "🔧 SRE Diagnosis initiated for: $problem",
  "agent_message": "Instead of running a shell command, use the SRE Diagnose API to analyze this infrastructure issue.

To diagnose '$problem':

1. **Start diagnosis** by calling:
   curl -X POST '$SRE_DIAGNOSE_URL/diagnose' \\
     -H 'Content-Type: application/json' \\
     -d '{\"problem\": \"$problem\", \"environment\": \"prod\"}'

2. **Check the streaming response** for the session_id and analysis.

3. **If approval is required**, the response will indicate 'waiting_for_approval'.
   To approve credential access:
   curl -X POST '$SRE_DIAGNOSE_URL/approve/{session_id}' \\
     -H 'Content-Type: application/json' \\
     -d '{\"approved\": true}'

4. **Check status** at any time:
   curl '$SRE_DIAGNOSE_URL/status/{session_id}'

5. **View the dashboard** at: $SRE_DIAGNOSE_URL

The diagnosis agent will analyze the problem, identify potential root causes, and request approval before accessing any credentials or systems."
}
EOF
    exit 0
fi

# For other matched commands, allow them to proceed
cat << EOF
{
  "continue": true,
  "permission": "allow"
}
EOF

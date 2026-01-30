#!/bin/bash
# audit.sh - Audit hook for SRE Diagnose sessions
#
# This hook runs when a Cursor task completes.
# It logs session activity for auditing and analytics.

# Read JSON input from stdin
json_input=$(cat)

# Extract session details
conversation_id=$(echo "$json_input" | jq -r '.conversation_id // empty')
generation_id=$(echo "$json_input" | jq -r '.generation_id // empty')
status=$(echo "$json_input" | jq -r '.status // empty')
hook_event_name=$(echo "$json_input" | jq -r '.hook_event_name // empty')

# Create timestamp
timestamp=$(date '+%Y-%m-%d %H:%M:%S')

# Ensure log directory exists
mkdir -p /tmp/sre-diagnose

# Write audit log entry
cat << EOF >> /tmp/sre-diagnose/audit.log
[$timestamp] event=$hook_event_name conversation=$conversation_id generation=$generation_id status=$status
EOF

# Output empty JSON (fire-and-forget hook)
echo '{}'

exit 0

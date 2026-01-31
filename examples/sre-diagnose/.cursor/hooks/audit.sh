#!/bin/bash
# audit.sh - Audit hook for SRE Diagnose sessions
#
# This hook runs after tool calls complete.
# It logs activity for auditing and analytics.

set -euo pipefail

# Read JSON input from stdin
json_input=$(cat)

# Extract details from the tool call result
tool_name=$(echo "$json_input" | jq -r '.tool_name // "unknown"')
tool_result=$(echo "$json_input" | jq -r '.tool_result // empty' | head -c 200)
exit_code=$(echo "$json_input" | jq -r '.exit_code // "N/A"')

# Create timestamp
timestamp=$(date '+%Y-%m-%d %H:%M:%S')

# Ensure log directory exists
mkdir -p /tmp/sre-diagnose

# Write audit log entry
cat << EOF >> /tmp/sre-diagnose/audit.log
[$timestamp] tool=$tool_name exit_code=$exit_code result_preview=${tool_result:0:100}
EOF

# Output empty JSON (no modifications needed for post-tool-call)
echo '{}'

exit 0

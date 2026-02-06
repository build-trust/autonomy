#!/bin/bash
# deny.sh - Deny pending credential requests
set -euo pipefail

ONCALL_URL="${ONCALL_URL:-https://a9eb812238f753132652ae09963a05e9-oncall.cluster.autonomy.computer}"

echo "=== Denying Credential Request ==="
echo ""

# Check current status first
status=$(curl -s --max-time 5 "$ONCALL_URL/oncall/status" | jq -r '.status // "unknown"')

if [[ "$status" != "waiting_for_activation_approval" && "$status" != "waiting_for_write_approval" ]]; then
    echo "Error: No pending approval request."
    echo "Current status: $status"
    exit 1
fi

# Deny the request
response=$(curl -s --max-time 10 -X POST "$ONCALL_URL/oncall/approve" \
    -H "Content-Type: application/json" \
    -d '{"approved": false}')

new_status=$(echo "$response" | jq -r '.status // "unknown"')
message=$(echo "$response" | jq -r '.message // ""')

echo "Result: DENIED"
echo "New status: $new_status"
if [[ -n "$message" ]]; then
    echo "Message: $message"
fi

echo ""
echo "Dashboard: $ONCALL_URL"

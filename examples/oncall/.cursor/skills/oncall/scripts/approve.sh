#!/bin/bash
# approve.sh - Approve pending credential requests
set -euo pipefail

ONCALL_URL="${ONCALL_URL:-https://a9eb812238f753132652ae09963a05e9-oncall.cluster.autonomy.computer}"

echo "=== Approving Credential Request ==="
echo ""

# Check current status first
status=$(curl -s --max-time 5 "$ONCALL_URL/oncall/status" | jq -r '.status // "unknown"')

if [[ "$status" != "waiting_for_activation_approval" && "$status" != "waiting_for_write_approval" ]]; then
    echo "No pending approval. Current status: $status"
    exit 0
fi

# Send approval
response=$(curl -s --max-time 10 -X POST \
    -H "Content-Type: application/json" \
    -d '{"approved": true}' \
    "$ONCALL_URL/oncall/approve")

new_status=$(echo "$response" | jq -r '.status // "unknown"')
message=$(echo "$response" | jq -r '.message // ""')

echo "Approved!"
echo ""
echo "New status: $new_status"
if [[ -n "$message" ]]; then
    echo "Message: $message"
fi

echo ""
echo "Dashboard: $ONCALL_URL"

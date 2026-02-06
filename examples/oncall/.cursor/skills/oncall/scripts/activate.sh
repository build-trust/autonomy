#!/bin/bash
# activate.sh - Activate the OnCall agent
set -euo pipefail

ONCALL_URL="${ONCALL_URL:-https://a9eb812238f753132652ae09963a05e9-oncall.cluster.autonomy.computer}"

echo "=== Activating OnCall Agent ==="
echo ""

# Check current status first
current_status=$(curl -s --max-time 5 "$ONCALL_URL/oncall/status" | jq -r '.status // "unknown"')

if [[ "$current_status" == "monitoring" ]]; then
    echo "Agent is already active and monitoring."
    echo "Use simulate.sh to trigger a demo anomaly."
    exit 0
fi

if [[ "$current_status" == "waiting_for_activation_approval" ]]; then
    echo "Agent is already waiting for READ credential approval."
    echo "Use approve.sh to grant access or deny.sh to reject."
    exit 0
fi

if [[ "$current_status" != "inactive" ]]; then
    echo "Agent is currently in '$current_status' state."
    echo "Use reset.sh first to return to inactive state."
    exit 1
fi

# Activate the agent
response=$(curl -s --max-time 10 -X POST "$ONCALL_URL/oncall/activate")
new_status=$(echo "$response" | jq -r '.status // "error"')

if [[ "$new_status" == "waiting_for_activation_approval" ]]; then
    echo "✅ Activation requested!"
    echo ""
    echo "The agent now needs READ credential approval to start monitoring."
    echo ""
    echo "Requested credentials:"
    curl -s --max-time 5 "$ONCALL_URL/oncall/status" | jq -r '.approval.requested_credentials[]?' 2>/dev/null | sed 's/^/  - /'
    echo ""
    echo "Use approve.sh to grant access or deny.sh to reject."
else
    echo "Activation response: $(echo "$response" | jq -r '.message // .error // "Unknown"')"
fi

echo ""
echo "Dashboard: $ONCALL_URL"

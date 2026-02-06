#!/bin/bash
# reset.sh - Reset the OnCall demo to inactive state
set -euo pipefail

ONCALL_URL="${ONCALL_URL:-https://a9eb812238f753132652ae09963a05e9-oncall.cluster.autonomy.computer}"

echo "Resetting OnCall demo..."

response=$(curl -s --max-time 10 -X POST "$ONCALL_URL/oncall/reset")

status=$(echo "$response" | jq -r '.status // "unknown"')
message=$(echo "$response" | jq -r '.message // ""')

if [[ "$status" == "inactive" ]] || [[ "$message" == *"reset"* ]]; then
    echo "✅ Demo reset successfully"
    echo ""
    echo "Status: inactive"
    echo "The OnCall agent is ready to be activated again."
    echo ""
    echo "Use activate.sh to start a new demo session."
else
    echo "Reset response:"
    echo "$response" | jq .
fi

echo ""
echo "Dashboard: $ONCALL_URL"

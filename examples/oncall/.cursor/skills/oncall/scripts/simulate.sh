#!/bin/bash
# simulate.sh - Trigger a demo anomaly
set -euo pipefail

ONCALL_URL="${ONCALL_URL:-https://a9eb812238f753132652ae09963a05e9-oncall.cluster.autonomy.computer}"

echo "=== Simulating Anomaly ==="
echo ""

# Check current status first
status=$(curl -s --max-time 5 "$ONCALL_URL/oncall/status" | jq -r '.status // "unknown"')

if [[ "$status" != "monitoring" ]]; then
    echo "Error: Agent must be in 'monitoring' state to simulate an anomaly."
    echo "Current status: $status"
    echo ""
    if [[ "$status" == "inactive" ]]; then
        echo "Use activate.sh first to start the agent."
    elif [[ "$status" == "waiting_for_activation_approval" ]]; then
        echo "Use approve.sh to approve READ credentials first."
    else
        echo "Use reset.sh to reset the demo, then activate again."
    fi
    exit 1
fi

response=$(curl -s --max-time 10 -X POST "$ONCALL_URL/oncall/simulate-anomaly")

new_status=$(echo "$response" | jq -r '.status // "unknown"')
message=$(echo "$response" | jq -r '.message // ""')

if [[ "$new_status" == "incident_detected" || "$new_status" == "diagnosing" ]]; then
    echo "✓ Anomaly triggered successfully!"
    echo ""
    echo "Status: $new_status"
    echo ""
    echo "The OnCall agent is now deploying 150 diagnostic agents across 3 regions."
    echo "Watch the dashboard to see the investigation unfold."
    echo ""
    echo "Use status.sh to monitor progress."
else
    echo "Response: $message"
    echo "Status: $new_status"
fi

echo ""
echo "Dashboard: $ONCALL_URL"

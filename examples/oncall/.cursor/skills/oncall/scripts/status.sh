#!/bin/bash
# status.sh - Check OnCall agent status
set -euo pipefail

ONCALL_URL="${ONCALL_URL:-https://a9eb812238f753132652ae09963a05e9-oncall.cluster.autonomy.computer}"

response=$(curl -s --max-time 10 "$ONCALL_URL/oncall/status")

echo "=== OnCall Agent Status ==="
echo ""

status=$(echo "$response" | jq -r '.status // "unknown"')
waiting=$(echo "$response" | jq -r '.waiting_for_approval // false')

echo "Status: $status"
echo "Waiting for approval: $waiting"

case "$status" in
    "inactive")
        echo ""
        echo "The agent is not running. Use activate.sh to start it."
        ;;
    "waiting_for_activation_approval")
        echo ""
        echo "Pending READ credential approval:"
        echo "$response" | jq -r '.approval.requested_credentials[]?' 2>/dev/null | sed 's/^/  - /'
        echo ""
        echo "Use approve.sh to grant access or deny.sh to reject."
        ;;
    "monitoring")
        echo ""
        echo "Agent is actively monitoring infrastructure."
        echo "Use simulate.sh to trigger a demo anomaly."
        ;;
    "incident_detected"|"diagnosing")
        agents=$(echo "$response" | jq -r '.agent.agents_deployed // "unknown"')
        anomaly=$(echo "$response" | jq -r '.agent.anomaly_type // "unknown"')
        echo ""
        echo "Anomaly type: $anomaly"
        echo "Diagnostic agents deployed: $agents"
        echo ""
        echo "Investigation in progress..."
        ;;
    "diagnosis_complete")
        echo ""
        echo "Root cause identified:"
        echo "$response" | jq -r '.agent.diagnosis_result // "See dashboard"' | head -20
        ;;
    "waiting_for_write_approval")
        echo ""
        echo "Pending WRITE credential approval:"
        echo "$response" | jq -r '.approval.requested_credentials[]?' 2>/dev/null | sed 's/^/  - /'
        echo ""
        echo "Pending action:"
        echo "$response" | jq -r '.approval.pending_action.description // "remediation"'
        echo ""
        echo "Use approve.sh to execute or deny.sh to skip."
        ;;
    "remediating")
        progress=$(echo "$response" | jq -r '.agent.remediation_progress // 0')
        message=$(echo "$response" | jq -r '.agent.remediation_message // "In progress"')
        error_rate=$(echo "$response" | jq -r '.agent.error_rate_current // "unknown"')
        echo ""
        echo "Progress: $progress%"
        echo "Status: $message"
        echo "Error rate: $error_rate%"
        ;;
    "resolved")
        echo ""
        echo "Incident resolved. Use reset.sh to start fresh."
        ;;
esac

echo ""
echo "Dashboard: $ONCALL_URL"

#!/bin/bash

set -e

echo "Testing Legal Assistant Example..."

# Wait for service to be ready
echo "Waiting for service to start..."
sleep 30

# Test that the agent endpoint responds
echo "Testing agent endpoint..."
response=$(curl -s -X POST http://localhost:32100/agents/henry \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello, what can you help me with?"}')

if echo "$response" | grep -q "legal\|assistant\|help"; then
    echo "✓ Agent responds appropriately"
else
    echo "✗ Agent response unexpected: $response"
    exit 1
fi

# Test a legal question
echo "Testing legal knowledge..."
response=$(curl -s -X POST http://localhost:32100/agents/henry \
    -H "Content-Type: application/json" \
    -d '{"message": "What is Section 330 about?"}')

if echo "$response" | grep -q -i "weather\|modification\|reporting"; then
    echo "✓ Agent has legal knowledge loaded"
else
    echo "✗ Agent lacks expected legal knowledge: $response"
    exit 1
fi

echo "All tests passed! ✓"

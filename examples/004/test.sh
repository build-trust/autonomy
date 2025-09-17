#!/bin/bash

# Test script for the translation API

set -e

# Check if secrets.yaml exists
if [ ! -f "secrets.yaml" ]; then
    echo "Error: secrets.yaml not found. Please create it first using:"
    echo "sed \"s|your_api_key_here|\$(openssl rand -hex 32)|\" secrets.example.yaml > secrets.yaml"
    exit 1
fi

# Extract API key from secrets.yaml
API_KEY=$(grep "API_KEY:" secrets.yaml | cut -d' ' -f2)

if [ -z "$API_KEY" ]; then
    echo "Error: Could not extract API_KEY from secrets.yaml"
    exit 1
fi

# Set up environment variables
export URL='http://localhost:32100'
export API_KEY="$API_KEY"

echo "Testing translation API..."
echo "URL: $URL"
echo "API Key: ${API_KEY:0:8}..."

# Wait for the service to be ready
echo "Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s "$URL/agents" > /dev/null 2>&1; then
        echo "Service is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Error: Service did not start within 30 seconds"
        exit 1
    fi
    sleep 1
done

# Test the translation API
echo ""
echo "Testing translation API with sample data..."
echo ""

curl -s -X POST "$URL/analyses" \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"items":["hello", "goodbye", "thank you", "how are you?"]}' | python3 -m json.tool

echo ""
echo "Test completed!"

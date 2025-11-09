# HITL Integration Tests

This directory contains integration tests for Human-in-the-Loop (HITL) functionality with real AI models.

## Overview

These tests verify HITL pause/resume behavior using actual AWS Bedrock Claude models, ensuring that:
- Agents pause correctly when requesting user input
- State is properly maintained during pauses
- Agents resume correctly after receiving user responses
- Streaming and non-streaming modes both work
- Multiple pause cycles are handled correctly
- Conversation states are isolated

## Prerequisites

### AWS Credentials
You must have AWS credentials configured with access to Amazon Bedrock:

```bash
# Option 1: AWS CLI configuration
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
```

### Required Environment Variables

```bash
# Enable direct Bedrock client
export AUTONOMY_USE_DIRECT_BEDROCK=1

# Use in-memory database (no PostgreSQL required)
export AUTONOMY_USE_IN_MEMORY_DATABASE=1

# Optional: Set AWS region (defaults to us-east-1)
export AWS_REGION="us-east-1"

# Optional: Enable debug logging
export AUTONOMY_LOG_LEVEL="INFO,agent=DEBUG,model=DEBUG"
```

### Python Dependencies
All dependencies are managed via `uv`. The key dependencies are:
- `pytest>=8.3.5`
- `pytest-asyncio>=0.26.0`
- `boto3>=1.38.45` (for AWS Bedrock)

## Running the Tests

### Run All Integration Tests

```bash
cd source/python

# With environment variables
AUTONOMY_USE_DIRECT_BEDROCK=1 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
  uv run --active pytest tests/agents/test_hitl_integration.py -v
```

### Run Specific Test Classes

```bash
# Test basic pause/resume
uv run --active pytest tests/agents/test_hitl_integration.py::TestBasicPauseResume -v

# Test streaming mode
uv run --active pytest tests/agents/test_hitl_integration.py::TestStreamingPauseResume -v

# Test multiple pauses
uv run --active pytest tests/agents/test_hitl_integration.py::TestMultiplePauses -v

# Test edge cases
uv run --active pytest tests/agents/test_hitl_integration.py::TestEdgeCases -v

# Test performance
uv run --active pytest tests/agents/test_hitl_integration.py::TestPerformance -v
```

### Run Specific Test Functions

```bash
# Test a single function
uv run --active pytest tests/agents/test_hitl_integration.py::TestBasicPauseResume::test_agent_pauses_on_ask_user_for_input -v
```

### Skip Integration Tests

By default, integration tests are skipped if `AUTONOMY_USE_DIRECT_BEDROCK` is not set.

To run only unit tests (skip integration tests):

```bash
cd source/python
uv run --active pytest -m "not integration" -v
```

To run only integration tests:

```bash
cd source/python
AUTONOMY_USE_DIRECT_BEDROCK=1 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
  uv run --active pytest -m "integration" -v
```

## Test Structure

### Test Classes

- **TestBasicPauseResume**: Core pause/resume functionality
  - Agent pauses when `ask_user_for_input` is called
  - Agent resumes after user provides response
  - State transitions are correct

- **TestStreamingPauseResume**: Streaming mode specific tests
  - Streaming pauses without timeout
  - Streaming resume sends chunks correctly
  - Part numbering resets properly on resume

- **TestMultiplePauses**: Multiple pause cycles
  - Multiple consecutive pauses work correctly
  - Conversation state isolation between different conversations

- **TestEdgeCases**: Edge case handling
  - Empty user responses
  - Conversation state accuracy
  - Various error conditions

- **TestPerformance**: Performance characteristics
  - Pause response time
  - Resume response time

### Fixtures

- **event_loop**: Module-scoped event loop for async tests
- **test_node**: Shared Node instance for all tests (avoids port conflicts)

## Expected Behavior

### Model Behavior
Real AI models (Claude) may:
- Ask follow-up questions even after receiving answers
- Request clarification if responses are unclear
- Take multiple pause/resume cycles to complete a task

**This is expected behavior** and demonstrates the pause/resume mechanism works correctly across multiple cycles.

### Streaming Behavior
- Streams complete with `finished=True` when paused (not timeout)
- Part numbers (`part_nb`) reset on each resume
- Each resume creates a new streaming context

### State Transitions
1. **Initial Request** → Agent enters `THINKING` state
2. **Tool Call** → Agent calls `ask_user_for_input`
3. **Pause** → State becomes `WAITING_FOR_INPUT`
4. **User Response** → Converted to `ToolCallResponseMessage`
5. **Resume** → Agent returns to `THINKING` state
6. **Completion** → State becomes `DONE`

## Timing Expectations

These tests use real AI models, so timing varies:

- **Basic pause/resume**: 3-10 seconds per cycle
- **Streaming pause/resume**: 5-15 seconds per cycle
- **Multiple pauses**: 10-30 seconds for full scenario
- **Total test suite**: 2-5 minutes

Timeouts are set generously (30 seconds) to account for:
- Model inference time
- Network latency to AWS Bedrock
- Multiple pause cycles

## Debugging Failed Tests

### Enable Debug Logging

```bash
AUTONOMY_LOG_LEVEL="INFO,agent=DEBUG,model=DEBUG" \
AUTONOMY_USE_DIRECT_BEDROCK=1 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
  uv run --active pytest tests/agents/test_hitl_integration.py -v -s
```

### Common Issues

**AWS Credentials Not Configured**
```
Error: Unable to locate credentials
```
Solution: Configure AWS credentials (see Prerequisites)

**Wrong AWS Region**
```
Error: Could not connect to the endpoint URL
```
Solution: Set `AWS_REGION` to a region with Bedrock access (e.g., us-east-1)

**Port Already in Use**
```
OSError: [Errno 48] address already in use
```
Solution: The `test_node` fixture should prevent this, but if it occurs, kill any running Autonomy processes:
```bash
lsof -ti:8000 | xargs kill -9
```

**Test Timeout**
```
asyncio.TimeoutError
```
This can happen if:
- Network is slow
- Model is taking unusually long
- Bug in pause/resume logic

Check logs for details. The 30-second timeout should be sufficient for normal operation.

**Model Asks Too Many Follow-up Questions**
Some tests allow for multiple pause cycles (up to 3). If a test fails because the model asks more than 3 follow-up questions, this might indicate:
- Instructions need to be more specific
- Model behavior has changed
- Test should allow more cycles

## CI/CD Integration

### Skip in CI by Default

Since integration tests require AWS credentials, they should be skipped by default in CI:

```yaml
# .github/workflows/test.yml
- name: Run unit tests
  run: |
    cd source/python
    uv run --active pytest -m "not integration" -v
```

### Run in CI with Secrets

To run integration tests in CI:

```yaml
# .github/workflows/integration-test.yml
- name: Run integration tests
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    AWS_REGION: us-east-1
    AUTONOMY_USE_DIRECT_BEDROCK: "1"
    AUTONOMY_USE_IN_MEMORY_DATABASE: "1"
  run: |
    cd source/python
    uv run --active pytest -m "integration" -v
```

## Maintenance

### Adding New Tests

1. Add test function to appropriate class
2. Mark with `@pytest.mark.asyncio`
3. Use the `test_node` fixture
4. Keep instructions brief to minimize model variance
5. Allow for multiple pause cycles if needed
6. Set reasonable timeouts (30s is usually sufficient)

### Updating Model Instructions

When updating agent instructions:
- Keep them brief and specific
- Explicitly tell the agent to use `ask_user_for_input`
- Request brief responses to minimize test time
- Avoid ambiguous phrasing that might confuse the model

### Performance Benchmarking

To benchmark performance:

```bash
# Run with timing
uv run --active pytest tests/agents/test_hitl_integration.py::TestPerformance -v --durations=0
```

## Related Documentation

- **Unit Tests**: `test_hitl_core.py`, `test_hitl_streaming.py`, etc.
- **Manual HTTP Tests**: `autonomy/.scratch/hitl_tests/test_http_manual.sh`
- **Code Review**: `autonomy/.scratch/code_review_agent_hitl.md`
- **Test Results**: `autonomy/.scratch/hitl_test_results.md`
- **Fix Plan**: `autonomy/.scratch/hitl_fix_plan.md`

## Status

**Status**: ✅ ALL TESTS PASSING  
**Last Updated**: 2025-01-09  
**Model Used**: Claude Sonnet 4 v1 (via AWS Bedrock)  
**Coverage**: Basic pause/resume, streaming, multiple pauses, edge cases, performance
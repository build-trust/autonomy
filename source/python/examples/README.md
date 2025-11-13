# Autonomy Examples

This directory contains examples demonstrating various features of the Autonomy framework.

## Running Examples

From inside the `source/python` directory, run:

```bash
AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
CLUSTER="$(autonomy cluster show)" \
uv run --active examples/006.py
```

## Available Examples

### Example 001 - Basic Node
```bash
uv run --active examples/001.py
```
Minimal example showing how to start an Autonomy node.

### Example 002-004
Basic examples demonstrating core functionality.

### Example 005 - Model Chat
```bash
uv run --active examples/005.py
```
Using the Model API directly for chat completions.

### Example 006 - Streaming Chat
```bash
uv run --active examples/006.py
```
Streaming responses from the Model API.

### Example 007 - Agent Basics
```bash
uv run --active examples/007.py
```
Creating and using a basic agent.

### Example 008 - REPL Interface
```bash
AUTONOMY_WAIT_UNTIL_INTERRUPTED=1 AUTONOMY_USE_IN_MEMORY_DATABASE=1 uv run --active examples/008.py
```
Interactive REPL interface with an agent. Connect via: `nc 127.0.0.1 7000`

### Example 009 - Context Logging with Multi-Step Tool Usage
```bash
AUTONOMY_TRANSCRIPTS=1 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
uv run --active examples/009_context_logging.py
```

**Demonstrates:**
- Multi-step agent reasoning
- Multiple tool calls and orchestration
- Transcript logging to see exactly what goes to the model
- Conversation memory across turns

```bash
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
AUTONOMY_TRANSCRIPTS=1 \
CLUSTER="$(autonomy cluster show)" \
uv run --active examples/009_context_logging.py
```

**With raw API payloads:**
```bash
AUTONOMY_TRANSCRIPTS=1 \
AUTONOMY_TRANSCRIPTS_RAW=1 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
uv run --active examples/009_context_logging.py
```

This example shows a travel planning agent that uses multiple tools (weather, flights, hotels) to answer complex queries. Enable transcript logging to see the complete context sent to the model at each step, including system instructions, tool calls, tool results, and conversation history.

### Example 010 - Agent Transcripts
```bash
AUTONOMY_TRANSCRIPTS=1 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
uv run --active examples/010_agent_transcripts.py
```

**Demonstrates:**
- Human-readable transcript logging
- Raw API payload inspection
- Provider-specific transformations
- Multi-turn conversations with context

**With raw API payloads:**
```bash
AUTONOMY_TRANSCRIPTS=1 \
AUTONOMY_TRANSCRIPTS_RAW=1 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
uv run --active examples/010_agent_transcripts.py
```

**Only raw JSON (for piping to jq):**
```bash
AUTONOMY_TRANSCRIPTS_RAW_ONLY=1 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
uv run --active examples/010_agent_transcripts.py 2>/dev/null
```

### Example 011 - Code Review Assistant with Filesystem Tools
```bash
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
AUTONOMY_TRANSCRIPTS=1 \
uv run --active examples/011_code_review_assistant.py
```

**Demonstrates:**
- Complex multi-step task execution
- Filesystem tools for reading and analyzing code
- Creating structured documentation
- Multi-turn conversations with state preservation
- Agent reasoning across multiple files
- Organizing information in different formats

```bash
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
AUTONOMY_TRANSCRIPTS=1 \
CLUSTER="$(autonomy cluster show)" \
uv run --active examples/011_code_review_assistant.py
```

This example shows a code review agent that:
- Analyzes a Python codebase for issues
- Identifies security vulnerabilities and performance problems
- Generates comprehensive review reports
- Creates action item checklists
- Produces executive summaries for stakeholders

The agent uses filesystem tools to explore project structure, read code files, search for patterns, and create multiple review documents in an organized way.

## Transcript Logging

Transcript logging shows what's sent to the model and what comes back, in both human-readable and raw API format.

**Environment Variables:**
```bash
AUTONOMY_TRANSCRIPTS=1          # Enable human-readable transcripts
AUTONOMY_TRANSCRIPTS_RAW=1      # Also show raw API payloads/responses
AUTONOMY_TRANSCRIPTS_RAW_ONLY=1 # Show ONLY raw JSON (no human-readable)
AUTONOMY_TRANSCRIPTS_FILE=/path # Write output to file (appends)
```

**Examples:**
```bash
# Human-readable context and responses
AUTONOMY_TRANSCRIPTS=1 uv run --active examples/010_agent_transcripts.py

# Human-readable + raw API payloads (see transformations)
AUTONOMY_TRANSCRIPTS=1 AUTONOMY_TRANSCRIPTS_RAW=1 uv run --active examples/010_agent_transcripts.py

# Only raw JSON (for piping to jq, etc.)
AUTONOMY_TRANSCRIPTS_RAW_ONLY=1 uv run --active examples/010_agent_transcripts.py 2>/dev/null | jq

# Save to file
AUTONOMY_TRANSCRIPTS=1 AUTONOMY_TRANSCRIPTS_FILE=/tmp/transcript.log uv run --active examples/010_agent_transcripts.py
```

Transcript logging helps you:
- Debug unexpected agent behavior
- Understand what information the model receives
- Inspect raw API payloads and transformations
- Validate context templates
- Optimize context size
- Monitor production systems

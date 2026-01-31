# SRE Incident Diagnosis System - Implementation Plan

## Project Location

**Directory**: `autonomy/examples/sre-diagnose`

This will be added as an example app alongside other examples in the Autonomy repository.

## Development Workflow

### Commit After Every Step

After completing each task:
1. Test that everything works
2. Run `git add -A && git diff --cached --stat | cat`
3. Commit with a descriptive message: `git commit -m "sre-diagnose: <description>"`
4. Note any issues encountered for the next session

### Session Handover Protocol

Each phase ends with a **HANDOVER PROMPT** - copy this to start a new Claude session when:
- Context window is getting full
- Taking a break
- Switching focus

---

## Phase 1: Foundation (MVP)

**Goal**: Basic end-to-end flow - Cursor → Autonomy → simple diagnosis → response

### Task 1.1: Create Project Structure

Create the basic Autonomy app skeleton.

**Files to create:**
```
autonomy/examples/sre-diagnose/
├── autonomy.yaml
├── README.md
└── images/
    └── main/
        ├── Dockerfile
        ├── main.py
        └── index.html
```

**autonomy.yaml:**
```yaml
name: srediag
pods:
  - name: main-pod
    public: true
    containers:
      - name: main
        image: main
```

**Acceptance criteria:**
- [ ] Files created in correct location
- [ ] `autonomy zone deploy` succeeds
- [ ] Can access the zone URL
- [ ] Basic health check works

**Commit:** `git commit -m "sre-diagnose: initial project structure"`

---

### Task 1.2: Basic Orchestrator Agent

Add a simple orchestrator agent that analyzes problems.

**Update main.py to include:**
- FastAPI app with POST /diagnose endpoint
- Basic orchestrator agent (no tools yet)
- Simple problem analysis

**Acceptance criteria:**
- [ ] POST /diagnose returns agent analysis
- [ ] Streaming response works
- [ ] Agent provides sensible analysis of the problem description

**Commit:** `git commit -m "sre-diagnose: add basic orchestrator agent"`

---

### Task 1.3: Human-in-the-Loop Approval Flow

Implement the credential approval mechanism using `ask_user_for_input`.

**Add:**
- Session state management (dict in memory for MVP)
- `enable_ask_for_user_input=True` on agent
- POST /approve/{session_id} endpoint
- Logic to resume agent after approval

**Acceptance criteria:**
- [ ] Agent pauses and requests approval
- [ ] /approve endpoint resumes execution
- [ ] Denial cancels the diagnosis
- [ ] Session state properly tracked

**Commit:** `git commit -m "sre-diagnose: implement human-in-the-loop approval flow"`

---

### PHASE 1 HANDOVER PROMPT

```
Continue building the SRE Incident Diagnosis Autonomy app.

## Context
- Project location: autonomy/examples/sre-diagnose
- Architecture doc: autonomy/examples/sre-diagnose/planning/ARCHITECTURE.md
- Implementation plan: autonomy/examples/sre-diagnose/planning/IMPLEMENTATION_PLAN.md

## Completed
- Phase 1: Foundation (MVP)
  - Project structure created
  - Basic orchestrator agent working
  - Human-in-the-loop approval flow implemented

## Next Step
- Phase 2: Mock 1Password & Credential Flow

## Key Files to Review
- autonomy/examples/sre-diagnose/images/main/main.py
- autonomy/examples/sre-diagnose/autonomy.yaml

Please read the implementation plan and continue from Phase 2.
```

---

## Phase 2: Mock 1Password HTTP Server & Credential Flow

**Goal**: Implement credential handling with a mock 1Password HTTP server (real integration comes later)

### Task 2.1: Create Mock 1Password HTTP Server

Create a simple HTTP server that simulates 1Password credential retrieval.

**Create new files:**
```
autonomy/examples/sre-diagnose/images/mock-1password/
├── Dockerfile
└── server.py
```

**Mock server capabilities:**
- FastAPI HTTP server running on port 8080
- `GET /secrets/{reference}` - returns fake credentials for known refs
- `GET /health` - health check
- Simulates 1Password secret reference format (op://vault/item/field)

**Example responses:**
```json
GET /secrets/op://Infrastructure/prod-db/password
{
  "reference": "op://Infrastructure/prod-db/password",
  "value": "mock-db-password-12345",
  "vault": "Infrastructure",
  "item": "prod-db",
  "field": "password"
}
```

**Acceptance criteria:**
- [ ] Mock server runs as HTTP server on port 8080
- [ ] Returns fake credentials for known secret refs
- [ ] Returns 404 for unknown refs

**Commit:** `git commit -m "sre-diagnose: add mock 1Password HTTP server"`

---

### Task 2.2: Create Python Tool to Call 1Password Server

Create a Python tool function that the orchestrator agent can use to retrieve credentials.

**Update main.py:**
- Add `get_credential(reference: str)` tool function
- Tool calls mock 1Password HTTP server internally
- Returns only success/failure to agent (credential stored in session, not returned to LLM)

**Security model:**
```python
async def get_credential(reference: str) -> str:
  """
  Retrieve a credential from 1Password.
  
  The actual credential value is stored securely and NOT returned to the LLM.
  Only a confirmation message is returned.
  """
  # Call mock 1Password server
  response = await http_client.get(f"http://localhost:8080/secrets/{reference}")
  
  # Store credential in session (for later use by diagnostic tools)
  session["credentials"][reference] = response.json()["value"]
  
  # Return only confirmation (not the actual credential!)
  return f"Successfully retrieved credential for {reference}"
```

**Acceptance criteria:**
- [ ] Orchestrator can call get_credential tool
- [ ] Credentials stored in session state
- [ ] Actual credential values NEVER returned to LLM

**Commit:** `git commit -m "sre-diagnose: add credential retrieval tool"`

---

### Task 2.3: Integrate Credential Flow with Approval

Wire up the approval flow to credential retrieval.

**Update main.py:**
- After approval, agent can call get_credential tool
- Credentials retrieved and stored in session
- Session tracks which credentials have been approved/retrieved

**Flow:**
1. Agent analyzes problem, identifies needed credentials
2. Agent calls ask_user_for_input requesting approval
3. User approves → agent resumes
4. Agent calls get_credential for each approved reference
5. Credentials stored (not in LLM context)
6. Agent proceeds with diagnosis using tools that read credentials from session

**Acceptance criteria:**
- [ ] Credentials only retrieved after approval
- [ ] Multiple credentials can be retrieved
- [ ] Session tracks credential state

**Commit:** `git commit -m "sre-diagnose: integrate credential flow with approval"`

---

### Task 2.4: Credential Security Verification

Verify credentials never appear in LLM conversation.

**Add:**
- Logging of agent conversation
- Verification that credentials not in logs
- Test script to check security

**Acceptance criteria:**
- [ ] Agent logs don't contain credential values
- [ ] Only credential references appear in conversation
- [ ] Actual credentials only in session storage

**Commit:** `git commit -m "sre-diagnose: verify credential security"`

---

### PHASE 2 HANDOVER PROMPT

```
Continue building the SRE Incident Diagnosis Autonomy app.

## Context
- Project location: autonomy/examples/sre-diagnose
- Architecture doc: autonomy/examples/sre-diagnose/planning/ARCHITECTURE.md
- Implementation plan: autonomy/examples/sre-diagnose/planning/IMPLEMENTATION_PLAN.md

## Completed
- Phase 1: Foundation (MVP) ✓
- Phase 2: Mock 1Password & Credential Flow ✓

## Next Step
- Phase 3: Diagnostic Tools & Specialized Agents

## Key Files to Review
- autonomy/examples/sre-diagnose/images/main/main.py
- autonomy/examples/sre-diagnose/images/mock-1password/server.py
- autonomy/examples/sre-diagnose/autonomy.yaml

Please read the implementation plan and continue from Phase 3.
```

---

## Phase 3: Diagnostic Tools & Specialized Agents (Single Pod)

**Goal**: Add diagnostic capabilities with mock tools and specialized agents, all running in the main pod

### Task 3.1: Create Mock Diagnostic Tools as Python Functions

Create Python tool functions that simulate diagnostic queries. These run in the main container.

**Add to main.py:**
```python
# Mock Database Diagnostic Tools
async def query_db_connections(environment: str) -> str:
  """Query database connection statistics."""
  # Return mock data
  return json.dumps({
    "active_connections": 145,
    "max_connections": 200,
    "idle_connections": 23,
    "waiting_queries": 12
  })

async def query_slow_queries(environment: str, threshold_ms: int = 1000) -> str:
  """Query slow database queries."""
  return json.dumps([
    {"query": "SELECT * FROM orders...", "duration_ms": 5420, "calls": 156},
    {"query": "UPDATE inventory...", "duration_ms": 2340, "calls": 89}
  ])

# Mock Cloud Diagnostic Tools  
async def get_cloudwatch_metrics(service: str, metric: str) -> str:
  """Get CloudWatch metrics for a service."""
  return json.dumps({
    "service": service,
    "metric": metric,
    "values": [75.2, 82.1, 91.5, 88.3, 79.6],
    "unit": "Percent",
    "period": "5min"
  })

async def check_instance_health(instance_id: str) -> str:
  """Check EC2/RDS instance health."""
  return json.dumps({
    "instance_id": instance_id,
    "status": "running",
    "cpu_utilization": 78.5,
    "memory_utilization": 85.2,
    "disk_io_wait": 12.3
  })
```

**Acceptance criteria:**
- [ ] Mock tools return realistic diagnostic data
- [ ] Tools are pure Python functions (no external dependencies)
- [ ] Tools can be added to agent's tool list

**Commit:** `git commit -m "sre-diagnose: add mock diagnostic tools as Python functions"`

---

### Task 3.2: Create Database Diagnostic Agent

Create a specialized agent for database diagnostics that runs in the main pod.

**Add to main.py:**
```python
DB_DIAGNOSTIC_INSTRUCTIONS = """You are a database diagnostic specialist.
Your role is to investigate database-related issues using the available tools.

When investigating:
1. Check connection pool status
2. Identify slow queries
3. Check replication status if applicable
4. Analyze query patterns

Provide findings in a structured format."""

async def run_db_diagnosis(node, session, problem):
  """Run database-focused diagnosis."""
  agent = await Agent.start(
    node=node,
    name=f"db_diagnostician_{session['id']}",
    instructions=DB_DIAGNOSTIC_INSTRUCTIONS,
    model=Model("claude-sonnet-4-v1"),
    tools=[
      Tool(query_db_connections),
      Tool(query_slow_queries),
    ]
  )
  # ... run diagnosis and return findings
```

**Acceptance criteria:**
- [ ] Agent focuses on database issues
- [ ] Uses mock database tools
- [ ] Returns structured findings

**Commit:** `git commit -m "sre-diagnose: add database diagnostic agent"`

---

### Task 3.3: Create Cloud Diagnostic Agent

Create a specialized agent for cloud/infrastructure diagnostics.

**Add to main.py:**
```python
CLOUD_DIAGNOSTIC_INSTRUCTIONS = """You are a cloud infrastructure diagnostic specialist.
Your role is to investigate AWS/cloud infrastructure issues.

When investigating:
1. Check instance health and metrics
2. Review CloudWatch alarms
3. Analyze resource utilization
4. Check for scaling issues

Provide findings in a structured format."""
```

**Acceptance criteria:**
- [ ] Agent focuses on cloud issues
- [ ] Uses mock cloud tools
- [ ] Returns structured findings

**Commit:** `git commit -m "sre-diagnose: add cloud diagnostic agent"`

---

### Task 3.4: Orchestrator Dispatches to Specialized Agents

Update orchestrator to dispatch diagnostic tasks to specialized agents.

**Update main.py:**
- After approval, orchestrator spawns appropriate diagnostic agents
- Agents run in parallel using asyncio.gather()
- Orchestrator collects and synthesizes findings

**Flow:**
```
Orchestrator (after approval)
    ├── spawn DB Diagnostic Agent → findings
    ├── spawn Cloud Diagnostic Agent → findings
    └── synthesize all findings → report
```

**Acceptance criteria:**
- [ ] Orchestrator spawns multiple agents
- [ ] Agents run in parallel
- [ ] Findings aggregated into final report

**Commit:** `git commit -m "sre-diagnose: orchestrator dispatches to specialized agents"`

---

### PHASE 3 HANDOVER PROMPT

```
Continue building the SRE Incident Diagnosis Autonomy app.

## Context
- Project location: autonomy/examples/sre-diagnose
- Architecture doc: autonomy/examples/sre-diagnose/planning/ARCHITECTURE.md
- Implementation plan: autonomy/examples/sre-diagnose/planning/IMPLEMENTATION_PLAN.md

## Completed
- Phase 1: Foundation (MVP) ✓
- Phase 2: Mock 1Password & Credential Flow ✓
- Phase 3: Diagnostic Tools & Specialized Agents ✓

## Current Architecture
- All agents run in main pod (single container)
- Mock tools are Python functions
- Orchestrator dispatches to specialized agents in parallel

## Next Step
- Phase 4: Polish & Production Readiness

## Key Files to Review
- autonomy/examples/sre-diagnose/images/main/main.py
- autonomy/examples/sre-diagnose/autonomy.yaml

Please read the implementation plan and continue from Phase 4.
```

---

## Phase 4: Polish & Production Readiness

**Goal**: Error handling, streaming updates, documentation

### Task 4.1: Error Handling & Timeouts

Add robust error handling throughout.

**Add:**
- Timeout handling for agent operations
- Graceful degradation if diagnostic agents fail
- Proper cleanup on errors
- Retry logic for transient failures

**Commit:** `git commit -m "sre-diagnose: add error handling and timeouts"`

---

### Task 4.2: Streaming Updates

Improve real-time feedback during diagnosis.

**Add:**
- Progress updates during diagnosis phases
- Streaming findings as they're discovered
- Phase transition notifications

**Commit:** `git commit -m "sre-diagnose: improve streaming updates"`

---

### Task 4.3: Status Dashboard Improvements

Enhance the web dashboard.

**Add:**
- Real-time status updates via polling
- Display analysis and findings
- Show approval prompts clearly
- Progress indicators

**Commit:** `git commit -m "sre-diagnose: enhance status dashboard"`

---

### Task 4.4: Documentation

Create comprehensive documentation.

**Add:**
- API documentation in README
- Architecture diagram
- Example curl commands
- Troubleshooting guide

**Commit:** `git commit -m "sre-diagnose: add documentation"`

---

### PHASE 4 HANDOVER PROMPT

```
Continue building the SRE Incident Diagnosis Autonomy app.

## Context
- Project location: autonomy/examples/sre-diagnose
- Architecture doc: autonomy/examples/sre-diagnose/planning/ARCHITECTURE.md
- Implementation plan: autonomy/examples/sre-diagnose/planning/IMPLEMENTATION_PLAN.md

## Completed
- Phase 1: Foundation (MVP) ✓
- Phase 2: Mock 1Password & Credential Flow ✓
- Phase 3: Diagnostic Tools & Specialized Agents ✓
- Phase 4: Polish & Production Readiness ✓

## Next Step
- Phase 5: Distributed Processing (Optional Scale-Out)

## Key Files to Review
- autonomy/examples/sre-diagnose/images/main/main.py
- autonomy/examples/sre-diagnose/autonomy.yaml
- autonomy/examples/sre-diagnose/README.md

Please read the implementation plan and continue from Phase 5.
```

---

## Phase 5: Distributed Processing (Optional Scale-Out)

**Goal**: Scale diagnosis across multiple runner pods for heavy workloads

### Task 5.1: Add Runner Pod Configuration

Update autonomy.yaml to include runner pods with clones.

**Update autonomy.yaml:**
```yaml
pods:
  - name: main-pod
    public: true
    containers:
      - name: main
        image: main
      - name: mock-1password
        image: mock-1password
        
  - name: runner-pod
    clones: 3
    containers:
      - name: runner
        image: runner
```

**Create:**
```
autonomy/examples/sre-diagnose/images/runner/
├── Dockerfile
└── main.py
```

**Commit:** `git commit -m "sre-diagnose: add runner pod configuration"`

---

### Task 5.2: Implement Worker Distribution

Add worker dispatch logic to orchestrator.

**Add:**
- DiagnosticWorker class in runner
- Zone.nodes() discovery from main pod
- Message passing between main and runner pods
- Result aggregation

**Acceptance criteria:**
- [ ] Workers discovered across pods
- [ ] Tasks distributed to workers via messaging
- [ ] Results collected and aggregated
- [ ] Failures handled gracefully

**Commit:** `git commit -m "sre-diagnose: implement worker distribution"`

---

### Task 5.3: Parallel Diagnosis Across Runners

Distribute diagnostic agents across runner pods.

**Add:**
- Diagnostic agents run on runner pods
- Main pod orchestrates and collects results
- Load balancing across available runners

**Commit:** `git commit -m "sre-diagnose: parallel diagnosis across runners"`

---

### PHASE 5 HANDOVER PROMPT

```
Continue building the SRE Incident Diagnosis Autonomy app.

## Context
- Project location: autonomy/examples/sre-diagnose
- Architecture doc: autonomy/examples/sre-diagnose/planning/ARCHITECTURE.md
- Implementation plan: autonomy/examples/sre-diagnose/planning/IMPLEMENTATION_PLAN.md

## Completed
- Phase 1: Foundation (MVP) ✓
- Phase 2: Mock 1Password & Credential Flow ✓
- Phase 3: Diagnostic Tools & Specialized Agents ✓
- Phase 4: Polish & Production Readiness ✓
- Phase 5: Distributed Processing ✓

## Next Step
- Phase 6: Cursor Hooks Integration

## Key Files to Review
- autonomy/examples/sre-diagnose/images/main/main.py
- autonomy/examples/sre-diagnose/images/runner/main.py
- autonomy/examples/sre-diagnose/autonomy.yaml

Please read the implementation plan and continue from Phase 6.
```

---

## Phase 6: Cursor Hooks Integration ✅ COMPLETED

**Goal**: Deep integration with Cursor via hooks

### Task 6.1: Create Cursor Hooks Configuration ✅

Created hooks.json and hook scripts in `.cursor/` directory.

**Created:**
```
autonomy/examples/sre-diagnose/.cursor/
├── hooks.json
└── hooks/
    ├── session-init.sh
    ├── sre-diagnose.sh
    └── audit.sh
```

**hooks.json:**
```json
{
  "version": 1,
  "hooks": {
    "session-start": [{"command": ".cursor/hooks/session-init.sh"}],
    "pre-tool-call": [{"command": ".cursor/hooks/sre-diagnose.sh", "matcher": "run_terminal_cmd"}],
    "post-tool-call": [{"command": ".cursor/hooks/audit.sh"}]
  }
}
```

---

### Task 6.2: Implement Session Hooks ✅

**session-init.sh:**
- Sets `SRE_DIAGNOSE_URL` and `SRE_SESSION_ID` environment variables
- Provides context about SRE Diagnose API capabilities

**sre-diagnose.sh:**
- Intercepts `sre-diagnose` and `diagnose-infra` commands
- Blocks shell execution and provides API guidance
- Returns `{"decision": "block", "instructions": "..."}` for diagnosis commands

**audit.sh:**
- Logs tool calls to `/tmp/sre-diagnose/audit.log`
- Fire-and-forget (returns `{}`)

---

### Task 6.3: Document Cursor Integration ✅

Added to README.md:
- How to copy `.cursor` folder to projects
- Hook configuration table
- Usage examples with natural language and command patterns
- Environment variables set by hooks

---

## Phase 7: Real 1Password Integration ✅ COMPLETED

**Goal**: Replace mock 1Password server with real SDK integration

### Task 7.1: Research 1Password Options ✅

Evaluated three options:
1. **1Password Connect Server** - Requires additional containers, complex setup
2. **1Password CLI** - Shell-based, less reliable
3. **1Password Python SDK** ✅ - Chosen approach, native Python integration

### Task 7.2: Implement Dual-Mode Support ✅

**Changes:**
- Added `onepassword-sdk` to `requirements.txt`
- Added `ONEPASSWORD_MODE` environment variable (`mock` or `sdk`)
- Implemented `retrieve_credential_sdk()` using official SDK
- Implemented `retrieve_credential_mock()` for development
- Updated `/health` endpoint to show 1Password mode and status
- Created `secrets.yaml.example` with setup instructions
- Added `.gitignore` to exclude `secrets.yaml`

**Code structure:**
```python
async def retrieve_credential(reference: str, session: dict) -> tuple[bool, str]:
    if ONEPASSWORD_MODE == "sdk":
        return await retrieve_credential_sdk(reference, session)
    else:
        return await retrieve_credential_mock(reference, session)
```

### Task 7.3: Update Documentation ✅

- Updated README.md with 1Password Integration section
- Rewrote `1PASSWORD_CONNECT_REFERENCE.md` with SDK details
- Documented both mock and SDK modes

---

## Phase 8: Real Diagnostic Tools (Optional - Future)

**Goal**: Replace mock diagnostic tools with real integrations

### Task 8.1: Error Handling & Timeouts

- Add comprehensive error handling
- Implement retry logic
- Add timeouts throughout
- Create user-friendly error messages

**Commit:** `git commit -m "sre-diagnose: add error handling and timeouts"`

---

### Task 6.2: Streaming Updates

- Implement WebSocket or SSE for real-time updates
- Stream diagnostic progress
- Show agent activity

**Commit:** `git commit -m "sre-diagnose: implement streaming updates"`

---

### Task 6.3: Status Dashboard

- Enhance index.html
- Show active diagnoses
- Display agent activity
- Add audit log viewer

**Commit:** `git commit -m "sre-diagnose: add status dashboard"`

---

### Task 6.4: Documentation

- Complete README.md
- Add architecture diagrams
- Write troubleshooting guide
- Document all APIs

**Commit:** `git commit -m "sre-diagnose: complete documentation"`

---

### PHASE 7 HANDOVER PROMPT

```
Continue building the SRE Incident Diagnosis Autonomy app.

## Context
- Project location: autonomy/examples/sre-diagnose
- Architecture doc: autonomy/examples/sre-diagnose/planning/ARCHITECTURE.md
- Implementation plan: autonomy/examples/sre-diagnose/planning/IMPLEMENTATION_PLAN.md

## Completed
- Phase 1-6: All core functionality complete ✓

## Next Step
- Phase 7: Real 1Password Integration (replacing mock)

Please read the implementation plan and continue from Phase 7.
```

---

## Phase 7: Real 1Password Integration

**Goal**: Replace mock 1Password with real integration

### Task 7.1: Research 1Password MCP Server

- Identify correct container image
- Understand authentication (service account)
- Document API for credential retrieval
- Test locally

**Questions to answer:**
- What is the exact container image?
- How does service account auth work?
- What's the API for retrieving secrets?
- Secret reference format (op://vault/item)?

**Commit:** `git commit -m "sre-diagnose: document 1Password integration requirements"`

---

### Task 7.2: Implement Real 1Password Integration

- Replace mock-1password container with real one
- Configure service account token
- Update secrets.yaml
- Test credential retrieval

**Commit:** `git commit -m "sre-diagnose: integrate real 1Password server"`

---

### Task 7.3: Real Diagnostic Tools (Optional)

Replace mock tools with real MCP servers:
- mcp-server-postgres
- @aws-mcp/server
- @modelcontextprotocol/server-kubernetes

**Commit:** `git commit -m "sre-diagnose: integrate real diagnostic MCP servers"`

---

### FINAL HANDOVER PROMPT

```
The SRE Incident Diagnosis Autonomy app is complete.

## Project Location
- autonomy/examples/sre-diagnose

## Documentation
- README.md - User guide
- autonomy/examples/sre-diagnose/planning/ARCHITECTURE.md - System architecture
- autonomy/examples/sre-diagnose/planning/IMPLEMENTATION_PLAN.md - This plan

## To Deploy
cd autonomy/examples/sre-diagnose
autonomy zone deploy

## To Test
See README.md for curl examples and Cursor integration instructions.
```

---

## File Structure (Final)

```
autonomy/examples/sre-diagnose/
├── autonomy.yaml
├── README.md
│
├── images/
│   ├── main/                    # Main container (all agents run here)
│   │   ├── Dockerfile
│   │   ├── main.py              # FastAPI + orchestrator + diagnostic agents + mock tools
│   │   └── index.html           # Status dashboard
│   │
│   ├── mock-1password/          # Phase 2 (HTTP server, replaced in Phase 7)
│   │   ├── Dockerfile
│   │   └── server.py
│   │
│   └── runner/                  # Phase 5 (optional distributed processing)
│       ├── Dockerfile
│       └── main.py
│
└── cursor-hooks/                # Phase 6
    ├── hooks.json
    └── hooks/
        ├── sre-session-init.sh
        ├── audit-mcp.sh
        └── cleanup-session.sh
```

**Note:** Phases 1-4 use only the `main` container. All agents (orchestrator, DB diagnostic, cloud diagnostic) 
run in the main pod. Mock diagnostic tools are Python functions in `main.py`, not separate containers.
The `runner` pod is only added in Phase 5 for optional scale-out.
```

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Foundation | 2-3 hours | None |
| Phase 2: Mock 1Password HTTP Server | 2-3 hours | Phase 1 |
| Phase 3: Diagnostic Tools & Agents (Single Pod) | 3-4 hours | Phase 2 |
| Phase 4: Polish & Production Readiness | 2-3 hours | Phase 3 | ✅ |
| Phase 5: Distributed Processing (Optional) | - | - | Skipped |
| Phase 6: Cursor Hooks | 2-3 hours | Phase 1 | ✅ |
| Phase 7: Real 1Password Integration | 2-3 hours | Phase 2 | ✅ |
| Phase 8: Real Diagnostic Tools (Optional) | 4-6 hours | All previous | Future |

**Total Completed: ~15-20 hours** across multiple sessions

---

## Starting Prompt (For First Session)

```
Build an SRE Incident Diagnosis Autonomy app.

## Goal
Enable developers to diagnose infrastructure problems from Cursor IDE by spinning up
autonomous diagnostic agents in Autonomy, with secure credential retrieval via
human-in-the-loop approval.

## Context
- Architecture: autonomy/examples/sre-diagnose/planning/ARCHITECTURE.md
- Plan: autonomy/examples/sre-diagnose/planning/IMPLEMENTATION_PLAN.md
- Location: autonomy/examples/sre-diagnose

## Start With
- Phase 1, Task 1.1: Create Project Structure

## Workflow
1. Read the architecture and plan documents
2. Implement the task
3. Test it works
4. Commit with descriptive message
5. Move to next task

Please start by reading the architecture document, then begin Phase 1.
```

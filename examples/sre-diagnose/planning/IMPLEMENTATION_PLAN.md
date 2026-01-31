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

---

## Phase 9: Agent Visualization (D3.js Force-Directed Graph)

**Goal**: Add real-time visualization of all agents in a session, similar to code-review/011 example

### Task 9.1: Add Graph State Tracking

Add global graph state to `main.py`:

```python
# Graph state for visualization
graph_state = {
    "nodes": [],      # All agent nodes
    "edges": [],      # Parent-child relationships
    "reports": {},    # Agent reports/findings
    "transcripts": {},# Agent conversation logs
    "activity": [],   # Recent activity feed
    "status": "idle", # idle, running, completed
}
graph_lock = asyncio.Lock()
MAX_ACTIVITY_ITEMS = 100

async def add_node(node_id: str, name: str, node_type: str, parent_id: str = None, meta: dict = None):
    """Add a node to the visualization graph."""
    async with graph_lock:
        node = {
            "id": node_id,
            "name": name,
            "type": node_type,  # root, region, service, runner, diagnostic-agent, sub-agent, synthesis
            "status": "pending",
            "parent": parent_id,
            "meta": meta or {},
            "created_at": datetime.utcnow().isoformat(),
        }
        graph_state["nodes"].append(node)
        if parent_id:
            graph_state["edges"].append({"source": parent_id, "target": node_id})
        return node

async def add_edge(source_id: str, target_id: str):
    """Add an edge between two nodes."""
    async with graph_lock:
        graph_state["edges"].append({"source": source_id, "target": target_id})

async def update_node_status(node_id: str, status: str, report: dict = None):
    """Update a node's status and optionally add a report."""
    async with graph_lock:
        for node in graph_state["nodes"]:
            if node["id"] == node_id:
                node["status"] = status
                node["updated_at"] = datetime.utcnow().isoformat()
                break
        if report:
            graph_state["reports"][node_id] = report

async def add_transcript_entry(node_id: str, role: str, content: str, entry_type: str = "message"):
    """Add an entry to a node's transcript and activity feed."""
    async with graph_lock:
        if node_id not in graph_state["transcripts"]:
            graph_state["transcripts"][node_id] = []
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "content": content,
            "type": entry_type,
        }
        graph_state["transcripts"][node_id].append(entry)
        
        # Add to activity feed
        node_name = node_id
        for node in graph_state["nodes"]:
            if node["id"] == node_id:
                node_name = node.get("name", node_id)
                break
        
        activity_entry = {
            "timestamp": entry["timestamp"],
            "node_id": node_id,
            "node_name": node_name,
            "role": role,
            "content": content[:200] + ("..." if len(content) > 200 else ""),
            "type": entry_type,
        }
        graph_state["activity"].insert(0, activity_entry)
        
        if len(graph_state["activity"]) > MAX_ACTIVITY_ITEMS:
            graph_state["activity"] = graph_state["activity"][:MAX_ACTIVITY_ITEMS]

async def reset_graph():
    """Reset the graph state for a new session."""
    async with graph_lock:
        graph_state["nodes"] = []
        graph_state["edges"] = []
        graph_state["reports"] = {}
        graph_state["transcripts"] = {}
        graph_state["activity"] = []
        graph_state["status"] = "idle"
```

**Commit:** `git commit -m "sre-diagnose: add graph state tracking for visualization"`

---

### Task 9.2: Add Visualization API Endpoints

Add endpoints to serve graph data:

```python
@app.get("/graph")
async def get_graph():
    """Return current graph state for visualization."""
    async with graph_lock:
        return JSONResponse(content={
            "nodes": graph_state["nodes"],
            "edges": graph_state["edges"],
            "status": graph_state["status"],
        })

@app.get("/graph/report/{node_id}")
async def get_node_report(node_id: str):
    """Return report and transcript for a specific node."""
    async with graph_lock:
        return JSONResponse(content={
            "report": graph_state["reports"].get(node_id),
            "transcript": graph_state["transcripts"].get(node_id, []),
        })

@app.get("/activity")
async def get_activity():
    """Return recent activity feed."""
    async with graph_lock:
        return JSONResponse(content=graph_state["activity"][:50])

@app.post("/graph/reset")
async def reset_graph_endpoint():
    """Reset the graph state."""
    await reset_graph()
    return JSONResponse(content={"status": "reset"})
```

**Commit:** `git commit -m "sre-diagnose: add visualization API endpoints"`

---

### Task 9.3: Add D3.js Force-Directed Graph to Dashboard

Update `index.html` with D3.js visualization:

**Key Components:**

1. **Graph Container** - Full-width SVG with zoom/pan
2. **Node Types and Colors:**
   - `root` (purple #a78bfa) - Investigation root
   - `region` (cyan #22d3ee) - AWS region
   - `service` (blue #60a5fa) - Service being investigated
   - `runner` (teal #06b6d4) - Runner pod
   - `diagnostic-agent` (green #4ade80) - Primary diagnostic agent
   - `sub-agent` (gold #fbbf24) - Sub-agent for deep investigation
   - `synthesis` (pink #ec4899) - Synthesis agent

3. **Status Colors:**
   - `pending` (gray #666)
   - `running` (yellow #facc15, animated pulse)
   - `completed` (green #4ade80)
   - `error` (red #f87171)

4. **Features:**
   - Force-directed layout with collision detection
   - Auto-zoom to fit all nodes
   - Click nodes to see reports/transcripts
   - Activity feed showing real-time agent progress
   - Stats panel (total agents, running, completed)

5. **Split-View Layout:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔧 SRE Incident Diagnosis                              [Stats: 156/200]│
├─────────────────────────────────────────┬───────────────────────────────┤
│                                         │  Activity Feed               │
│      Force-Directed Graph               │  ─────────────────            │
│      Visualization                      │  [Runner 2] cache-agent done  │
│                                         │  [Runner 1] Starting db-agent │
│      (nodes, edges, animation)          │  [Runner 3] network-agent...  │
│                                         │                               │
├─────────────────────────────────────────┴───────────────────────────────┤
│  Node Details (click to select)                                         │
│  Selected: us-east-1/api-gateway/database-agent | Status: ✓ Completed   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Commit:** `git commit -m "sre-diagnose: add D3.js force-directed graph visualization"`

---

### Task 9.4: Integrate Graph Updates with Existing Diagnosis Flow

Update the diagnosis flow to emit graph events:

```python
async def diagnose(request: DiagnoseRequest, node: NodeDep):
    # Create root node for visualization
    session_id = secrets.token_hex(8)
    root_id = f"investigation-{session_id}"
    await add_node(root_id, "Incident Investigation", "root")
    await update_node_status(root_id, "running")
    graph_state["status"] = "running"
    
    # ... existing analysis code ...
    
    # Create analysis agent node
    analysis_id = f"analysis-{session_id}"
    await add_node(analysis_id, "Analysis Agent", "diagnostic-agent", root_id)
    await update_node_status(analysis_id, "running")
    await add_transcript_entry(analysis_id, "system", f"Analyzing: {request.problem}")
    
    # ... run analysis ...
    
    await update_node_status(analysis_id, "completed", {"analysis": analysis_text})

async def approve(session_id: str, ...):
    # Create specialist agent nodes
    for agent_type in specialist_agents:
        agent_id = f"{agent_type}-{session_id}"
        await add_node(agent_id, f"{agent_type.title()} Specialist", "diagnostic-agent", root_id)
        await update_node_status(agent_id, "running")
        
    # ... run specialists in parallel ...
    
    # Create synthesis node
    synthesis_id = f"synthesis-{session_id}"
    await add_node(synthesis_id, "Synthesis Agent", "synthesis", root_id)
```

**Commit:** `git commit -m "sre-diagnose: integrate graph updates with diagnosis flow"`

---

### PHASE 9 HANDOVER PROMPT

```
# SRE Incident Diagnosis - Handover Prompt

## Project Location
- Path: `autonomy/examples/sre-diagnose`
- Branch: `sre-diagnose-phase9-visualization`

## Completed Phases
- [x] Phase 1-7: Core functionality complete
- [x] Phase 9: Agent Visualization with D3.js force-directed graph

## What Was Added
- Graph state tracking (nodes, edges, reports, transcripts, activity)
- Visualization API endpoints (/graph, /activity, /graph/report/{id})
- D3.js force-directed graph in dashboard
- Real-time activity feed
- Click-to-inspect node details

## Next Step
- Phase 10: Agent Swarm Scaling (100s of parallel agents)

## Deploy & Test
cd autonomy/examples/sre-diagnose
autonomy zone deploy

# View visualization at dashboard URL
```

---

## Phase 10: Agent Swarm Scaling (100s of Parallel Agents)

**Goal**: Scale to 100s of parallel agents using distributed runner pods, demonstrating swarm-like diagnosis

### Task 10.1: Add Runner Pod Infrastructure

Update `autonomy.yaml`:

```yaml
name: srediag
pods:
  - name: main-pod
    size: big
    public: true
    containers:
      - name: main
        image: main
        env:
          ONEPASSWORD_MODE: mock
      - name: onepass
        image: mock-1password

  - name: runner-pod
    size: big
    clones: 5  # 5 runner pods for parallel diagnosis
    containers:
      - name: runner
        image: runner
```

Create `images/runner/main.py`:

```python
from autonomy import Node

# Simple runner - starts a node that can receive workers
Node.start()
```

Create `images/runner/Dockerfile`:

```dockerfile
FROM ghcr.io/build-trust/autonomy-python:latest
COPY main.py .
ENTRYPOINT ["python", "main.py"]
```

**Commit:** `git commit -m "sre-diagnose: add runner pod infrastructure"`

---

### Task 10.2: Create DiagnosticWorker Class

Add worker that runs on remote runners:

```python
class DiagnosticWorker:
    """Worker that runs on remote runners to execute diagnostic agents."""
    
    async def run_service_diagnosis(self, service_info: dict) -> dict:
        """Run multiple diagnostic agents for a single service."""
        from autonomy import Agent, Model
        
        service_name = service_info["service"]
        region = service_info["region"]
        node_id = service_info["node_id"]
        
        # Define diagnostic agent types
        agent_configs = [
            ("database", "Analyze database connections, query latency, connection pool health..."),
            ("cache", "Analyze cache hit rates, eviction patterns, memory usage..."),
            ("network", "Check network latency, DNS resolution, connection errors..."),
            ("resources", "Check CPU, memory, disk I/O, container resource limits..."),
            ("logs", "Scan application logs for errors, warning patterns, anomalies..."),
        ]
        
        model = Model(
            "nova-micro-v1",
            throttle=True,
            throttle_requests_per_minute=1000,
            throttle_max_requests_in_progress=100,
        )
        
        results = {}
        for agent_type, instructions in agent_configs:
            agent = await Agent.start(
                node=self.node,
                instructions=instructions,
                model=model,
                max_execution_time=300.0,
                max_iterations=3,
            )
            
            try:
                message = f"Diagnose {agent_type} for {service_name} in {region}"
                response = await agent.send(message, timeout=120)
                results[agent_type] = {
                    "status": "completed",
                    "findings": response[-1].content.text if response else "No response"
                }
            except Exception as e:
                results[agent_type] = {"status": "error", "error": str(e)}
            finally:
                await Agent.stop(self.node, agent.name, timeout=5.0)
        
        return {
            "service": service_name,
            "region": region,
            "results": results,
        }
    
    async def handle_message(self, context, message):
        """Handle batch of services to diagnose with progress updates."""
        import json
        
        request = json.loads(message)
        services = request.get("services", [])
        runner_id = request.get("runner_id", "unknown")
        
        results = []
        for idx, service_info in enumerate(services):
            # Send starting progress
            await context.reply(json.dumps({
                "type": "progress",
                "runner_id": runner_id,
                "status": "starting",
                "service": service_info["service"],
                "region": service_info["region"],
                "index": idx + 1,
                "total": len(services),
            }))
            
            result = await self.run_service_diagnosis(service_info)
            results.append(result)
            
            # Send completed progress
            await context.reply(json.dumps({
                "type": "progress",
                "runner_id": runner_id,
                "status": "completed",
                "service": service_info["service"],
                "region": service_info["region"],
                "index": idx + 1,
                "total": len(services),
                "result": result,
            }))
        
        # Send final results
        await context.reply(json.dumps({
            "type": "final",
            "runner_id": runner_id,
            "results": results,
        }))
```

**Commit:** `git commit -m "sre-diagnose: add DiagnosticWorker for distributed diagnosis"`

---

### Task 10.3: Implement Distributed Diagnosis Flow

Add orchestration for swarm diagnosis:

```python
def split_list_into_n_parts(lst, n):
    """Split a list into n roughly equal parts."""
    if n <= 0:
        return [lst]
    q, r = divmod(len(lst), n)
    return [lst[i * q + min(i, r): (i + 1) * q + min(i + 1, r)] for i in range(n)]


async def run_distributed_diagnosis(node, problem: str, session_id: str, root_id: str):
    """Distribute diagnosis across runners with massive parallelism."""
    
    # Define investigation targets (multi-region, multi-service)
    regions = ["us-east-1", "us-west-2", "eu-west-1"]
    services = [
        "api-gateway", "user-service", "order-service", "payment-service",
        "inventory-service", "notification-service", "auth-service",
        "analytics-service", "search-service", "recommendation-service"
    ]
    
    # Create region and service nodes in visualization
    targets = []
    for region in regions:
        region_id = f"region-{region}-{session_id}"
        await add_node(region_id, region, "region", root_id)
        await update_node_status(region_id, "running")
        
        for service in services:
            service_id = f"service-{region}-{service}-{session_id}"
            await add_node(service_id, service, "service", region_id)
            targets.append({
                "service": service,
                "region": region,
                "node_id": service_id,
            })
    
    await add_transcript_entry(root_id, "system", 
        f"Created {len(targets)} investigation targets across {len(regions)} regions")
    
    # Discover runners and distribute work
    runners = await Zone.nodes(node, filter="runner")
    num_runners = len(runners)
    
    if num_runners == 0:
        # Fallback to local execution
        await add_transcript_entry(root_id, "system", "No runners found, running locally")
        return await run_local_diagnosis(node, targets, session_id, root_id)
    
    await add_transcript_entry(root_id, "system", 
        f"Found {num_runners} runner nodes, distributing {len(targets)} targets")
    
    # Create runner nodes in visualization
    runner_node_ids = []
    for i in range(num_runners):
        runner_node_id = f"runner-{i+1}-{session_id}"
        await add_node(runner_node_id, f"Runner {i+1}", "runner", root_id, {
            "runner_name": runners[i].name,
        })
        await update_node_status(runner_node_id, "running")
        runner_node_ids.append(runner_node_id)
    
    # Split targets across runners
    target_batches = split_list_into_n_parts(targets, num_runners)
    
    async def run_on_runner(runner_idx, runner, batch):
        """Run diagnosis on a single runner."""
        worker_name = f"diagnostic-{secrets.token_hex(4)}"
        runner_id = f"Runner {runner_idx + 1}"
        runner_node_id = runner_node_ids[runner_idx]
        
        await runner.start_worker(worker_name, DiagnosticWorker())
        
        request = json.dumps({
            "services": batch,
            "runner_id": runner_id,
        })
        
        results = []
        try:
            mailbox = await node.send(worker_name, request, node=runner.name)
            
            while True:
                try:
                    reply_json = await mailbox.receive(timeout=1800)
                    msg = json.loads(reply_json)
                    msg_type = msg.get("type", "")
                    
                    if msg_type == "progress":
                        status = msg.get("status", "")
                        service = msg.get("service", "")
                        region = msg.get("region", "")
                        service_node_id = f"service-{region}-{service}-{session_id}"
                        
                        if status == "starting":
                            await update_node_status(service_node_id, "running")
                            
                            # Create diagnostic agent nodes
                            for agent_type in ["database", "cache", "network", "resources", "logs"]:
                                agent_node_id = f"agent-{region}-{service}-{agent_type}-{session_id}"
                                await add_node(agent_node_id, agent_type, "diagnostic-agent", service_node_id)
                                await update_node_status(agent_node_id, "running")
                            
                            await add_transcript_entry(runner_node_id, "system", 
                                f"Starting diagnosis: {service} in {region}")
                        
                        elif status == "completed":
                            await update_node_status(service_node_id, "completed")
                            
                            # Update agent nodes with results
                            result = msg.get("result", {})
                            for agent_type, agent_result in result.get("results", {}).items():
                                agent_node_id = f"agent-{region}-{service}-{agent_type}-{session_id}"
                                agent_status = "completed" if agent_result.get("status") == "completed" else "error"
                                await update_node_status(agent_node_id, agent_status, agent_result)
                            
                            await add_transcript_entry(runner_node_id, "system", 
                                f"Completed: {service} in {region}")
                    
                    elif msg_type == "final":
                        results = msg.get("results", [])
                        await update_node_status(runner_node_id, "completed")
                        break
                
                except Exception as e:
                    await add_transcript_entry(runner_node_id, "error", str(e))
                    break
        
        except Exception as e:
            await update_node_status(runner_node_id, "error")
            await add_transcript_entry(runner_node_id, "error", str(e))
        finally:
            try:
                await runner.stop_worker(worker_name)
            except:
                pass
        
        return results
    
    # Run all runners in parallel
    futures = [run_on_runner(i, runners[i], target_batches[i]) for i in range(num_runners)]
    all_results = await asyncio.gather(*futures)
    
    # Flatten results
    flattened = []
    for batch in all_results:
        if isinstance(batch, list):
            flattened.extend(batch)
    
    # Update region nodes to completed
    for region in regions:
        region_id = f"region-{region}-{session_id}"
        await update_node_status(region_id, "completed")
    
    return flattened
```

**Commit:** `git commit -m "sre-diagnose: implement distributed diagnosis across runners"`

---

### Task 10.4: Demo Scenario - "Production Latency Spike"

Create a compelling demo that shows 200+ agents:

**Scenario Input:** "Production API experiencing latency spikes across all regions"

**Agent Hierarchy Created:**

```
Investigation Root
├── Runner 1 (handling us-east-1)
├── Runner 2 (handling us-west-2)
├── Runner 3 (handling eu-west-1)
├── Runner 4 (overflow)
├── Runner 5 (overflow)
│
├── us-east-1 (Region)
│   ├── api-gateway
│   │   ├── database-agent
│   │   ├── cache-agent
│   │   ├── network-agent
│   │   ├── resources-agent
│   │   └── logs-agent
│   ├── user-service (5 agents)
│   ├── order-service (5 agents)
│   └── ... (10 services × 5 agents = 50 agents)
│
├── us-west-2 (Region) - 50 agents
├── eu-west-1 (Region) - 50 agents
│
└── Synthesis Agent
```

**Scale Math:**
- 3 regions × 10 services = 30 investigation targets
- 5 diagnostic agents per service = 150 base agents
- Sub-agents for critical findings = ~50 more
- **Total: 200+ agents** visualized in the graph

**Integration:**

```python
async def diagnose_with_swarm(request: DiagnoseRequest, node: NodeDep):
    """Enhanced diagnosis using agent swarm."""
    session_id = secrets.token_hex(8)
    root_id = f"investigation-{session_id}"
    
    await reset_graph()
    await add_node(root_id, "Incident Investigation", "root", meta={
        "problem": request.problem,
        "environment": request.environment,
    })
    await update_node_status(root_id, "running")
    graph_state["status"] = "running"
    
    # Run distributed diagnosis across all regions/services
    results = await run_distributed_diagnosis(node, request.problem, session_id, root_id)
    
    # Run synthesis
    synthesis_id = f"synthesis-{session_id}"
    await add_node(synthesis_id, "Synthesis Agent", "synthesis", root_id)
    await update_node_status(synthesis_id, "running")
    
    # ... synthesize findings ...
    
    await update_node_status(synthesis_id, "completed")
    await update_node_status(root_id, "completed")
    graph_state["status"] = "completed"
```

**Commit:** `git commit -m "sre-diagnose: add swarm demo scenario with 200+ agents"`

---

### Task 10.5: Update Dashboard for Swarm Visualization

Enhance the dashboard to handle large agent counts:

1. **Performance optimizations:**
   - Throttle graph updates (batch every 100ms)
   - Use canvas rendering for 500+ nodes
   - Implement node clustering for dense areas

2. **Stats panel:**
   - Total agents spawned
   - Running / Completed / Error counts
   - Agents per runner breakdown
   - Time elapsed

3. **Activity feed improvements:**
   - Filter by runner
   - Search functionality
   - Collapse/expand regions

**Commit:** `git commit -m "sre-diagnose: optimize dashboard for large agent counts"`

---

### PHASE 10 HANDOVER PROMPT

```
# SRE Incident Diagnosis - Handover Prompt

## Project Location
- Path: `autonomy/examples/sre-diagnose`
- Branch: `sre-diagnose-phase10-swarm`

## Completed Phases
- [x] Phase 1-7: Core functionality
- [x] Phase 9: Agent Visualization
- [x] Phase 10: Agent Swarm Scaling

## What Was Added
- Runner pod infrastructure (5 clones)
- DiagnosticWorker for distributed execution
- Multi-region, multi-service investigation
- 200+ parallel agents in swarm demo
- Optimized dashboard for large graphs

## Demo
1. Deploy: `autonomy zone deploy`
2. Open dashboard
3. Enter: "Production API latency spike across all regions"
4. Watch 200+ agents spawn and investigate in parallel
5. See results synthesized

## Scale Math
- 3 regions × 10 services = 30 targets
- 5 agents per service = 150 base agents
- Sub-agents + synthesis = 200+ total

## Files Changed
- autonomy.yaml (added runner pods)
- images/runner/ (new runner image)
- images/main/main.py (DiagnosticWorker, distributed flow)
- images/main/index.html (enhanced visualization)
```

---

### Task 8.1: Real Database Integration

- Replace mock `query_db_connections` with real PostgreSQL queries
- Use `mcp-server-postgres` or direct `asyncpg` connection
- Requires real database credentials from 1Password

**Commit:** `git commit -m "sre-diagnose: integrate real database diagnostics"`

---

### Task 8.2: Real AWS Integration

- Replace mock CloudWatch/EC2 functions with real AWS SDK calls
- Use `@aws-mcp/server` or `boto3` directly
- Requires AWS credentials from 1Password

**Commit:** `git commit -m "sre-diagnose: integrate real AWS diagnostics"`

---

### Task 8.3: Real Kubernetes Integration

- Replace mock `check_kubernetes_pods` with real K8s API calls
- Use `@modelcontextprotocol/server-kubernetes` or `kubernetes` Python client
- Requires kubeconfig from 1Password

**Commit:** `git commit -m "sre-diagnose: integrate real Kubernetes diagnostics"`

---

### Task 8.4: Real Log Aggregation

- Integrate with real log systems (Datadog, CloudWatch Logs, ELK)
- Use appropriate MCP servers or SDKs
- Pattern matching for anomaly detection

**Commit:** `git commit -m "sre-diagnose: integrate real log aggregation"`

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

## File Structure (Final with Swarm Support)

```
autonomy/examples/sre-diagnose/
├── autonomy.yaml                # Zone config with runner pods (clones: 5)
├── README.md
│
├── images/
│   ├── main/                    # Main container (orchestration + visualization)
│   │   ├── Dockerfile
│   │   ├── main.py              # FastAPI + graph state + distributed orchestration
│   │   ├── index.html           # D3.js force-directed graph dashboard
│   │   └── requirements.txt
│   │
│   ├── mock-1password/          # Mock credential server
│   │   ├── Dockerfile
│   │   └── server.py
│   │
│   └── runner/                  # Runner pods for distributed diagnosis
│       ├── Dockerfile
│       └── main.py              # Simple Node.start() for receiving workers
│
├── .cursor/                     # Cursor IDE hooks
│   └── hooks.json
│
└── planning/
    ├── ARCHITECTURE.md
    ├── IMPLEMENTATION_PLAN.md
    └── 1PASSWORD_CONNECT_REFERENCE.md
```

**Architecture Notes:**
- `main` pod handles orchestration, visualization APIs, and synthesis
- `runner` pods (5 clones) execute DiagnosticWorker for parallel diagnosis
- Communication via Autonomy's Zone.nodes() discovery and mailbox messaging
- Graph state tracked in main pod, polled by dashboard for visualization
```

---

## Timeline Estimate

| Phase | Duration | Dependencies | Status |
|-------|----------|--------------|--------|
| Phase 1: Foundation | 2-3 hours | None | ✅ |
| Phase 2: Mock 1Password HTTP Server | 2-3 hours | Phase 1 | ✅ |
| Phase 3: Diagnostic Tools & Agents (Single Pod) | 3-4 hours | Phase 2 | ✅ |
| Phase 4: Polish & Production Readiness | 2-3 hours | Phase 3 | ✅ |
| Phase 5: Distributed Processing (Original) | - | - | Skipped |
| Phase 6: Cursor Hooks | 2-3 hours | Phase 1 | ✅ |
| Phase 7: Real 1Password Integration | 2-3 hours | Phase 2 | ✅ |
| Phase 8: Real Diagnostic Tools | 4-6 hours | All previous | Future |
| **Phase 9: Agent Visualization** | 3-4 hours | Phase 4 | **Next** |
| **Phase 10: Agent Swarm Scaling** | 4-6 hours | Phase 9 | **Next** |

**Estimated Remaining: ~7-10 hours** for Phases 9-10

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

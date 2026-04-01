# HlAB Platform Optimization — Design Spec

**Date**: 2026-04-01
**Approach**: Bottom-Up Foundation First (4 phases)
**Scope**: Full platform optimization — DB, security, pgvector, workflow engine, agent runtime, frontend UX

---

## Context

After comprehensive analysis of the entire codebase (57 backend issues, 40+ frontend issues, UX score 4/10), six major areas need improvement:

1. **Workflow engine is half-built** — `next_step_rules` conditional branching defined in schema but never evaluated in execution. `on_failure="rollback"` has no handler.
2. **FAISS is a scalability dead-end** — file-based, single-process, no ACID, no concurrent writes, crash = data loss. No pgvector support.
3. **Frontend UX is not fool-proof** — raw JSON for workflow fields, no visual builder, 1184-line monolith pages, pervasive `any` types, mixed Chinese/English with no i18n.
4. **Security is production-blocking** — auth disabled by default, hardcoded secret key, tenant isolation from request body, plaintext credentials.
5. **Missing DB indexes** — `knowledge_chunks.entity_key` (critical for fast channel) unindexed. No composite indexes for multi-tenant queries.
6. **Agent delegation loses context** — delegated agents get new session with zero conversation history.

---

## Phase 1: Foundation (DB, Security, pgvector, Error Handling)

### 1.1 Database Indexes & Constraints

**New indexes:**
- `knowledge_chunks.entity_key` — fast channel lookup performance
- `knowledge_chunks(source_id, domain)` — joint retrieval queries
- `conversation_sessions.tenant_id` — multi-tenant query filtering
- `conversation_sessions(user_id, tenant_id)` — user-scoped queries
- `messages.created_at` — pagination and time-range queries
- `audit_traces.timestamp` — time-range audit queries
- `audit_traces(agent_id, timestamp)` — per-agent audit history
- `audit_traces.event_type` — event type filtering

**New constraints:**
- `UNIQUE(tenant_id, name)` on `agents` table
- `UNIQUE(tenant_id, name)` on `knowledge_sources` table
- `UNIQUE(name)` on `tool_definitions` table
- `CHECK(chunk_overlap < chunk_size)` on knowledge upload
- `CHECK(max_retries >= 0)` on tool definitions
- `CHECK(timeout_ms > 0)` on tool definitions
- CASCADE DELETE: knowledge_source deletion triggers vector entry cleanup

**Files:** `server/models/knowledge.py`, `session.py`, `audit.py`, `agent.py`, `workflow.py`
**Migration:** Alembic migration script

### 1.2 Security Hardening

**Critical fixes:**
- Flip `disable_auth` default to `False` in `server/config.py`
- Secret key: fail startup if value is `"change-me-in-production"` in non-dev mode
- Tenant isolation: extract `tenant_id` from JWT auth token, not from request body
- Tool credentials: AES-256 encryption at rest (encrypt/decrypt in tool_gateway.py)
- Rate limiting: per-IP + per-tenant, Redis-backed (slowapi or custom middleware)

**Auth/RBAC system:**
- JWT token authentication replacing SimpleLock
- 3 roles: `admin`, `editor`, `viewer`
- Per-resource permissions (agent, workflow, knowledge CRUD)
- API key management (create/rotate/revoke) via new API + UI
- Startup validation: check required env vars, fail if missing

**New files:** `server/middleware/auth.py`, `server/models/user.py`
**Modified:** `server/config.py`, `server/api/auth.py`, all API route files (add auth dependency)

### 1.3 pgvector Integration

**Strategy:** Auto-detect — PostgreSQL detected = use pgvector automatically. SQLite = FAISS fallback. User can override via `vector_store` config.

**Implementation:**
1. New abstract `VectorStoreAdapter` interface (`server/engine/vector_adapter.py`)
   - Methods: `add(chunk_id, text, domain)`, `add_batch(items)`, `search(query, top_k, domain, ef_search)`, `delete(chunk_ids)`, `rebuild()`
2. `PgVectorStore` implementation (`server/engine/pgvector_store.py`)
   - SQLAlchemy model with `pgvector.sqlalchemy.Vector(dim)` column
   - HNSW index: `m=32, ef_construction=200` (matching existing FAISS HNSW config)
   - Configurable `ef_search` per preset (fast=64, balanced=128, accurate=256)
   - Uses same async session as main DB
3. Config change: `vector_store: Literal["faiss", "pgvector", "milvus"] = "auto"`
   - `"auto"`: PostgreSQL → pgvector, SQLite → FAISS
4. Migration script: read FAISS sidecar → bulk insert into pgvector table
5. Keep existing FAISS adapter as `FaissVectorStore` implementing same interface
6. `VectorStoreManager` factory selects adapter based on config/DB engine

**New files:** `server/engine/vector_adapter.py`, `server/engine/pgvector_store.py`
**Modified:** `server/config.py`, `server/engine/vector_store.py` (refactor to adapter), `server/engine/knowledge_retriever.py`

### 1.4 Error Handling & Observability

**Structured logging:**
- JSON log format (ELK/Splunk compatible) via `python-json-logger`
- Request ID propagation: generate `X-Request-ID` in middleware, inject into all log records
- Replace 7+ silent `except: pass` blocks with specific exception handling + logging

**Exception hierarchy:**
```
HlABError (base)
├── LLMError
│   ├── LLMTimeoutError
│   ├── LLMRateLimitError
│   └── LLMModelError
├── RetrievalError
│   ├── VectorSearchError
│   ├── KeywordSearchError
│   └── FastLookupError
├── WorkflowError
│   ├── WorkflowValidationError
│   ├── WorkflowStepError
│   └── WorkflowEscalationError
└── ToolInvocationError
```

Replace string-matching error classification in `invoke.py` with proper exception type checking.

**New file:** `server/exceptions.py`
**Modified:** All engine files, `server/api/invoke.py`

---

## Phase 2: Engine (Workflow + Agent Runtime)

### 2.1 Workflow Conditional Branching

**The gap:** `next_step_rules` field exists in `WorkflowStep` model and API schemas but is never evaluated in `workflow_executor.py`. The `_advance()` method only does `current_step_index + 1`.

**Rule engine design:**
```json
{
  "next_step_rules": [
    {
      "condition": {
        "field": "user_type",
        "op": "eq",
        "value": "enterprise"
      },
      "goto_step": "collect_business_license"
    },
    {
      "condition": null,
      "goto_step": "complete"
    }
  ]
}
```

**Supported operators:** `eq`, `ne`, `gt`, `lt`, `gte`, `lte`, `contains`, `regex`, `in`, `not_in`

**Implementation:**
- New `server/engine/rule_evaluator.py` — `evaluate_rules(rules, collected_data) -> step_name | None`
- Modify `workflow_executor.py._advance()`: check `step.next_step_rules` before linear progression
- Step resolution: `goto_step` matches by step `name` field or `order` number
- Default fallthrough: rule with `condition: null` is the default path
- Validation: schema check that goto_step references exist in workflow

**Files:** `server/engine/rule_evaluator.py` (new), `server/engine/workflow_executor.py`, `server/schemas/workflow.py`

### 2.2 Workflow Rollback & Versioning

**Rollback handler:**
- Before each step execution, save snapshot to `workflow_state.snapshots[]`
- On `on_failure="rollback"`: restore previous snapshot, undo collected fields from failed step
- Retry from the previous step (not current)
- Max rollback depth: configurable (default 3)

**Version history:**
- New `workflow_versions` table: `workflow_id`, `version`, `snapshot_data` (full workflow + steps JSON), `published_at`, `published_by`
- Active sessions pin to version number at workflow start
- API: `GET /workflows/{id}/versions`, `POST /workflows/{id}/publish`, `POST /workflows/{id}/rollback/{version}`
- UI: version list with diff view

**Files:** `server/engine/workflow_executor.py`, `server/models/workflow.py` (new table), `server/api/workflows.py`

### 2.3 Agent Runtime Improvements

**Context propagation on delegation:**
- Delegation carries last N messages (configurable, default 5) as `context_messages` in the delegated request
- RAG citations forwarded to delegated agent's system context
- New `parent_session_id` field on `ConversationSession` — links delegation chain
- Shared key-value memory: `session.shared_context` dict accessible across agent chain

**Circuit breaker:**
- Track failure rate per LLM provider / tool endpoint
- States: closed → open (after 5 failures in 60s) → half-open (probe every 30s)
- Prevents cascade failures when external service is down
- New `server/engine/circuit_breaker.py`

**Files:** `server/engine/agent_runtime.py`, `server/engine/circuit_breaker.py` (new), `server/models/session.py`

### 2.4 Configurable Limits (Magic Numbers → Config)

| Current Location | Value | New Location |
|---|---|---|
| `agent_runtime.py:44` | MAX_TOOL_ROUNDS=5 | Agent model `max_tool_rounds` field |
| `agent_runtime.py:47` | MAX_DELEGATION_DEPTH=3 | runtime_config |
| `workflow_executor.py:25` | MAX_AUTO_ADVANCE=20 | Workflow model field |
| `knowledge_retriever.py:130` | RRF k=60 | runtime_config `rrf_k` |
| `knowledge_retriever.py:434` | timeout=10000ms | runtime_config (already partial) |
| `workflow_executor.py:384` | webhook timeout=30s | Workflow step config |
| `knowledge.py:165` | ALLOWED_EXTENSIONS | config.py |
| `vector_store.py:343` | over-fetch 3x | runtime_config |
| `knowledge_retriever.py:493` | reranker over-fetch 2x | runtime_config |
| `invoke.py` | SSE queue unbounded | config maxsize=1000 |
| `agent_runtime.py:429` | history limit=6 | Agent model field |

All configurable via API (`POST /performance/update-config`) and Settings UI.

---

## Phase 3: UX Overhaul

### 3.1 Visual Workflow Builder

**Library:** React Flow (reactflow.dev) — MIT licensed, drag-and-drop nodes + edges

**Custom node types:**
- `CollectNode` — data collection step with field list
- `ToolCallNode` — external tool invocation
- `DisplayNode` — display information to user
- `HumanReviewNode` — pause for human review
- `CompleteNode` — workflow completion + webhook

**Features:**
- Edge labels: conditional rules rendered as readable text
- Side panel: click node → edit fields via form (no JSON)
- Field builder: add/remove fields visually, select type from dropdown, toggle required
- Live preview: test workflow in split-screen playground
- Serialization: React Flow graph ↔ existing workflow API models (bidirectional)

**New components:** `WorkflowCanvas.tsx`, `WorkflowNode.tsx` (per type), `WorkflowEdge.tsx`, `FieldBuilder.tsx`

### 3.2 Agent Setup Wizard

**5-step guided creation:**
1. **Basic Info** — name, description, system prompt (with template selection)
2. **LLM Config** — choose provider type (Cloud/Local/Custom), only relevant fields shown
3. **Knowledge** — select/create knowledge sources, preview retrieval
4. **Tools & Workflows** — bind tools and workflows, configure capabilities
5. **Test & Deploy** — inline agent testing before enabling

**Agent templates:**
- "Customer Service" — pre-filled system prompt, FAQ knowledge type, escalation workflow
- "FAQ Bot" — knowledge-focused, minimal tools
- "Data Collection" — workflow-focused, form collection agent
- "Multi-Agent Router" — delegation-focused, connects to sub-agents

**Features:**
- Progressive disclosure: show only relevant fields per step
- Validation at each step: can't proceed until valid
- Skip optional steps (knowledge, tools)
- Template auto-fills all fields, user can customize

**New components:** `AgentWizard.tsx`, `WizardStep.tsx`, `TemplateSelector.tsx`

### 3.3 Component Architecture Refactor

**Split PlaygroundPage (1184 lines):**
- `PlaygroundPage.tsx` — slim orchestrator
- `components/ChatMessage.tsx` — message rendering
- `components/TracePanel.tsx` — trace visualization
- `components/WorkflowForm.tsx` — workflow field collection
- `hooks/usePlayground.ts` — state management + SSE logic

**New shared infrastructure:**
- `types/` directory — Agent, Workflow, Knowledge, Tool, Skill, LLMConfig types (kill all `any`)
- `hooks/useApi.ts` — error handling, retry, cache wrapper around api.ts
- `hooks/useForm.ts` — form state management with validation
- `components/FormModal.tsx` — shared modal pattern with validation
- `components/DataTable.tsx` — shared table with search/filter/sort
- `components/LoadingState.tsx` — skeleton, empty, error states

**Fully typed api.ts:**
- Define request/response interfaces for every endpoint
- Generic typed API methods: `api.get<AgentResponse>(url)`
- Global error interceptor with retry logic

### 3.4 Form Validation & i18n

**Validation on all forms:**
- Required + length + format on all fields
- URL fields: regex validation
- JSON fields: live syntax highlighting + validation
- Async validation: name uniqueness checks
- Cross-field validation: overlap < chunk_size, etc.
- Inline error messages (under fields, not toasts)

**Loading/empty/error states:**
- Loading: skeleton placeholders (not spinner)
- Empty: illustrated empty state + action button ("Create your first agent")
- Error: message + retry button + context
- Unsaved changes: warning on navigation

**i18n:**
- Library: `react-i18next`
- Default: Chinese
- Supported: Chinese + English
- Language switcher in header
- All hardcoded strings extracted to translation files
- Ant Design locale provider integration

---

## Phase 4: Polish & Monitoring

### 4.1 Dashboard Upgrade
- Historical metrics charts (7d/30d trends) via lightweight charting (recharts)
- Per-agent performance breakdown
- RAG hit rate visualization
- LLM cost tracking per agent
- Circuit breaker status widgets
- Real-time SSE push (replace 60s polling)

### 4.2 Operations
- Bulk enable/disable agents
- Import/export agent configs (JSON)
- Agent cloning (duplicate with new name)
- Knowledge deduplication tool
- Audit log advanced filters + CSV export
- System health page (DB, Redis, LLM, vector store status)

---

## Verification Plan

### Phase 1 Verification
- [ ] Run Alembic migration on fresh DB + existing DB with data
- [ ] `EXPLAIN ANALYZE` on key queries to confirm index usage
- [ ] Auth enabled: verify all endpoints require valid JWT
- [ ] RBAC: verify viewer cannot create agents, editor can, admin can manage users
- [ ] pgvector: add 1000 vectors, search, verify results match FAISS output
- [ ] pgvector auto-detect: PostgreSQL DB → pgvector selected; SQLite → FAISS selected
- [ ] Error logging: trigger each exception type, verify JSON log output
- [ ] Rate limiting: exceed limit, verify 429 response

### Phase 2 Verification
- [ ] Workflow branching: create workflow with 3 branches, verify correct path taken for each condition
- [ ] Workflow rollback: trigger step failure with on_failure="rollback", verify previous step restored
- [ ] Workflow versioning: publish v2, start session on v1, verify v1 still executes
- [ ] Agent delegation: verify delegated agent receives last 5 messages as context
- [ ] Circuit breaker: kill LLM endpoint, verify circuit opens after 5 failures, verify half-open probe
- [ ] Configurable limits: change `max_tool_rounds` via API, verify new value takes effect

### Phase 3 Verification
- [ ] Visual workflow builder: create a 5-step workflow with branching using only mouse clicks
- [ ] Agent wizard: create agent using "Customer Service" template, test in playground
- [ ] Type safety: `npx tsc --noEmit` passes with zero `any` types in new code
- [ ] Form validation: submit invalid data on every form, verify inline error shown
- [ ] i18n: switch to English, verify all text translated on every page
- [ ] `npx vite build` succeeds with no errors

### Phase 4 Verification
- [ ] Dashboard: verify charts render with 30 days of test data
- [ ] Bulk operations: select 10 agents, disable all at once
- [ ] Import/export: export agent, delete it, import back, verify identical config
- [ ] Health page: kill Redis, verify health page shows Redis as unhealthy

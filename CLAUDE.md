# HlAB - Headless AI Agent Builder

> **Quick Context**: Enterprise AI Agent platform with multi-tenancy, workflows, RAG, tool calling. Full backend + frontend console.

---

## Code Cleanup Policy

**All contributors must follow these rules:**

1. **No dead imports** - Remove any import that is not used in the file. Run builds to verify.
2. **No dead functions** - If a function is not called anywhere, delete it entirely. Do not leave commented-out code.
3. **No TODO placeholders in shipped code** - If a feature is not implemented, either implement it or remove the placeholder. Never ship `// TODO` in production paths.
4. **Centralized API client** - All frontend HTTP calls MUST go through `console/src/api.ts`. Never use raw `axios` in page components. Add new methods to the centralized client first.
5. **No duplicate logic** - If two functions do the same thing, keep one and delete the other.
6. **No stale documentation** - If code changes, update CLAUDE.md and relevant docs. Documentation that contradicts the code is worse than no documentation.
7. **Build must pass** - Every change must result in a clean `npx vite build` with no errors. Warnings about chunk size are acceptable.

---

## Project Overview

**Production-grade AI Agent platform** for enterprise/private deployment:
- Multi Agent management with independent config and knowledge domain isolation
- **Conversation-first architecture**: LLM drives the conversation naturally; skills are exposed as function-calling tools. No explicit router/classifier layer — the LLM decides when to call tools (knowledge search, HTTP tools, workflows, delegation) based on context.
- **Capabilities API**: Auto-manage skills from Agent page — users configure knowledge/workflows/tools directly, system creates Skills behind the scenes (`managed_by = "agent:{id}"`)
- **Multi-Agent collaboration**: Agent delegation via `delegate` skills with depth/cycle protection
- **Multi-workflow support**: One agent can bind multiple workflows, LLM auto-selects via function calling
- Workflow orchestration with sequential steps, field validation, tool calling, **file upload**, **LLM-assisted validation**, **interruptible workflows**
- RAG knowledge base with three-channel retrieval (KV + Vector + BM25) + RRF fusion + optional cross-encoder reranking + PDF/DOCX/TXT upload
- HTTP tool integration with exponential backoff retry
- Full audit trail with call chain replay
- Web console with React + Ant Design

---

## Architecture

```
HlAB-headlessagentbuilder/
├── server/                       # FastAPI backend
│   ├── api/                      # 70+ API endpoints (all implemented)
│   │   ├── invoke.py             # Core invoke + SSE streaming
│   │   ├── agents.py             # Agent CRUD
│   │   ├── skills.py             # Skill CRUD
│   │   ├── agent_skills.py       # Agent-Skill binding
│   │   ├── agent_capabilities.py # Auto-manage skills via capabilities API
│   │   ├── agent_connections.py  # Inter-agent connections
│   │   ├── workflows.py          # Workflow management
│   │   ├── knowledge.py          # Knowledge base + document upload
│   │   ├── tools.py              # Tool management
│   │   ├── audit.py              # Audit logs
│   │   ├── llm_configs.py        # LLM config CRUD + templates + test
│   │   ├── performance.py        # Performance presets + runtime config
│   │   ├── vector_admin.py       # Vector index rebuild/migrate/stats
│   │   ├── mock_tools.py         # Dev-only mock tool endpoints
│   │   └── auth.py               # API key authentication
│   ├── engine/                   # Core engine
│   │   ├── agent_runtime.py      # Agent execution: conversational pipeline + skill-tools
│   │   ├── workflow_executor.py  # Workflow executor + recursion guard
│   │   ├── knowledge_retriever.py # Hybrid retrieval: Exact + Vector + Keyword + RRF fusion
│   │   ├── llm_adapter.py        # LLM adapter (OpenAI/DashScope/ZhipuAI/Ollama)
│   │   ├── tool_executor.py      # Tool execution + exponential backoff
│   │   └── vector_store.py       # FAISS vector store + embedding cascade fallback
│   ├── models/                   # Database models (10 tables)
│   ├── schemas/                  # Pydantic request/response schemas
│   ├── performance_presets.py    # fast/balanced/accurate presets
│   └── config.py                 # Environment variable config
├── console/                      # React frontend console
│   └── src/
│       ├── pages/                # 10 management pages
│       │   ├── DashboardPage.tsx  # Dashboard with metrics
│       │   ├── PlaygroundPage.tsx  # Agent chat test + call trace + perf config
│       │   ├── AgentsPage.tsx     # Agent CRUD + skill binding + agent connections
│       │   ├── SkillsPage.tsx     # Skill CRUD + dynamic type-based config
│       │   ├── WorkflowsPage.tsx  # Workflow + step management
│       │   ├── KnowledgePage.tsx   # Knowledge sources + RAG config panel
│       │   ├── ToolsPage.tsx      # Tool CRUD + connectivity test
│       │   ├── LLMConfigsPage.tsx # LLM config + provider templates
│       │   ├── SettingsPage.tsx   # Performance presets + advanced tuning
│       │   ├── AuditPage.tsx      # Audit trace replay
│       │   └── SimpleLock.tsx     # Simple password lock
│       └── api.ts                # Centralized API client (all endpoints)
├── deploy-quick.sh               # One-click deployment
├── verify-all.sh                 # Feature verification
└── check-production-ready.sh     # Production readiness check
```

---

## Tech Stack

### Backend
- **Framework**: FastAPI + SQLAlchemy (async) + Pydantic v2
- **Database**: SQLite (dev) / PostgreSQL (production)
- **Cache**: Redis
- **Vector Store**: FAISS / Milvus
- **LLM**: OpenAI-compatible / DashScope / ZhipuAI / Local Ollama
- **Embedding**: sentence-transformers (local) / DashScope API

### Frontend
- **Framework**: React 18 + TypeScript
- **UI**: Ant Design v5
- **HTTP**: Axios (centralized in api.ts)
- **Build**: Vite

---

## Quick Start

```bash
# Clone and start
git clone <repo> && cd HlAB-headlessagentbuilder
bash deploy-quick.sh

# Access
open http://localhost:8000
```

---

## Current State

### Completed

| Module | Status | Details |
|--------|--------|---------|
| Backend API | 70+ endpoints | Including SSE streaming, performance config, vector admin, skills, agent connections |
| Database | 13 tables | Full ORM with async SQLAlchemy (added skills, agent_skills, agent_connections) |
| Skill Architecture | Complete | Conversation-first: skills exposed as function-calling tools; LLM decides when to invoke |
| Multi-Agent | Complete | Agent delegation with depth (max 3) and cycle protection |
| RAG | Hybrid 3-channel + RRF | Exact KV (<50ms), Vector HNSW (<200ms), jieba BM25 keyword, RRF fusion (k=60), optional cross-encoder reranker |
| Workflow Engine | Complete | Executor + validator + recursion guard + conditional branching + cancel/exit + file upload + LLM validation |
| Conversational Pipeline | Complete | LLM-driven with pre-retrieval + function calling; no explicit intent router |
| Tool Calling | Complete | HTTP tools + exponential backoff + LLM native function calling |
| Audit Logs | Complete | Full call chain with 7-step traces |
| Capabilities API | Complete | Auto-manage skills from Agent page; GET/PUT /agents/{id}/capabilities; cascade delete |
| Frontend Console | 11 pages | Dashboard, Playground, Agents (tabbed capabilities), Skills (filtered), Workflows, Knowledge (chunk viewer), Tools, LLM Configs, Settings, Audit, Lock |
| RAG Config UI | Complete | Inline presets, embedding model selector, HNSW params, reranker, intent detection |
| Deployment | Complete | One-click deploy script |
| SSE Streaming | Complete | POST /invoke/stream endpoint |

---

## Key Files

### Backend
| File | Purpose |
|------|---------|
| `server/main.py` | FastAPI entry + CORS + router registration |
| `server/config.py` | Environment variable config |
| `server/api/invoke.py` | Core invoke API (POST /invoke + POST /invoke/stream SSE) |
| `server/engine/agent_runtime.py` | Agent execution: conversation-first pipeline, `_build_skill_tools()` converts skills to function defs, multi-round function calling loop |
| `server/engine/knowledge_retriever.py` | RAG three-channel retrieval (fast KV + vector + BM25) + RRF fusion + cross-encoder reranker |
| `server/engine/llm_adapter.py` | LLM adapter + function calling (chat_with_tools) |
| `server/api/agent_capabilities.py` | GET/PUT capabilities API (auto-manages skills) |
| `server/engine/vector_store.py` | FAISS IndexHNSWFlat vector store + bge-m3 embedding with instruction prefix + cascade fallback |
| `server/performance_presets.py` | fast/balanced/accurate preset definitions |

### Frontend
| File | Purpose |
|------|---------|
| `console/src/App.tsx` | Route config (no raw axios - uses page components) |
| `console/src/api.ts` | Centralized API client (agents, agentCapabilities, skills, agentSkills, agentConnections, workflows, knowledge, tools, audit, invoke, llmConfig, performance, vectorAdmin) |
| `console/src/pages/AgentsPage.tsx` | Agent CRUD + tabbed modal (Basic Info / Capabilities / Advanced) + auto-managed skills |
| `console/src/pages/SkillsPage.tsx` | Skill CRUD + dynamic type-based config forms |
| `console/src/pages/PlaygroundPage.tsx` | Agent chat + SSE streaming + function calling chain trace |
| `console/src/pages/KnowledgePage.tsx` | Knowledge management + RAG config panel (presets, embedding, HNSW, reranker) |
| `console/src/pages/LLMConfigsPage.tsx` | LLM config CRUD + provider templates (uses llmConfigApi) |
| `console/src/pages/SettingsPage.tsx` | Global performance presets + advanced parameter tuning |

---

## Core API

### Invoke Agent (sync)
```bash
POST /api/v1/invoke
{"agent_id": "example_agent", "message": "Hello", "tenant_id": "default"}
```

### Invoke Agent (SSE streaming)
```bash
POST /api/v1/invoke/stream
{"agent_id": "example_agent", "message": "Hello", "tenant_id": "default"}
# Returns: event: status → event: answer → event: done
```

### Upload Document (RAG)
```bash
POST /api/v1/knowledge/upload
FormData: {file, source_id, domain, chunk_size, embedding_model, reranker_enabled}
```

### Agent Capabilities (auto-manage skills)
```bash
GET  /api/v1/agents/{agent_id}/capabilities     # Get knowledge/workflow/tool capabilities
PUT  /api/v1/agents/{agent_id}/capabilities     # Upsert capabilities (auto-creates/updates/deletes skills)
```

### Performance Config
```bash
GET  /api/v1/performance/current-config    # Get current runtime config
POST /api/v1/performance/presets/apply      # Apply preset (fast/balanced/accurate)
POST /api/v1/performance/update-config      # Update custom config (immediate effect)
```

---

## Bug Fix History

### Fixed
- CORS hardcoded -> configurable
- Database init failure -> auto-create directory
- `vector_admin.py` wrong import path -> `server.db`
- `VectorStoreManager` wrong kwargs -> fixed
- Tool retry linear -> exponential backoff `2^attempt`
- Workflow executor infinite recursion -> `MAX_AUTO_ADVANCE_DEPTH=20`
- LLM adapter `import json` inside loop -> module level
- LLM adapter singleton no lock -> `threading.Lock` double-check
- No streaming endpoint -> SSE `POST /invoke/stream`
- Playground not implemented -> 820-line implementation + call trace
- Agent page incomplete -> full LLM/tool/knowledge/template config
- RAG keyword search broken (SQL LIKE full query) -> n-gram keyword extraction + OR conditions
- Document chunks invisible to fast channel -> auto-generate entity_key from first sentence
- Settings page TODO placeholder -> fully wired to performanceApi
- LLMConfigsPage raw axios -> centralized llmConfigApi
- Dead `get_llm_adapter_from_db()` function -> removed
- Unused `Column` import in llm_config model -> removed
- Unused `axios` import in App.tsx -> removed
- Workflow delete had useless SELECT -> removed
- Agent runtime: undefined `llm_resp` in for-else -> safe None guard
- Agent runtime: unused imports (`selectinload`, `WorkflowCard`, `ToolCallRequest`) -> removed
- Agent runtime: `max_tool_rounds` not validated -> clamped to [1, 20]
- LLM adapter: streaming `json.loads` unguarded -> try/except + safe dict access
- Vector store: singleton not thread-safe -> double-check locking with `threading.Lock`
- Workflow executor: dead `load_workflow()` method -> removed; unused `selectinload` import -> removed
- Knowledge upload: path traversal via file extension -> allowlist sanitization
- Knowledge upload: no validation on `chunk_size`/`chunk_overlap` -> Form constraints + overlap < size check
- Auth.py: TODO placeholder code -> removed
- LLM configs API: unused `require_auth` import -> removed
- Workflows API: redundant circular `selectinload(WorkflowStep.workflow)` -> removed
- Frontend: all delete handlers missing try/catch -> added error messages
- Frontend: destructive deletes without confirmation -> Popconfirm on all delete buttons
- PlaygroundPage: fragile SSE parsing (indexOf in for-of loop) -> proper event/data pair accumulation
- AuditPage: silent catch blocks -> user-visible error messages
- KnowledgePage: handleSearch/handleOpenChunks/handleUpdateChunk unguarded -> try/catch added
- QA pipeline: `get_vector_store()` crash kills entire pipeline -> graceful fallback to fast channel only
- QA pipeline: retriever.retrieve() crash unhandled -> empty RetrievalResponse fallback
- QA pipeline: LLM failure with KV data available -> return fast_answer directly (retrieval-only mode)
- Agent runtime: generic "系统暂时无法响应" -> classified errors (connection/timeout/model/rate limit)
- SSE error event: only `detail` field -> added `error_type` and `error_msg` for debugging
- Sync invoke error: same generic message -> classified error messages with error_detail in metadata
- Greeting handler: LLM failure crashes greeting -> canned response fallback
- Function calling path: LLM init/call failure unhandled -> retrieval-only fallback + classified errors
- Vector search: embedding model load failure crashes retriever -> try/except returns empty results
- PlaygroundPage: SSE error throws and loses context -> error shown in trace panel, content preserved
- Fast lookup: raw query matching fails for KV entries -> bidirectional keyword matching + reverse containment
- Retrieval pipeline: sequential channels -> concurrent asyncio.gather for all 3 channels
- Keyword search: simple overlap counting -> BM25-style scoring (TF saturation + length normalization)
- Vector store: single model fails = no vector search -> cascade fallback (bge-m3 -> MiniLM-L6 -> multilingual)
- Retrieval fusion: no fusion between channels -> RRF (Reciprocal Rank Fusion, k=60, weighted)
- LLM says "无此数据" when retrieval found results -> refusal override safety net
- Pipeline timeout: no limit -> 90s asyncio.wait_for on entire invoke pipeline
- LLM timeout: hardcoded 60s -> configurable 30s via HLAB_LLM_TIMEOUT
- Dead `rag_search()` method -> removed (replaced by hybrid pipeline in retrieve())
- LLM refusal override incomplete ("没有提供" not caught) -> expanded to 18+ refusal phrases
- FAST_ANSWER_PROMPT too permissive -> explicit "禁止说没有" + must-use-data instructions
- Dead `conditions` variable in fast_lookup -> removed
- KV exact match (score=1.0) still goes through LLM (which may refuse) -> bypass LLM entirely, return data directly
- Function-calling path missing refusal override -> added same override as standard QA path
- LLM adapter: Qwen3/DeepSeek-R1 reasoning models return empty `content` → extract answer from `reasoning` field
- Agent runtime: forced JSON output format breaks small models → natural language prompts (JSON optional/backward-compatible)
- Agent runtime: multi-turn follow-ups ("它有哪些应用") lose context → query rewriting with conversation history
- Agent runtime: empty LLM response after tool calling → fallback to retrieval or tool result summary
- LLM adapter: streaming doesn't handle reasoning models → buffer reasoning, fallback to extracted content
- LLM timeout: 30s too tight for reasoning models → default 60s
- Agent runtime: no auto-generated followups when LLM doesn't provide them → contextual followup generation
- Intent routing: hardcoded keyword-based `_detect_intent()` only recognizes greetings → LLM-driven `_classify_intent()` with hybrid fast-path + LLM fallback
- Agent model: single `workflow_id` forces ALL messages into workflow → `workflow_scope` multi-workflow support with intent-based routing
- Workflow: no way to exit mid-process → `_WORKFLOW_EXIT_KEYWORDS` + `cancel_workflow()` with data preservation
- Workflow: no file upload in collect steps → `field_type="file"` with `file_config` (extension/size validation)
- Workflow: no semantic validation → `llm_validate` + `llm_validate_prompt` for LLM-assisted field validation
- Agent runtime: no chitchat handling → `_handle_chitchat()` for casual conversation without retrieval
- AgentsPage: single workflow dropdown → multi-workflow Select with per-workflow description inputs
- AgentsPage: dual-system confusion (legacy scope fields + skill routing coexist) → skill-only UI; Agent page shows basic info + skill binding + connections; legacy scope fields removed from UI (API still accepts them for backward compatibility)
- Agent model/schema: `skill_routing_mode` default → "conversational"; conversation-first pipeline replaces SkillRouter/SkillExecutor; skills exposed as function-calling tools
- KnowledgePage: no way to view chunks/entries in a knowledge source → chunk viewer modal with clickable source name + View button
- AgentsPage: required 5+ steps across 3 pages to configure agent → single-page tabbed modal (Basic Info / Capabilities / Advanced) with auto-managed skills
- SkillsPage: auto-managed skills cluttered the list → filtered out with `managed_by=null` query param + info banner
- Agent delete: orphaned auto-managed skills left behind → cascade delete of managed skills + AgentSkill bindings
- Dead code: `skill_router.py` (362 lines) + `skill_executor.py` (300 lines) → deleted (zero references project-wide)
- Agent model: deprecated fields `workflow_id`, `workflow_scope`, `knowledge_scope`, `tool_scope` → removed from model, schemas, and API
- Agent runtime: `get_vector_store` imported inside closures → moved to module-level import
- Agent runtime: duplicate retrieval (pre-retrieval + LLM tool call) → knowledge tool description annotated when pre-retrieval succeeded
- AgentsPage: single-option routing mode dropdown → removed (hardcoded "conversational")
- AgentsPage: Knowledge QA redundant domain selector → removed; auto-inferred from selected sources
- AgentsPage: Description fields too small → TextArea with full-width layout
- SkillsPage: trigger_config UI (unused in conversational mode) → removed entirely
- SkillsPage: Chinese labels inconsistent with English AgentsPage → all labels converted to English
- PlaygroundPage: trace panel fixed 380px → 340px responsive with auto-hide below 900px
- PlaygroundPage: missing event colors for conversational mode → added conversational_init, pre_retrieval, query_rewrite
- RAG keyword search: SQL LIKE + hardcoded score=0.5 → jieba tokenization + real BM25 scoring (k1=1.5, b=0.75, IDF+TF normalization)
- RAG fusion: simple sort by score → Reciprocal Rank Fusion (k=60, weighted vector=1.0/keyword=configurable)
- RAG channels: sequential execution → concurrent asyncio.gather with return_exceptions
- RAG reranker: config existed but never wired → cross-encoder Reranker class (bge-reranker-v2-m3) activated by reranker_enabled
- FAISS index: IndexFlatIP brute force → IndexHNSWFlat (M=32, efConstruction=200, configurable efSearch)
- Embedding: bge-small-zh-v1.5 default → bge-m3 (1024-dim) with instruction prefix for queries
- Embedding cascade: 3 models → 4 models (bge-m3 → bge-small-zh → MiniLM-L6 → multilingual-MiniLM)
- Vector store: no dimension mismatch detection → auto-detect + create fresh HNSW index + log rebuild prompt
- runtime_config: singleton existed but engine never read it → wired into retriever (keyword_weight, ef_search, reranker_enabled, retrieval_timeout_ms, retrieval_top_k)
- Performance presets: missing ef_search → added per-preset ef_search (fast=64, balanced=128, accurate=256)
- main.py lifespan: no default preset → initialize balanced preset + jieba.initialize() on startup
- Document upload: only .txt/.md → added .pdf (pypdf) and .docx (python-docx) support
- Chunking: simple paragraph split → recursive splitting (paragraph → line → sentence → character) with jieba sentence boundary detection
- Retrieval timeout: no timeout → configurable asyncio.wait_for wrapping entire retrieve() pipeline

### Known Limitations
- API Key stored as plaintext in DB (production: encrypt)
- Frontend auth is simple password lock (production: replace with JWT/OAuth)

---

## Configuration

### Production Required (3 items)
```bash
HLAB_CORS_ORIGINS=https://yourdomain.com
HLAB_DISABLE_AUTH=false
HLAB_API_KEY=<random-64-chars>
```

### Common Config
```bash
HLAB_DATABASE_URL=postgresql+asyncpg://user:pass@host/hlab
HLAB_LLM_PROVIDER=openai_compatible|dashscope|zhipu|local
HLAB_LLM_BASE_URL=http://localhost:11434/v1
HLAB_LLM_MODEL=qwen2.5
HLAB_VECTOR_STORE=faiss|milvus
```

---

## Database Models

| Table | Purpose | Key Fields |
|-------|---------|-----------|
| agents | Agent config | id, name, system_prompt, skill_routing_mode (default "conversational"), response_config, risk_config |
| skills | Skill definitions | id, name, skill_type, trigger_config, execution_config, priority, managed_by |
| agent_skills | Agent-Skill bindings | agent_id, skill_id, priority_override, config_override |
| agent_connections | Inter-agent connections | source_agent_id, target_agent_id, connection_type, shared_context |
| workflows | Workflow definitions | id, name, mode, version |
| workflow_steps | Workflow steps | workflow_id, step_type, tool_id, on_failure |
| knowledge_sources | Knowledge sources | id, domain, source_type, embedding_model |
| knowledge_chunks | Knowledge chunks | source_id, content, entity_key, domain |
| tool_definitions | Tool definitions | id, name, endpoint_url, method |
| conversation_sessions | Chat sessions | id, agent_id, tenant_id, active_skill_id, delegation_chain |
| messages | Message records | session_id, role, content |
| audit_traces | Audit traces | trace_id, session_id, event_type, event_data |
| llm_configs | LLM configurations | id, name, provider, base_url, model, is_default |

---

## Future Optimization

1. ~~Hybrid BM25 + Vector retrieval (RRF fusion)~~ ✅ Implemented (jieba BM25 + HNSW vector + RRF k=60)
2. ~~Cross-encoder reranking~~ ✅ Implemented (bge-reranker-v2-m3, activated in accurate preset)
3. ~~HNSW vector index~~ ✅ Implemented (IndexHNSWFlat M=32, efConstruction=200, configurable efSearch)
4. ~~PDF/DOCX document upload~~ ✅ Implemented (pypdf + python-docx)
5. ~~Recursive text chunking~~ ✅ Implemented (paragraph → line → sentence with jieba boundaries)
6. ~~Runtime config wiring~~ ✅ Implemented (presets → engine parameters)
7. Query Expansion / HyDE query enhancement
8. Unit test coverage
9. Prometheus monitoring integration
10. SPLADE learned sparse retrieval (upgrade from BM25)
11. ColBERT-style late interaction for precision-critical domains

---

**Last Updated**: 2026-02-27
**Status**: Production ready
**Core Strengths**: Complete features, optimized RAG pipeline, one-click deploy, centralized API client, Headless API + SSE streaming

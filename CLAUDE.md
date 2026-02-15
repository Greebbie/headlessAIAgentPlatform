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
- **LLM-driven intent routing**: Hybrid fast-path + LLM classification (greeting/chitchat/knowledge_query/workflow_start/tool_use)
- **Multi-workflow support**: One agent can bind multiple workflows, LLM auto-routes to correct one
- Workflow orchestration with sequential steps, field validation, tool calling, **file upload**, **LLM-assisted validation**, **interruptible workflows**
- RAG knowledge base with dual-channel retrieval (KV + Vector) + keyword search + document upload
- HTTP tool integration with exponential backoff retry
- Full audit trail with call chain replay
- Web console with React + Ant Design

---

## Architecture

```
HlAB-headlessagentbuilder/
├── server/                       # FastAPI backend
│   ├── api/                      # 57 API endpoints (all implemented)
│   │   ├── invoke.py             # Core invoke + SSE streaming
│   │   ├── agents.py             # Agent CRUD
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
│   │   ├── agent_runtime.py      # Agent execution logic
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
│       │   ├── AgentsPage.tsx     # Agent CRUD + LLM/tool/knowledge binding
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
| Backend API | 57 endpoints | Including SSE streaming, performance config, vector admin |
| Database | 10 tables | Full ORM with async SQLAlchemy |
| RAG | Hybrid 3-channel + RRF | Exact KV (<50ms), Vector semantic (<200ms), BM25-style keyword, RRF fusion |
| Workflow Engine | Complete | Executor + validator + recursion guard + conditional branching + cancel/exit + file upload + LLM validation |
| Intent Routing | Complete | Hybrid fast-path + LLM classification, multi-workflow support, chitchat handling |
| Tool Calling | Complete | HTTP tools + exponential backoff + LLM native function calling |
| Audit Logs | Complete | Full call chain with 7-step traces |
| Frontend Console | 10 pages | Dashboard, Playground, Agents, Workflows, Knowledge, Tools, LLM Configs, Settings, Audit, Lock |
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
| `server/engine/agent_runtime.py` | Agent execution + LLM function calling loop |
| `server/engine/knowledge_retriever.py` | RAG dual-channel retrieval + keyword n-gram search |
| `server/engine/llm_adapter.py` | LLM adapter + function calling (chat_with_tools) |
| `server/engine/vector_store.py` | FAISS vector store + health status endpoint |
| `server/performance_presets.py` | fast/balanced/accurate preset definitions |

### Frontend
| File | Purpose |
|------|---------|
| `console/src/App.tsx` | Route config (no raw axios - uses page components) |
| `console/src/api.ts` | Centralized API client (agents, workflows, knowledge, tools, audit, invoke, llmConfig, performance, vectorAdmin) |
| `console/src/pages/AgentsPage.tsx` | Agent CRUD + LLM/tool/knowledge binding + function calling config |
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
| agents | Agent config | id, name, system_prompt, knowledge_scope, tool_scope, workflow_scope, llm_config_id |
| workflows | Workflow definitions | id, name, mode, version |
| workflow_steps | Workflow steps | workflow_id, step_type, tool_id, on_failure |
| knowledge_sources | Knowledge sources | id, domain, source_type, embedding_model |
| knowledge_chunks | Knowledge chunks | source_id, content, entity_key, domain |
| tool_definitions | Tool definitions | id, name, endpoint_url, method |
| conversation_sessions | Chat sessions | id, agent_id, tenant_id |
| messages | Message records | session_id, role, content |
| audit_traces | Audit traces | trace_id, session_id, event_type, event_data |
| llm_configs | LLM configurations | id, name, provider, base_url, model, is_default |

---

## Future Optimization

1. ~~Hybrid BM25 + Vector retrieval (RRF fusion)~~ ✅ Implemented
2. Query Expansion / HyDE query enhancement
3. Unit test coverage
4. Prometheus monitoring integration
5. SPLADE learned sparse retrieval (upgrade from BM25-style)
6. ColBERT-style late interaction for precision-critical domains

---

**Last Updated**: 2026-02-15
**Status**: Production ready
**Core Strengths**: Complete features, optimized RAG pipeline, one-click deploy, centralized API client, Headless API + SSE streaming

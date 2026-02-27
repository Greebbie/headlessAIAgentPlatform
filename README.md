# HlAB — Headless AI Agent Builder

A self-hosted AI Agent platform with a web console. Build, configure, and deploy conversational AI agents that can answer questions from knowledge bases, execute multi-step workflows, call external APIs, and delegate to other agents — all through a simple chat interface or headless API.

**No cloud lock-in.** Runs with any OpenAI-compatible LLM — local Ollama, DashScope (Qwen), vLLM, or any provider.

---

## What It Does

- **Multi-Agent Management** — Create multiple agents, each with their own system prompt, knowledge base, and tools
- **Knowledge Base (RAG)** — Upload TXT/PDF/DOCX documents. The system chunks, embeds, and retrieves them using hybrid search (BM25 + vector + exact match) with RRF fusion
- **Workflow Engine** — Define multi-step business processes (e.g., repair tickets, applications). The LLM automatically triggers them when users express intent
- **Tool Calling** — Connect external HTTP APIs as tools. Agents call them via native function calling
- **Multi-Agent Delegation** — Agents can delegate to other agents with depth and cycle protection
- **LLM Configuration** — Manage multiple LLM providers (Ollama, DashScope, OpenAI, vLLM, ZhipuAI) and assign them per-agent
- **Web Console** — 10-page React admin console for managing everything
- **Headless API** — Every feature is accessible via REST API + SSE streaming

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend dev only)

### 1. Clone and set up

```bash
git clone https://github.com/your-repo/headlessAIAgentPlatform.git
cd headlessAIAgentPlatform
```

### 2. Create virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Apple Silicon note**: If `faiss-cpu` fails, run `pip install faiss-cpu --no-cache-dir`.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your LLM provider settings:

```bash
# Option A: Local Ollama (free, no API key needed)
HLAB_LLM_PROVIDER=openai_compatible
HLAB_LLM_BASE_URL=http://localhost:11434/v1
HLAB_LLM_MODEL=qwen2.5

# Option B: DashScope (Alibaba Cloud)
HLAB_LLM_PROVIDER=dashscope
HLAB_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
HLAB_LLM_API_KEY=sk-your-key-here
HLAB_LLM_MODEL=qwen-flash
```

### 4. Start the server

```bash
source .venv/bin/activate
HLAB_DISABLE_AUTH=true python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### 5. Open the console

If `static/` directory exists (pre-built frontend):
```
http://localhost:8000
```

For frontend development:
```bash
cd console
npm install
npm run dev
# Opens at http://localhost:3000, proxies API to :8000
```

---

## One-Click Deploy

```bash
bash deploy-quick.sh
```

This script creates the venv, installs dependencies, builds the frontend, and starts the server.

---

## Project Structure

```
headlessAIAgentPlatform/
├── server/                    # FastAPI backend
│   ├── api/                   # REST API endpoints (70+)
│   ├── engine/                # Core runtime
│   │   ├── agent_runtime.py   # Conversation pipeline + function calling
│   │   ├── knowledge_retriever.py  # Hybrid RAG (BM25 + vector + exact)
│   │   ├── llm_adapter.py     # Multi-provider LLM client
│   │   ├── workflow_executor.py    # Multi-step workflow engine
│   │   └── vector_store.py    # FAISS HNSW vector index
│   ├── models/                # SQLAlchemy ORM models (13 tables)
│   └── schemas/               # Pydantic request/response schemas
├── console/                   # React + Ant Design frontend
│   └── src/pages/             # 10 management pages
├── .env.example               # Environment config template
└── deploy-quick.sh            # One-click deployment
```

---

## Architecture

```
User Message
    │
    ▼
┌─────────────────────────────────────┐
│           Agent Runtime             │
│                                     │
│  1. Risk Check                      │
│  2. Intent Detection                │
│     ├── Action intent → skip RAG    │
│     └── Info query → pre-retrieval  │
│  3. Build skill tools (OpenAI fn)   │
│  4. LLM with function calling       │
│     ├── search_knowledge(query)     │
│     ├── start_workflow_xxx(reason)  │
│     ├── http_tool_xxx(params)       │
│     └── delegate_to_xxx(message)    │
│  5. Execute tool → return to LLM    │
│  6. Final answer                    │
└─────────────────────────────────────┘
```

**Conversation-first**: No explicit intent classifier or router. The LLM decides when to call tools based on the conversation context. Skills are exposed as OpenAI-format function definitions.

---

## Core API

### Chat with an agent

```bash
# Synchronous
curl -X POST http://localhost:8000/api/v1/invoke \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "your-agent-id", "message": "Hello"}'

# SSE Streaming
curl -X POST http://localhost:8000/api/v1/invoke/stream \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "your-agent-id", "message": "Hello"}'
```

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/knowledge/upload \
  -F "file=@manual.pdf" \
  -F "source_id=your-source-id" \
  -F "domain=docs"
```

### Full API docs

Start the server and visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Console Pages

| Page | What it does |
|------|-------------|
| **Dashboard** | Metrics overview (total agents, messages, avg latency) |
| **Playground** | Chat with any agent, view function calling traces in real time |
| **Agents** | Create/edit agents with tabbed config (Basic Info, Capabilities, Advanced) |
| **Skills** | Manage standalone skills (auto-managed ones are hidden) |
| **Workflows** | Define multi-step business processes with field validation |
| **Knowledge** | Upload documents, manage knowledge sources, view chunks |
| **Tools** | Register external HTTP APIs with connectivity testing |
| **LLM Configs** | Configure LLM providers with templates and connection testing |
| **Settings** | Performance presets (fast/balanced/accurate) and fine-tuning |
| **Audit** | Full call chain replay with latency breakdown |

---

## LLM Provider Support

| Provider | Config | Models |
|----------|--------|--------|
| **Ollama** (local) | `HLAB_LLM_PROVIDER=openai_compatible` | qwen2.5, llama3, mistral, etc. |
| **DashScope** | `HLAB_LLM_PROVIDER=dashscope` | qwen-flash, qwen-plus, qwen-max |
| **OpenAI** | `HLAB_LLM_PROVIDER=openai_compatible` | gpt-4o, gpt-4o-mini |
| **vLLM** | `HLAB_LLM_PROVIDER=openai_compatible` | Any model served via vLLM |
| **ZhipuAI** | `HLAB_LLM_PROVIDER=zhipu` | glm-4, glm-4-flash |

You can also manage multiple LLM configs via the **LLM Configs** page in the console and assign different providers to different agents.

---

## RAG Pipeline

Three-channel hybrid retrieval with fusion:

1. **Exact Match (KV)** — Sub-50ms entity key lookup
2. **Vector Search (HNSW)** — FAISS IndexHNSWFlat with bge-m3 embeddings (1024-dim)
3. **Keyword Search (BM25)** — jieba tokenization + BM25 scoring

Results are fused using **Reciprocal Rank Fusion (k=60)** with configurable weights. Optional **cross-encoder reranking** (bge-reranker-v2-m3) can be enabled in the "accurate" preset.

---

## Configuration

### Required for production

```bash
HLAB_CORS_ORIGINS=https://yourdomain.com   # Lock down CORS
HLAB_DISABLE_AUTH=false                      # Enable API key auth
HLAB_API_KEY=<random-64-chars>              # Your API key
```

### Common options

| Variable | Default | Description |
|----------|---------|-------------|
| `HLAB_DATABASE_URL` | `sqlite+aiosqlite:///./data/hlab.db` | Database URL (SQLite dev, PostgreSQL prod) |
| `HLAB_LLM_PROVIDER` | `openai_compatible` | LLM provider |
| `HLAB_LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API endpoint |
| `HLAB_LLM_API_KEY` | (empty) | API key for cloud providers |
| `HLAB_LLM_MODEL` | `qwen-flash` | Default model name |
| `HLAB_LLM_TIMEOUT` | `60` | LLM call timeout in seconds |
| `HLAB_EMBEDDING_PROVIDER` | `local` | Embedding provider |
| `HLAB_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `HLAB_VECTOR_STORE` | `faiss` | Vector store backend |

See `.env.example` for the full list.

---

## Tech Stack

**Backend**: FastAPI, SQLAlchemy (async), Pydantic v2, FAISS, jieba
**Frontend**: React 18, TypeScript, Ant Design v5, Vite
**Database**: SQLite (dev) / PostgreSQL (prod)
**LLM**: OpenAI-compatible API format

---

## License

MIT

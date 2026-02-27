# HlAB — Headless AI Agent Builder

可私有部署的 AI Agent 平台，附带 Web 管理控制台。可视化创建、配置和运行对话式 AI 智能体 —— 支持知识库问答、多步骤工作流、外部 API 调用、多 Agent 协作 —— 同时提供 Headless API 和 SSE 流式接口，方便集成到任何系统。

**零厂商锁定。** 兼容任意 OpenAI 格式的 LLM —— 本地 Ollama、阿里 DashScope（通义千问）、vLLM、OpenAI、智谱 AI 等。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| **多 Agent 管理** | 创建多个独立 Agent，各自拥有系统提示词、知识库、工具和工作流 |
| **知识库 (RAG)** | 上传 TXT / PDF / DOCX 文档，自动分块、向量化，三通道混合检索 + RRF 融合 |
| **工作流引擎** | 定义多步骤业务流程（如报修工单、申请审批），LLM 根据用户意图自动触发 |
| **工具调用** | 注册外部 HTTP API 作为工具，Agent 通过 Function Calling 自动调用 |
| **多 Agent 协作** | Agent 之间可互相委派任务，内置深度限制（最大 3 层）和循环检测 |
| **LLM 配置管理** | 统一管理多个 LLM 供应商配置，按 Agent 独立分配，支持模板一键填充和连通性测试 |
| **Web 控制台** | 10 页 React 管理界面，覆盖所有功能的可视化操作 |
| **Headless API** | 全部功能均可通过 REST API + SSE 流式接口调用，70+ 个接口 |

---

## 快速开始

### 环境要求

- **Python 3.10+**
- **Node.js 18+**（仅在需要修改前端或自行构建时需要）

### 第一步：克隆项目

```bash
git clone https://github.com/your-repo/headlessAIAgentPlatform.git
cd headlessAIAgentPlatform
```

### 第二步：创建虚拟环境并安装依赖

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[rag]"           # 安装后端 + RAG 相关依赖
```

依赖通过 `pyproject.toml` 管理。`[rag]` 可选依赖包含 FAISS、sentence-transformers、jieba、pypdf、python-docx 等。

> **Apple Silicon 用户**：如果 `faiss-cpu` 安装失败，尝试 `pip install faiss-cpu --no-cache-dir`。

### 第三步：配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，最少只需配置 LLM 连接信息即可启动：

```bash
# ─── 方案 A：本地 Ollama（免费，无需 API Key）───
HLAB_LLM_PROVIDER=openai_compatible
HLAB_LLM_BASE_URL=http://localhost:11434/v1
HLAB_LLM_MODEL=qwen2.5

# ─── 方案 B：阿里 DashScope ───
HLAB_LLM_PROVIDER=dashscope
HLAB_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
HLAB_LLM_API_KEY=sk-你的密钥
HLAB_LLM_MODEL=qwen-flash
```

> **关于 LLM 配置的层级**：`.env` 里的 LLM 设置是启动时的兜底默认值。服务运行后，你可以在控制台 **LLM Configs** 页面创建多套 LLM 配置（Ollama / DashScope / OpenAI / vLLM 等），然后在 **Agents** 页面为每个 Agent 单独选择使用哪套配置。
>
> 优先级：Agent 绑定的配置 > 租户默认配置 > `.env` 兜底值。

### 第四步：启动服务

```bash
source venv/bin/activate
HLAB_DISABLE_AUTH=true python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

服务启动后自动创建 SQLite 数据库和所有表，无需手动迁移。

### 第五步：打开控制台

**生产模式**（需先构建前端）：

```bash
cd console
npm install
npm run build                     # 输出到 console/dist/
cp -r dist/ ../static/            # 复制到 static/ 供后端托管
cd ..
# 重启后端，访问 http://localhost:8000
```

**开发模式**（前后端分离，支持热更新）：

```bash
cd console
npm install
npm run dev
# 在 http://localhost:3000 打开，API 自动代理到 :8000
```

---

## 使用流程

服务启动后，典型的使用流程如下：

### 1. 配置 LLM

进入 **LLM Configs** 页面 → 点击 "Create Config" → 选择供应商（如 DashScope） → 点击 "Load Template" 自动填充参数 → 填入 API Key → 点击 "Test Config" 测试连通性 → 保存。

可以创建多套配置（如一个快速便宜的 qwen-flash 用于日常，一个 qwen-max 用于复杂场景），并设置其中一个为默认。

### 2. 创建 Agent

进入 **Agents** 页面 → 点击 "Create Agent" →

- **Basic Info** 标签：填写名称、描述、系统提示词，选择 LLM 配置
- **Capabilities** 标签：配置 Agent 的能力（知识库、工作流、工具、委派）
- **Advanced** 标签：响应格式、风控配置

### 3. 添加知识库（可选）

进入 **Knowledge** 页面 → 创建知识源 → 上传文档（TXT / PDF / DOCX）→ 系统自动分块和向量化。

也可以手动添加 KV 实体条目（如"物业电话: 0571-88001234"），用于精确匹配的快速查找。

在 Agent 的 Capabilities 标签中绑定对应的知识域，Agent 就能自动回答知识库中的问题。

### 4. 创建工作流（可选）

进入 **Workflows** 页面 → 创建工作流 → 定义步骤（信息收集、确认、完成等）→ 每步可配置字段类型、校验规则。

在 Agent 的 Capabilities 标签中绑定工作流，Agent 会根据用户意图自动触发（如用户说"我要报修"时启动报修工作流）。

### 5. 注册工具（可选）

进入 **Tools** 页面 → 注册外部 HTTP API（填写 URL、Method、参数 Schema）→ 测试连通性。

在 Agent 的 Capabilities 标签中绑定工具，Agent 会在对话中通过 Function Calling 自动调用。

### 6. 测试对话

进入 **Playground** 页面 → 选择 Agent → 开始对话。右侧面板实时展示 Function Calling 调用链、检索结果、延迟等信息。

---

## 项目结构

```
headlessAIAgentPlatform/
├── server/                        # FastAPI 后端
│   ├── api/                       # REST API（70+ 个接口）
│   │   ├── invoke.py              #   对话调用 + SSE 流式
│   │   ├── agents.py              #   Agent 增删改查
│   │   ├── agent_capabilities.py  #   Agent 能力配置（自动管理 Skill）
│   │   ├── agent_connections.py   #   Agent 间连接（委派关系）
│   │   ├── skills.py              #   Skill 管理
│   │   ├── agent_skills.py        #   Agent-Skill 绑定
│   │   ├── workflows.py           #   工作流管理
│   │   ├── knowledge.py           #   知识库 + 文档上传
│   │   ├── tools.py               #   工具管理
│   │   ├── llm_configs.py         #   LLM 配置 + 供应商模板 + 连接测试
│   │   ├── performance.py         #   性能预设 + 运行时参数
│   │   ├── vector_admin.py        #   向量索引管理
│   │   ├── audit.py               #   审计日志
│   │   └── mock_tools.py          #   内置 Mock 工具（计算器、天气等）
│   ├── engine/                    # 核心引擎
│   │   ├── agent_runtime.py       #   Agent 执行管线：意图识别 + 预检索 + Function Calling
│   │   ├── knowledge_retriever.py #   三通道混合检索 + RRF 融合 + 交叉编码器重排
│   │   ├── llm_adapter.py         #   多供应商 LLM 适配器 + Function Calling
│   │   ├── workflow_executor.py   #   工作流执行器（字段验证 + 文件上传 + LLM 验证）
│   │   ├── tool_executor.py       #   工具执行器 + 指数退避重试
│   │   └── vector_store.py        #   FAISS HNSW 向量索引 + 嵌入模型级联降级
│   ├── models/                    # SQLAlchemy ORM 模型（13 张表）
│   ├── schemas/                   # Pydantic 请求/响应 Schema
│   ├── config.py                  # 环境变量配置
│   ├── runtime_config.py          # 运行时参数（性能预设生效位置）
│   └── performance_presets.py     # 快速/均衡/精确 预设定义
├── console/                       # React + Ant Design 前端
│   ├── src/
│   │   ├── api.ts                 #   集中式 API 客户端（所有前端请求统一入口）
│   │   ├── App.tsx                #   路由配置
│   │   └── pages/                 #   10 个管理页面
│   ├── package.json
│   └── vite.config.ts             #   Vite 配置（dev 模式代理 API 到 :8000）
├── tests/                         # 测试用例（单元测试 + E2E）
├── pyproject.toml                 # Python 依赖配置
└── .env.example                   # 环境变量模板
```

---

## 架构设计

### 对话处理管线

```
用户消息
    │
    ▼
┌──────────────────────────────────────────┐
│            Agent Runtime                 │
│                                          │
│  1. 风险检查（配置的风控关键词过滤）       │
│  2. 意图识别（LLM 驱动 + 快速关键词兜底）  │
│     ├── 办事意图 → 跳过预检索              │
│     └── 信息查询 → 并行预检索知识库         │
│  3. 查询改写（多轮对话上下文补全）          │
│  4. 构建 Skill 工具列表（OpenAI 函数格式）  │
│  5. LLM + Function Calling               │
│     ├── search_knowledge(query)           │
│     ├── start_workflow_xxx(reason)        │
│     ├── http_tool_xxx(params)            │
│     └── delegate_to_xxx(message)         │
│  6. 执行工具 → 结果返回 LLM               │
│  7. 生成最终回复 + 推荐后续问题             │
└──────────────────────────────────────────┘
```

**对话优先架构**：没有显式的意图分类器或路由层。LLM 根据对话上下文自主决定何时调用工具。所有 Skill（知识检索、工作流启动、HTTP 工具、Agent 委派）以 OpenAI Function Calling 格式暴露给模型，由模型自然选择。

### LLM 配置优先级

```
Agent 绑定的 LLM Config（在 Agents 页面选择）
    │ 未绑定？
    ▼
租户默认 LLM Config（在 LLM Configs 页面设置 Default）
    │ 不存在？
    ▼
.env 环境变量兜底值
```

### 知识库检索管线 (RAG)

```
用户查询
    │
    ├─── 精确匹配 (KV)     <50ms    实体键值直接查找
    ├─── 向量检索 (HNSW)   <200ms   bge-m3 嵌入 → FAISS 近邻搜索
    └─── 关键词检索 (BM25)  <100ms   jieba 分词 → BM25 评分
         │
         ▼
    RRF 融合排序（k=60，加权合并三通道结果）
         │
         ▼
    [可选] 交叉编码器重排（bge-reranker-v2-m3）
         │
         ▼
    Top-K 结果送入 LLM 生成回答
```

三通道**并行执行**（asyncio.gather），互不阻塞。任一通道失败时自动降级，不影响其余通道。

---

## 控制台页面

| 页面 | 功能 |
|------|------|
| **Dashboard** | 总览面板：Agent 总数、消息量、平均延迟等统计指标 |
| **Playground** | 与 Agent 实时对话测试，右侧面板展示 Function Calling 调用链、检索结果、耗时分解 |
| **Agents** | 创建/编辑 Agent，分标签页配置：基本信息、能力（知识库/工作流/工具/委派）、高级设置 |
| **Skills** | 管理独立 Skill（Agent 页面自动创建的托管 Skill 自动隐藏，不在此显示） |
| **Workflows** | 定义多步骤业务流程，每步可配置字段类型、校验规则、文件上传、LLM 语义验证 |
| **Knowledge** | 上传文档（TXT/PDF/DOCX）、管理知识源、添加 KV 实体条目、查看分块详情 |
| **Tools** | 注册外部 HTTP API 为工具，配置参数 Schema，支持一键连通性测试 |
| **LLM Configs** | 管理 LLM 供应商配置，支持模板一键填充（Ollama/DashScope/OpenAI/vLLM/智谱）和连接测试 |
| **Settings** | 性能预设切换（快速/均衡/精确）和高级参数微调（检索权重、超时、重排等） |
| **Audit** | 完整调用链回放，每一步的事件类型、耗时、输入输出均可查看 |

---

## 核心 API

所有 API 前缀为 `/api/v1`。启动服务后访问 `http://localhost:8000/docs` 查看完整的 Swagger 交互式文档。

### 与 Agent 对话

```bash
# 同步调用
curl -X POST http://localhost:8000/api/v1/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "你的agent-id",
    "message": "你好",
    "session_id": "可选-会话ID",
    "tenant_id": "default"
  }'
```

```bash
# SSE 流式调用（适合前端实时展示）
curl -X POST http://localhost:8000/api/v1/invoke/stream \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "你的agent-id",
    "message": "你好"
  }'
# 返回格式：
#   event: status    data: {"step": "intent_detection", ...}
#   event: answer    data: {"content": "你好！有什么可以帮你的？", ...}
#   event: done      data: {"session_id": "xxx", ...}
```

### 管理 Agent

```bash
# 创建 Agent
curl -X POST http://localhost:8000/api/v1/agents/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "客服助手",
    "description": "智能客服",
    "system_prompt": "你是一个专业的客服助手，用简洁的中文回答问题。",
    "llm_config_id": "可选-LLM配置ID"
  }'

# 配置 Agent 能力（知识库、工作流、工具）
curl -X PUT http://localhost:8000/api/v1/agents/{agent_id}/capabilities \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_qa": {
      "enabled": true,
      "domain": "customer_service",
      "description": "搜索客服知识库"
    },
    "workflows": {
      "workflow_ids": ["工作流ID"],
      "descriptions": {"工作流ID": "处理用户投诉的流程"}
    }
  }'
```

### 上传文档

```bash
curl -X POST http://localhost:8000/api/v1/knowledge/upload \
  -F "file=@产品手册.pdf" \
  -F "source_id=知识源ID" \
  -F "domain=docs" \
  -F "chunk_size=500" \
  -F "chunk_overlap=50"
```

支持 `.txt`、`.pdf`、`.docx` 格式。上传后自动进行递归分块（段落 → 行 → 句子 → 字符级别）和向量化。

### 健康检查

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0"}
```

---

## 支持的 LLM 供应商

| 供应商 | 配置方式 | 推荐模型 | 说明 |
|--------|----------|----------|------|
| **Ollama**（本地） | `openai_compatible` | qwen2.5, llama3, mistral | 免费，无需 API Key，需本地安装 Ollama |
| **DashScope**（阿里） | `dashscope` | qwen-flash, qwen-plus, qwen-max | 性价比高，qwen-flash 极低价 |
| **OpenAI** | `openai_compatible` | gpt-4o, gpt-4o-mini | 直接兼容 |
| **vLLM** | `openai_compatible` | 任意 vLLM 部署模型 | 自建推理服务 |
| **智谱 AI** | `zhipu` | glm-4, glm-4-flash | 国产大模型 |

所有使用 OpenAI 兼容 API 格式的供应商（包括 DeepSeek、MiniMax、零一万物等）均可通过 `openai_compatible` 接入。

在控制台 **LLM Configs** 页面，可以同时管理多套配置，为不同 Agent 分配不同的 LLM 供应商和模型。

---

## 性能预设

通过 **Settings** 页面或 API 切换，影响检索策略和参数：

| 预设 | 检索超时 | 向量 efSearch | 重排 | 适用场景 |
|------|---------|--------------|------|---------|
| **快速** | 3s | 64 | 关闭 | 实时对话、低延迟需求 |
| **均衡**（默认） | 5s | 128 | 关闭 | 通用场景 |
| **精确** | 10s | 256 | 开启 (bge-reranker-v2-m3) | 需要高准确率的专业问答 |

也可以在 Settings 页面逐项调整参数（检索通道权重、Top-K、超时等），实时生效无需重启。

---

## 环境变量配置

### 生产环境必填

```bash
HLAB_CORS_ORIGINS=https://你的域名.com    # 限制跨域来源
HLAB_DISABLE_AUTH=false                    # 开启 API Key 认证
HLAB_API_KEY=随机64位字符串                 # 你的 API Key
```

### 完整配置项

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HLAB_DEBUG` | `false` | 调试模式 |
| `HLAB_DATABASE_URL` | `sqlite+aiosqlite:///./data/hlab.db` | 数据库连接（开发用 SQLite，生产建议 PostgreSQL） |
| `HLAB_LLM_PROVIDER` | `openai_compatible` | LLM 供应商类型 |
| `HLAB_LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API 地址 |
| `HLAB_LLM_API_KEY` | （空） | 云端供应商的 API Key |
| `HLAB_LLM_MODEL` | `qwen-flash` | 默认模型名称 |
| `HLAB_LLM_TIMEOUT` | `60` | LLM 调用超时（秒） |
| `HLAB_EMBEDDING_PROVIDER` | `local` | 嵌入模型供应商（`local` 为本地 sentence-transformers） |
| `HLAB_EMBEDDING_MODEL` | `BAAI/bge-m3` | 嵌入模型（首次运行自动下载） |
| `HLAB_EMBEDDING_DIM` | `1024` | 嵌入向量维度 |
| `HLAB_VECTOR_STORE` | `faiss` | 向量存储后端 |
| `HLAB_FAISS_INDEX_DIR` | `./data/faiss` | FAISS 索引文件目录 |
| `HLAB_CORS_ORIGINS` | `*` | 允许的跨域来源（生产环境必须限制） |
| `HLAB_DISABLE_AUTH` | `true` | 是否关闭 API 认证（生产环境必须设为 false） |
| `HLAB_SECRET_KEY` | `change-me-in-production` | JWT 签名密钥 |
| `HLAB_AUDIT_ENABLED` | `true` | 是否启用审计日志 |

---

## 数据库表结构

| 表 | 用途 | 关键字段 |
|----|------|----------|
| `agents` | Agent 配置 | name, system_prompt, llm_config_id, response_config |
| `skills` | Skill 定义 | skill_type, execution_config, managed_by |
| `agent_skills` | Agent-Skill 绑定 | agent_id, skill_id |
| `agent_connections` | Agent 间委派关系 | source_agent_id, target_agent_id |
| `workflows` | 工作流定义 | name, mode, version |
| `workflow_steps` | 工作流步骤 | step_type, fields, prompt_template |
| `knowledge_sources` | 知识源 | domain, source_type, embedding_model |
| `knowledge_chunks` | 知识分块 | content, entity_key, embedding |
| `tool_definitions` | 工具定义 | endpoint_url, method, input_schema |
| `conversation_sessions` | 会话 | agent_id, tenant_id, delegation_chain |
| `messages` | 消息记录 | session_id, role, content |
| `audit_traces` | 审计日志 | event_type, event_data, latency |
| `llm_configs` | LLM 配置 | provider, base_url, model, is_default |

数据库表在服务首次启动时自动创建，无需手动迁移。

---

## 技术栈

| 层 | 技术 |
|----|------|
| **后端** | FastAPI, SQLAlchemy (async), Pydantic v2, FAISS (HNSW), jieba, sentence-transformers |
| **前端** | React 18, TypeScript, Ant Design v5, Vite |
| **数据库** | SQLite（开发）/ PostgreSQL（生产） |
| **LLM** | OpenAI 兼容 API 格式（支持 Function Calling） |
| **嵌入模型** | bge-m3（1024 维），支持级联降级到 MiniLM 等小模型 |
| **向量索引** | FAISS IndexHNSWFlat（M=32, efConstruction=200） |

---

## 开发指南

### 后端开发

```bash
source venv/bin/activate
pip install -e ".[rag,dev]"       # 安装开发依赖（pytest, ruff 等）
python -m uvicorn server.main:app --reload --port 8000
```

### 前端开发

```bash
cd console
npm install
npm run dev                       # http://localhost:3000，自动代理 API
```

### 构建前端用于生产

```bash
cd console
npm run build                     # 输出到 console/dist/
cp -r dist/ ../static/            # 复制到项目根目录的 static/
```

后端检测到 `static/` 目录后，自动托管前端页面（SPA 模式），无需额外配置 Nginx。

### 运行测试

```bash
pytest tests/                     # 单元测试
pytest tests/e2e/ -m e2e          # E2E 测试（需要后端运行中）
```

---

## License

MIT

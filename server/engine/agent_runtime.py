"""Agent Runtime — the core orchestrator.

Pipeline: Input → Risk Check → Intent Classification → Routing → Response
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.models.agent import Agent
from server.models.session import ConversationSession, Message
from server.models.tool import ToolDefinition
from server.schemas.invoke import (
    InvokeRequest,
    InvokeResponse,
    Citation,
)
from server.engine.llm_adapter import LLMMessage, get_llm_adapter
from server.engine.knowledge_retriever import KnowledgeRetriever
from server.engine.tool_gateway import ToolGateway
from server.engine.workflow_executor import WorkflowExecutor
from server.engine.audit_logger import AuditLogger, new_trace_id

logger = logging.getLogger(__name__)

# Maximum tool-calling rounds before forcing a final answer
MAX_TOOL_ROUNDS = 5


# ── Intent classification result ────────────────────────────────

@dataclass
class IntentResult:
    intent: str  # greeting|chitchat|knowledge_query|workflow_start|workflow_continue|workflow_exit|tool_use
    workflow_id: str | None = None
    confidence: str = "high"  # high|low
    source: str = "fast_path"  # fast_path|llm


# ── Prompt templates ─────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """{persona}

# 回复规则
1. 默认用1-3句话精准回答，不要一股脑输出
2. 如果用户说"详细""展开""完整流程"等，才给出完整回答
3. 回答必须基于提供的参考资料，不得编造
4. 如果参考资料不足，明确说"暂无相关资料，建议咨询人工客服"
5. 用自然语言直接回答，不要输出JSON格式
"""

SYSTEM_PROMPT_WITH_TOOLS = """{persona}

# 你拥有以下能力
- 你可以调用工具来获取实时数据或执行操作
- 如果参考资料已经足够回答，直接回答，不需要调用工具
- 如果需要查询实时信息或执行操作，请调用合适的工具
- 工具返回结果后，请用自然语言总结回答用户

# 回复规则
1. 优先基于参考资料和工具结果给出准确回答
2. 默认简洁回答，用户要求时才展开
3. 如果既无参考资料也无可用工具，说明情况并建议咨询人工客服
4. 用自然语言直接回答，不要输出JSON格式
"""

FAST_ANSWER_PROMPT = """{persona}

# 重要指令
参考资料中已包含用户问题的精确答案。你必须使用参考资料中的数据直接回答。

# 严格禁止
- 禁止说"没有提供"、"没有找到"、"没有相关"、"暂无"等拒绝性回答
- 禁止质疑参考资料的准确性
- 禁止忽略参考资料中的数据

# 回复规则
1. 直接引用参考资料中的数据回答，1-2句话
2. 用自然语言直接回答，不要输出JSON格式
"""

RETRIEVAL_CONTEXT_TEMPLATE = """# 参考资料
{context}
"""

INTENT_CLASSIFICATION_PROMPT = """你是意图分类器。根据用户消息和可用能力，输出一个JSON。
可用能力:
{capabilities}

输出格式(仅JSON，不要其他文字):
{{"intent":"greeting|chitchat|knowledge_query|workflow_start|tool_use","workflow_id":"仅workflow_start时填写，填写对应的workflow id","confidence":"high|low"}}

用户消息: {message}"""

CHITCHAT_PROMPT = """{persona}

请用自然、友好的方式回复用户的闲聊。保持简短，1-2句话即可。不需要提供专业知识或查询数据。"""


# ── Multi-turn query rewriting ─────────────────────────────────

# Patterns indicating the user is referencing prior conversation context
_CONTEXT_DEPENDENT_PATTERNS = [
    "这个", "那个", "它", "他", "她", "他们", "她们", "它们",
    "上面", "前面", "刚才", "之前", "上述", "上面说的",
    "还有吗", "还有呢", "继续", "然后呢", "接下来",
    "为什么", "怎么回事",
    "this", "that", "it", "them", "above", "previous",
    "more", "continue", "why", "how come",
]


def _needs_query_rewrite(message: str) -> bool:
    """Check if a message likely refers to prior conversation context."""
    msg = message.strip().lower()
    # Very short messages are often context-dependent
    if len(msg) <= 6:
        return True
    for pattern in _CONTEXT_DEPENDENT_PATTERNS:
        if pattern in msg:
            return True
    return False


def _rewrite_query_with_history(message: str, history_messages: list) -> str:
    """Rewrite a context-dependent query by prepending recent conversation context.

    This is a lightweight, no-LLM-call approach: we prepend the last
    assistant answer as context so the retriever searches with more signal.
    """
    if not history_messages:
        return message

    # Find the last assistant message
    last_assistant = None
    last_user = None
    for msg in reversed(history_messages):
        if msg.role == "assistant" and not last_assistant:
            last_assistant = msg.content
        elif msg.role == "user" and not last_user:
            last_user = msg.content
        if last_assistant and last_user:
            break

    # Combine the previous topic with the current question for retrieval
    if last_user:
        return f"{last_user} {message}"
    return message


# ── Refusal phrases (LLM hallucinated a "no data" when retrieval found results)
_REFUSAL_PHRASES = [
    "暂无", "无此", "没有相关", "无法找到", "不确定", "未找到",
    "没有提供", "没有包含", "不包含", "未提及", "未提供",
    "没有找到", "未能找到", "无法确定", "没有数据", "无数据",
    "没有记录", "未收录", "资料中没有", "参考资料不足",
]

# ── Workflow exit keywords ─────────────────────────────────────
_WORKFLOW_EXIT_KEYWORDS = {
    "取消", "退出", "不办了", "算了", "不要了", "放弃",
    "cancel", "quit", "exit", "abort", "stop",
}

# ── Greeting patterns ─────────────────────────────────────────
_GREETING_PATTERNS = [
    "你好", "您好", "早上好", "晚上好", "下午好", "hi", "hello", "hey",
    "谢谢", "谢了", "多谢", "thanks", "thank you",
    "再见", "拜拜", "bye", "goodbye",
    "好的", "好", "嗯", "ok", "okay", "是的", "对", "不是", "没有",
    "嗯", "哦", "啊", "呃", "额",
]


class AgentRuntime:
    """Main orchestrator for handling user requests."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Workflow scope resolution ─────────────────────────────
    @staticmethod
    def _resolve_workflow_scope(agent: Agent) -> dict | None:
        """Get effective workflow scope, with backward compatibility.

        If agent has workflow_scope, use it directly.
        If agent only has workflow_id (legacy), auto-convert.
        """
        if agent.workflow_scope:
            return agent.workflow_scope
        if agent.workflow_id:
            return {"workflow_ids": [agent.workflow_id]}
        return None

    # ── Intent Classification ─────────────────────────────────
    async def _classify_intent(
        self, message: str, agent: Agent, session: ConversationSession,
    ) -> IntentResult:
        """Hybrid intent classification: fast-path rules + LLM fallback.

        Fast-path (no LLM call):
          1. Active workflow_state → workflow_continue (or workflow_exit)
          2. Greeting keywords → greeting
          3. Agent has NO tools and NO workflows → knowledge_query

        LLM classification (only when ambiguous):
          - Agent has workflows AND/OR tools → ask LLM to classify
        """
        msg_lower = message.strip().lower()

        # ── Fast path 1: Active workflow → continue or exit
        wf_state = session.workflow_state or {}
        if wf_state.get("status") not in (None, "completed", "cancelled"):
            # Check for exit intent
            for kw in _WORKFLOW_EXIT_KEYWORDS:
                if kw in msg_lower:
                    return IntentResult(intent="workflow_exit", source="fast_path")
            return IntentResult(intent="workflow_continue", source="fast_path")

        # ── Fast path 2: Greeting detection
        if len(msg_lower) <= 10:
            for pattern in _GREETING_PATTERNS:
                if msg_lower == pattern or msg_lower.startswith(pattern):
                    return IntentResult(intent="greeting", source="fast_path")

        # ── Fast path 3: No tools and no workflows → knowledge_query
        workflow_scope = self._resolve_workflow_scope(agent)
        tool_ids = self._extract_tool_ids(agent.tool_scope)
        has_workflows = bool(workflow_scope and workflow_scope.get("workflow_ids"))
        has_tools = bool(tool_ids)

        if not has_workflows and not has_tools:
            return IntentResult(intent="knowledge_query", source="fast_path")

        # ── LLM classification (agent has workflows and/or tools)
        return await self._llm_classify_intent(message, agent, workflow_scope, tool_ids)

    async def _llm_classify_intent(
        self, message: str, agent: Agent,
        workflow_scope: dict | None, tool_ids: list[str],
    ) -> IntentResult:
        """Use LLM to classify intent when fast-path is ambiguous."""
        # Build capabilities list
        capabilities = []
        if workflow_scope:
            wf_ids = workflow_scope.get("workflow_ids", [])
            descriptions = workflow_scope.get("descriptions", {})
            for wf_id in wf_ids:
                desc = descriptions.get(wf_id, "")
                if desc:
                    capabilities.append(f"- workflow({wf_id}): {desc}")
                else:
                    capabilities.append(f"- workflow({wf_id})")
        if tool_ids:
            # Load tool names for better classification
            result = await self.db.execute(
                select(ToolDefinition.id, ToolDefinition.name, ToolDefinition.description)
                .where(ToolDefinition.id.in_(tool_ids), ToolDefinition.enabled == True)
            )
            for row in result.all():
                desc = row.description or row.name
                capabilities.append(f"- tool({row.name}): {desc}")

        capabilities_text = "\n".join(capabilities) if capabilities else "无特殊能力"

        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            capabilities=capabilities_text,
            message=message,
        )

        llm = get_llm_adapter(model=agent.llm_model) if agent.llm_model else get_llm_adapter()
        try:
            resp = await llm.chat(
                [LLMMessage(role="user", content=prompt)],
                max_tokens=2048,  # Reasoning models need room for thinking + output
                temperature=0.0,
            )
            # Parse JSON response
            text = resp.content.strip()
            # Strip markdown code block if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            parsed = json.loads(text.strip())

            intent = parsed.get("intent", "knowledge_query")
            valid_intents = {"greeting", "chitchat", "knowledge_query", "workflow_start", "tool_use"}
            if intent not in valid_intents:
                intent = "knowledge_query"

            return IntentResult(
                intent=intent,
                workflow_id=parsed.get("workflow_id"),
                confidence=parsed.get("confidence", "high"),
                source="llm",
            )
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}, falling back to knowledge_query")
            return IntentResult(intent="knowledge_query", source="fast_path", confidence="low")

    # ── Main invoke entry point ──────────────────────────────
    async def invoke(self, req: InvokeRequest) -> InvokeResponse:
        """Full pipeline: one user message in, one structured response out."""

        # 1. Load agent config
        agent = await self._load_agent(req.agent_id)
        if agent is None:
            return self._error_response("Agent 不存在或已停用", req)

        # 2. Get or create session
        session = await self._get_or_create_session(req, agent)

        # 3. Init audit
        trace_id = new_trace_id()
        audit = AuditLogger(self.db, trace_id, session.id, agent.id, agent.tenant_id)
        audit.log("user_input", event_data={"message": req.message, "form_data": req.form_data})

        # 4. Risk pre-check
        risk_block = self._risk_precheck(agent, req.message)
        if risk_block:
            audit.log("risk_block", event_data={"reason": risk_block})
            await audit.flush()
            return InvokeResponse(
                session_id=session.id,
                trace_id=trace_id,
                short_answer=risk_block,
                suggested_followups=["换个问题试试？"],
            )

        # 5. Intent classification
        intent = await self._classify_intent(req.message, agent, session)
        audit.log("intent", event_data={
            "intent": intent.intent,
            "workflow_id": intent.workflow_id,
            "confidence": intent.confidence,
            "source": intent.source,
        })

        # 6. Route based on intent
        if intent.intent == "greeting":
            return await self._handle_greeting(agent, session, req, trace_id, audit)

        elif intent.intent == "workflow_continue":
            return await self._handle_workflow(agent, session, req, trace_id, audit)

        elif intent.intent == "workflow_exit":
            return await self._handle_workflow_exit(agent, session, req, trace_id, audit)

        elif intent.intent == "workflow_start":
            return await self._handle_workflow_start(
                agent, session, req, trace_id, audit, intent.workflow_id,
            )

        elif intent.intent == "chitchat":
            return await self._handle_chitchat(agent, session, req, trace_id, audit)

        elif intent.intent == "tool_use":
            return await self._handle_qa(agent, session, req, trace_id, audit)

        else:  # knowledge_query or fallback
            return await self._handle_qa(agent, session, req, trace_id, audit)

    # ── Workflow start ────────────────────────────────────────
    async def _handle_workflow_start(
        self, agent: Agent, session: ConversationSession,
        req: InvokeRequest, trace_id: str, audit: AuditLogger,
        workflow_id: str | None,
    ) -> InvokeResponse:
        """Start a workflow. If workflow_id is None, try to resolve from scope."""
        wf_scope = self._resolve_workflow_scope(agent)
        wf_ids = (wf_scope or {}).get("workflow_ids", [])

        if not wf_ids:
            # No workflows available — fall back to QA
            return await self._handle_qa(agent, session, req, trace_id, audit)

        # Resolve which workflow to start
        target_wf_id = workflow_id
        if not target_wf_id or target_wf_id not in wf_ids:
            if len(wf_ids) == 1:
                target_wf_id = wf_ids[0]
            else:
                # LLM didn't specify or specified wrong workflow — present options
                descriptions = (wf_scope or {}).get("descriptions", {})
                options = []
                for wf_id in wf_ids:
                    desc = descriptions.get(wf_id, wf_id)
                    options.append(f"- {desc}")
                options_text = "\n".join(options)

                await self._save_message(session.id, "user", req.message, trace_id)
                short_answer = f"我可以帮您办理以下业务，请告诉我您需要哪个：\n{options_text}"
                await self._save_message(session.id, "assistant", short_answer, trace_id)
                await self._save_session(session)
                audit.log("response", event_data={"mode": "workflow_disambiguate"})
                await audit.flush()

                return InvokeResponse(
                    session_id=session.id,
                    trace_id=trace_id,
                    short_answer=short_answer,
                    suggested_followups=[descriptions.get(wf_id, wf_id) for wf_id in wf_ids[:3]],
                )

        # Initialize workflow state
        session.workflow_state = {
            "workflow_id": target_wf_id,
            "current_step_index": 0,
            "status": "in_progress",
        }

        return await self._handle_workflow(agent, session, req, trace_id, audit)

    # ── Workflow handling ────────────────────────────────────────
    async def _handle_workflow(
        self, agent: Agent, session: ConversationSession,
        req: InvokeRequest, trace_id: str, audit: AuditLogger,
    ) -> InvokeResponse:
        tool_gw = ToolGateway(self.db, audit)
        executor = WorkflowExecutor(self.db, tool_gw, audit)

        audit.start_timer("workflow")
        result = await executor.process_step(session, req.message, req.form_data)
        wf_latency = audit.elapsed_ms("workflow")

        audit.log("workflow_step", workflow_meta={
            "status": result.status,
            "message": result.message,
        }, latency_ms=wf_latency)

        # Save session state
        await self._save_session(session)

        # Save messages
        await self._save_message(session.id, "user", req.message, trace_id)
        await self._save_message(session.id, "assistant", result.message, trace_id)

        await audit.flush()

        return InvokeResponse(
            session_id=session.id,
            trace_id=trace_id,
            short_answer=result.message,
            workflow_card=result.card,
            workflow_status=result.status,
            escalated=result.escalated,
            escalation_reason=result.message if result.escalated else None,
            suggested_followups=self._workflow_followups(result),
        )

    # ── Workflow exit handling ────────────────────────────────
    async def _handle_workflow_exit(
        self, agent: Agent, session: ConversationSession,
        req: InvokeRequest, trace_id: str, audit: AuditLogger,
    ) -> InvokeResponse:
        """Handle user wanting to exit current workflow."""
        tool_gw = ToolGateway(self.db, audit)
        executor = WorkflowExecutor(self.db, tool_gw, audit)

        result = executor.cancel_workflow(session)

        audit.log("workflow_step", workflow_meta={
            "status": "cancelled",
            "message": result,
        })

        await self._save_message(session.id, "user", req.message, trace_id)
        await self._save_message(session.id, "assistant", result, trace_id)
        await self._save_session(session)
        await audit.flush()

        return InvokeResponse(
            session_id=session.id,
            trace_id=trace_id,
            short_answer=result,
            workflow_status="cancelled",
            suggested_followups=["有什么其他可以帮助你的吗？", "需要重新开始吗？"],
        )

    # ── Chitchat handling ─────────────────────────────────────
    async def _handle_chitchat(
        self, agent: Agent, session: ConversationSession,
        req: InvokeRequest, trace_id: str, audit: AuditLogger,
    ) -> InvokeResponse:
        """Handle casual conversation without knowledge retrieval."""
        persona = agent.system_prompt or "你是一个智能助手。"
        history = await self._get_history(session.id, limit=4)

        messages = [LLMMessage(role="system", content=CHITCHAT_PROMPT.format(persona=persona))]
        for msg in history:
            messages.append(LLMMessage(role=msg.role, content=msg.content))
        messages.append(LLMMessage(role="user", content=req.message))

        llm = get_llm_adapter(model=agent.llm_model) if agent.llm_model else get_llm_adapter()
        audit.start_timer("llm")
        try:
            llm_resp = await llm.chat(messages)
        except Exception as e:
            audit.log("error", event_data={"error": str(e), "stage": "chitchat_llm"})
            await audit.flush()
            short_answer = "嗯嗯，还有什么想聊的吗？"
            await self._save_message(session.id, "user", req.message, trace_id)
            await self._save_message(session.id, "assistant", short_answer, trace_id)
            await self._save_session(session)
            return InvokeResponse(
                session_id=session.id,
                trace_id=trace_id,
                short_answer=short_answer,
                suggested_followups=["有什么可以帮助你的吗？"],
            )

        llm_latency = audit.elapsed_ms("llm")
        audit.log("llm_call", llm_meta={
            "model": llm_resp.model,
            "prompt_tokens": llm_resp.prompt_tokens,
            "completion_tokens": llm_resp.completion_tokens,
        }, latency_ms=llm_latency)

        short_answer = llm_resp.content.strip() or "嗯嗯，还有什么想聊的吗？"

        await self._save_message(session.id, "user", req.message, trace_id)
        await self._save_message(session.id, "assistant", short_answer, trace_id)
        await self._save_session(session)

        audit.log("response", event_data={"short_answer": short_answer, "mode": "chitchat"})
        await audit.flush()

        return InvokeResponse(
            session_id=session.id,
            trace_id=trace_id,
            short_answer=short_answer,
            suggested_followups=["有什么可以帮助你的吗？"],
        )

    # ── Greeting handling ────────────────────────────────────
    async def _handle_greeting(
        self, agent: Agent, session: ConversationSession,
        req: InvokeRequest, trace_id: str, audit: AuditLogger,
    ) -> InvokeResponse:
        """Handle greetings without knowledge retrieval (fast path)."""
        persona = agent.system_prompt or "你是一个智能助手。"
        history = await self._get_history(session.id, limit=4)

        messages = [LLMMessage(role="system", content=f"{persona}\n\n# 简短回复规则\n请用1句话友好回复用户的问候或寒暄，不要解释太多。")]
        for msg in history:
            messages.append(LLMMessage(role=msg.role, content=msg.content))
        messages.append(LLMMessage(role="user", content=req.message))

        llm = get_llm_adapter(model=agent.llm_model) if agent.llm_model else get_llm_adapter()
        audit.start_timer("llm")
        try:
            llm_resp = await llm.chat(messages)
        except Exception as e:
            audit.log("error", event_data={"error": str(e), "stage": "greeting_llm"})
            await audit.flush()
            # Canned response when LLM is down
            short_answer = "你好！有什么可以帮助你的吗？"
            await self._save_message(session.id, "user", req.message, trace_id)
            await self._save_message(session.id, "assistant", short_answer, trace_id)
            await self._save_session(session)
            return InvokeResponse(
                session_id=session.id,
                trace_id=trace_id,
                short_answer=short_answer,
                suggested_followups=["有什么可以帮助你的吗？", "想了解哪方面的信息？"],
            )

        llm_latency = audit.elapsed_ms("llm")
        audit.log("llm_call", llm_meta={
            "model": llm_resp.model,
            "prompt_tokens": llm_resp.prompt_tokens,
            "completion_tokens": llm_resp.completion_tokens,
        }, latency_ms=llm_latency)

        short_answer = llm_resp.content.strip() or "你好！有什么可以帮助你的吗？"

        await self._save_message(session.id, "user", req.message, trace_id)
        await self._save_message(session.id, "assistant", short_answer, trace_id)
        await self._save_session(session)

        audit.log("response", event_data={"short_answer": short_answer, "mode": "greeting"})
        await audit.flush()

        return InvokeResponse(
            session_id=session.id,
            trace_id=trace_id,
            short_answer=short_answer,
            suggested_followups=["有什么可以帮助你的吗？", "想了解哪方面的信息？"],
        )

    # ── QA handling ──────────────────────────────────────────────
    async def _handle_qa(
        self, agent: Agent, session: ConversationSession,
        req: InvokeRequest, trace_id: str, audit: AuditLogger,
    ) -> InvokeResponse:
        # Determine knowledge domain
        domains = agent.knowledge_scope or []
        domain = domains[0] if domains else None

        # Get conversation history early (needed for query rewriting)
        history = await self._get_history(session.id, limit=6)

        # Multi-turn query rewriting
        retrieval_query = req.message
        if _needs_query_rewrite(req.message) and history:
            retrieval_query = _rewrite_query_with_history(req.message, history)
            audit.log("query_rewrite", event_data={
                "original": req.message,
                "rewritten": retrieval_query,
            })

        # Retrieve (graceful: empty result on failure)
        retriever = KnowledgeRetriever(self.db)
        audit.start_timer("retrieval")
        try:
            retrieval = await retriever.retrieve(
                retrieval_query,
                domain=domain,
                top_k=5,
            )
        except Exception as ret_err:
            logger.error(f"Retrieval failed: {ret_err}", exc_info=True)
            audit.log("error", event_data={"component": "retrieval", "error": str(ret_err)})
            from server.schemas.knowledge import RetrievalResponse
            retrieval = RetrievalResponse(hits=[], fast_answer=None, query=retrieval_query, latency_ms=0)

        retrieval_latency = audit.elapsed_ms("retrieval")

        audit.log("retrieval", retrieval_hits={
            "count": len(retrieval.hits),
            "fast_answer": retrieval.fast_answer,
            "hits": [{"id": h.chunk_id, "score": h.score, "channel": h.channel} for h in retrieval.hits],
        }, latency_ms=retrieval_latency)

        # Build context
        if retrieval.hits:
            context_parts = []
            for i, hit in enumerate(retrieval.hits, 1):
                cite_info = f"[来源: {hit.source_name}"
                if hit.page:
                    cite_info += f", 第{hit.page}页"
                cite_info += "]"
                context_parts.append(f"[{i}] {hit.content} {cite_info}")
            context_text = "\n\n".join(context_parts)
        else:
            context_text = "（无相关参考资料）"

        # Check if agent has bound tools and function calling enabled
        tool_ids = self._extract_tool_ids(agent.tool_scope)
        fc_enabled = (agent.tool_scope or {}).get("function_calling_enabled", True) if isinstance(agent.tool_scope, dict) else True
        raw_rounds = (agent.tool_scope or {}).get("max_tool_rounds", MAX_TOOL_ROUNDS) if isinstance(agent.tool_scope, dict) else MAX_TOOL_ROUNDS
        max_rounds = max(1, min(int(raw_rounds), 20))

        if tool_ids and fc_enabled:
            return await self._handle_qa_with_tools(
                agent, session, req, trace_id, audit,
                retrieval, context_text, domains, history, tool_ids,
                max_rounds=max_rounds,
            )

        # ── Standard QA path (no tools) ──
        persona = agent.system_prompt or "你是一个智能助手。"

        # If fast_answer found exact data, use a forceful prompt
        if retrieval.fast_answer:
            system_msg = FAST_ANSWER_PROMPT.format(persona=persona)
        else:
            system_msg = SYSTEM_PROMPT_TEMPLATE.format(persona=persona)

        context_msg = RETRIEVAL_CONTEXT_TEMPLATE.format(context=context_text)
        messages = [LLMMessage(role="system", content=system_msg)]
        for msg in history:
            messages.append(LLMMessage(role=msg.role, content=msg.content))
        messages.append(LLMMessage(role="user", content=f"{context_msg}\n\n用户问题: {req.message}"))

        if req.expand:
            messages.append(LLMMessage(role="user", content="请给出详细完整的回答。"))

        # Call LLM
        llm = get_llm_adapter(model=agent.llm_model) if agent.llm_model else get_llm_adapter()
        audit.start_timer("llm")
        try:
            llm_resp = await llm.chat(messages)
        except Exception as e:
            audit.log("error", event_data={"error": str(e)})
            await audit.flush()
            # If we have retrieval data, return it directly
            if retrieval.fast_answer:
                return self._retrieval_only_response(session.id, trace_id, retrieval, req.message)
            return self._fallback_response(session.id, trace_id, str(e))

        llm_latency = audit.elapsed_ms("llm")
        audit.log("llm_call", llm_meta={
            "model": llm_resp.model,
            "prompt_tokens": llm_resp.prompt_tokens,
            "completion_tokens": llm_resp.completion_tokens,
        }, latency_ms=llm_latency)

        # Parse LLM output
        parsed = self._parse_llm_output(llm_resp.content)

        # Build citations
        citations = []
        if retrieval.hits:
            for hit in retrieval.hits:
                citations.append(Citation(
                    source_id=hit.source_id,
                    source_name=hit.source_name,
                    content_snippet=hit.content[:100],
                    page=hit.page,
                    paragraph=hit.paragraph,
                    line_start=hit.line_start,
                    line_end=hit.line_end,
                    score=hit.score,
                ))

        # Build answer
        resp_config = agent.response_config or {}
        short_answer = parsed.get("short_answer", llm_resp.content).strip()

        # If LLM returned empty content, use retrieval data directly
        if not short_answer and retrieval.fast_answer:
            short_answer = retrieval.fast_answer
        elif not short_answer and retrieval.hits:
            short_answer = retrieval.hits[0].content

        # Safety net: if LLM says "no data" but retrieval actually found results
        if retrieval.hits and any(p in short_answer for p in _REFUSAL_PHRASES):
            if retrieval.fast_answer:
                short_answer = retrieval.fast_answer
            else:
                short_answer = retrieval.hits[0].content

        if (not retrieval.hits
                and domains
                and resp_config.get("no_citation_policy") == "refuse"):
            short_answer = "暂无相关资料，建议咨询人工客服或提交工单。"

        # Save messages
        await self._save_message(session.id, "user", req.message, trace_id)
        await self._save_message(session.id, "assistant", short_answer, trace_id)
        await self._save_session(session)

        followups = parsed.get("suggested_followups", [])
        if not followups:
            followups = self._generate_followups(retrieval, domains)

        audit.log("response", event_data={
            "short_answer": short_answer[:200],
            "citations_count": len(citations),
            "retrieval_hits": len(retrieval.hits),
        })
        await audit.flush()

        return InvokeResponse(
            session_id=session.id,
            trace_id=trace_id,
            short_answer=short_answer,
            expanded_answer=parsed.get("expanded_answer"),
            citations=citations,
            suggested_followups=followups,
        )

    # ── QA with Function Calling ─────────────────────────────────
    async def _handle_qa_with_tools(
        self, agent: Agent, session: ConversationSession,
        req: InvokeRequest, trace_id: str, audit: AuditLogger,
        retrieval, context_text: str, domains: list[str],
        history: list[Message], tool_ids: list[str],
        max_rounds: int = MAX_TOOL_ROUNDS,
    ) -> InvokeResponse:
        """QA pipeline with LLM native function calling.

        The LLM sees available tools and can decide to call them dynamically.
        Supports multi-round: LLM calls tool → gets result → calls another → final answer.
        """
        # Load tools in OpenAI function format
        openai_tools, tool_map = await self._load_tools_as_functions(tool_ids)

        audit.log("function_calling_init", event_data={
            "tools_loaded": len(openai_tools),
            "tool_names": [t["function"]["name"] for t in openai_tools],
        })

        # Build messages with tool-aware system prompt
        persona = agent.system_prompt or "你是一个智能助手。"
        system_msg = SYSTEM_PROMPT_WITH_TOOLS.format(persona=persona)
        context_msg = RETRIEVAL_CONTEXT_TEMPLATE.format(context=context_text)

        messages: list[LLMMessage] = [LLMMessage(role="system", content=system_msg)]
        for msg in history:
            messages.append(LLMMessage(role=msg.role, content=msg.content))
        messages.append(LLMMessage(role="user", content=f"{context_msg}\n\n用户问题: {req.message}"))

        if req.expand:
            messages.append(LLMMessage(role="user", content="请给出详细完整的回答。"))

        llm = get_llm_adapter(model=agent.llm_model) if agent.llm_model else get_llm_adapter()

        tool_gw = ToolGateway(self.db, audit)
        tool_calls_log: list[dict] = []

        # ── Tool calling loop ────────────────────────────────────
        final_content = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        llm_resp = None

        for round_idx in range(max_rounds):
            audit.start_timer(f"llm_round_{round_idx}")

            try:
                if openai_tools:
                    llm_resp = await llm.chat_with_tools(messages, openai_tools)
                else:
                    llm_resp = await llm.chat(messages)
            except Exception as e:
                audit.log("error", event_data={"error": str(e), "round": round_idx, "stage": "llm_call"})
                await audit.flush()
                if retrieval.fast_answer:
                    return self._retrieval_only_response(session.id, trace_id, retrieval, req.message)
                return self._fallback_response(session.id, trace_id, str(e))

            round_latency = audit.elapsed_ms(f"llm_round_{round_idx}")
            total_prompt_tokens += llm_resp.prompt_tokens
            total_completion_tokens += llm_resp.completion_tokens

            # If LLM gave a final text answer (no tool calls), we're done
            if not llm_resp.tool_calls:
                final_content = llm_resp.content
                audit.log("llm_call", llm_meta={
                    "model": llm_resp.model,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "rounds": round_idx + 1,
                    "tool_calls_total": len(tool_calls_log),
                }, latency_ms=round_latency)
                break

            # LLM wants to call tools — execute them
            raw_tool_calls = [tc.raw for tc in llm_resp.tool_calls]
            messages.append(LLMMessage(
                role="assistant",
                content=llm_resp.content or "",
                tool_calls=raw_tool_calls,
            ))

            # Execute each tool call
            for tc in llm_resp.tool_calls:
                tool_def = tool_map.get(tc.function_name)
                tool_result: dict[str, Any]

                if tool_def is None:
                    tool_result = {"error": f"Unknown tool: {tc.function_name}"}
                    audit.log("tool_call", tool_meta={
                        "function_name": tc.function_name,
                        "arguments": tc.arguments,
                        "success": False,
                        "error": "Tool not found in agent scope",
                        "round": round_idx,
                    })
                else:
                    audit.start_timer(f"tool_{tc.id}")
                    try:
                        tool_result = await tool_gw.invoke(tool_def.id, tc.arguments)
                        tool_latency = audit.elapsed_ms(f"tool_{tc.id}")
                        audit.log("tool_call", tool_meta={
                            "function_name": tc.function_name,
                            "tool_id": tool_def.id,
                            "arguments": tc.arguments,
                            "result": tool_result,
                            "success": True,
                            "round": round_idx,
                        }, latency_ms=tool_latency)
                    except Exception as e:
                        tool_result = {"error": str(e)}
                        tool_latency = audit.elapsed_ms(f"tool_{tc.id}")
                        audit.log("tool_call", tool_meta={
                            "function_name": tc.function_name,
                            "tool_id": tool_def.id,
                            "arguments": tc.arguments,
                            "success": False,
                            "error": str(e),
                            "round": round_idx,
                        }, latency_ms=tool_latency)

                tool_calls_log.append({
                    "function": tc.function_name,
                    "arguments": tc.arguments,
                    "result": tool_result,
                })

                # Add tool result to conversation for next LLM round
                messages.append(LLMMessage(
                    role="tool",
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_call_id=tc.id,
                ))
        else:
            # Exhausted all rounds without a final answer
            final_content = (llm_resp.content if llm_resp else None) or "抱歉，处理过程较为复杂，请重试或简化问题。"
            audit.log("llm_call", llm_meta={
                "model": llm_resp.model if llm_resp else "unknown",
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "rounds": max_rounds,
                "tool_calls_total": len(tool_calls_log),
                "exhausted": True,
            })

        # Build response
        short_answer = final_content.strip()

        # If empty after tool calling, fallback to retrieval or tool results
        if not short_answer:
            if retrieval.fast_answer:
                short_answer = retrieval.fast_answer
            elif retrieval.hits:
                short_answer = retrieval.hits[0].content
            elif tool_calls_log:
                last_result = tool_calls_log[-1].get("result", {})
                short_answer = json.dumps(last_result, ensure_ascii=False)[:500] if last_result else "工具调用完成，但未获取到有效结果。"
            else:
                short_answer = "抱歉，未能获取到有效回复，请重试。"

        # Safety net: if LLM refuses but retrieval found data
        if retrieval.hits and any(p in short_answer for p in _REFUSAL_PHRASES):
            if retrieval.fast_answer:
                short_answer = retrieval.fast_answer
            else:
                short_answer = retrieval.hits[0].content

        # Build citations from retrieval
        citations = []
        if retrieval.hits:
            for hit in retrieval.hits:
                citations.append(Citation(
                    source_id=hit.source_id,
                    source_name=hit.source_name,
                    content_snippet=hit.content[:100],
                    page=hit.page,
                    paragraph=hit.paragraph,
                    line_start=hit.line_start,
                    line_end=hit.line_end,
                    score=hit.score,
                ))

        followups = []
        if tool_calls_log:
            followups = ["查看详细结果", "还有其他问题吗？"]
        else:
            followups = self._generate_followups(retrieval, domains)

        # Save messages
        await self._save_message(session.id, "user", req.message, trace_id)
        await self._save_message(session.id, "assistant", short_answer, trace_id)
        await self._save_session(session)

        audit.log("response", event_data={
            "short_answer": short_answer[:200],
            "mode": "function_calling",
            "tool_calls": tool_calls_log,
            "citations_count": len(citations),
            "retrieval_hits": len(retrieval.hits),
        })
        await audit.flush()

        return InvokeResponse(
            session_id=session.id,
            trace_id=trace_id,
            short_answer=short_answer,
            citations=citations,
            suggested_followups=followups,
            metadata={
                "mode": "function_calling",
                "tool_calls": [{"function": tc["function"], "arguments": tc["arguments"]} for tc in tool_calls_log],
            } if tool_calls_log else None,
        )

    # ── Tool helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_tool_ids(tool_scope: dict | list | None) -> list[str]:
        """Extract tool IDs from the agent's tool_scope config."""
        if tool_scope is None:
            return []
        if isinstance(tool_scope, list):
            # Legacy format: plain list of IDs
            return [str(tid) for tid in tool_scope]
        if isinstance(tool_scope, dict):
            return [str(tid) for tid in tool_scope.get("tool_ids", [])]
        return []

    async def _load_tools_as_functions(
        self, tool_ids: list[str],
    ) -> tuple[list[dict], dict[str, ToolDefinition]]:
        """Load tool definitions from DB and convert to OpenAI function calling format.

        Returns:
            (openai_tools, tool_map) where tool_map maps function name → ToolDefinition
        """
        if not tool_ids:
            return [], {}

        result = await self.db.execute(
            select(ToolDefinition).where(
                ToolDefinition.id.in_(tool_ids),
                ToolDefinition.enabled == True,
            )
        )
        tools = list(result.scalars().all())

        openai_tools = []
        tool_map: dict[str, ToolDefinition] = {}

        for tool in tools:
            # Build OpenAI function schema from tool definition
            func_name = self._sanitize_function_name(tool.name)

            # Use input_schema if defined, otherwise build a generic one
            parameters = tool.input_schema or {
                "type": "object",
                "properties": {},
                "required": [],
            }

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": tool.description or f"调用 {tool.name} 工具",
                    "parameters": parameters,
                },
            })
            tool_map[func_name] = tool

        return openai_tools, tool_map

    @staticmethod
    def _sanitize_function_name(name: str) -> str:
        """Sanitize tool name for OpenAI function calling (alphanumeric + underscores)."""
        import re
        # Replace non-alphanumeric chars with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove leading digits
        sanitized = re.sub(r'^[0-9]+', '', sanitized)
        return sanitized or "unnamed_tool"

    # ── Helpers ──────────────────────────────────────────────────
    async def _load_agent(self, agent_id: str) -> Agent | None:
        result = await self.db.execute(
            select(Agent).where(Agent.id == agent_id, Agent.enabled == True)
        )
        return result.scalar_one_or_none()

    async def _get_or_create_session(self, req: InvokeRequest, agent: Agent) -> ConversationSession:
        if req.session_id:
            result = await self.db.execute(
                select(ConversationSession).where(ConversationSession.id == req.session_id)
            )
            session = result.scalar_one_or_none()
            if session:
                return session

        session = ConversationSession(
            id=req.session_id or str(uuid.uuid4()),
            agent_id=agent.id,
            user_id=req.user_id,
            tenant_id=agent.tenant_id,
        )
        self.db.add(session)
        await self.db.flush()
        return session

    async def _get_history(self, session_id: str, limit: int = 6) -> list[Message]:
        result = await self.db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        msgs = list(result.scalars().all())
        msgs.reverse()
        return msgs

    async def _save_message(self, session_id: str, role: str, content: str, trace_id: str):
        msg = Message(session_id=session_id, role=role, content=content, trace_id=trace_id)
        self.db.add(msg)
        await self.db.flush()

    async def _save_session(self, session: ConversationSession):
        session.message_count = (session.message_count or 0) + 1
        await self.db.flush()

    def _risk_precheck(self, agent: Agent, message: str) -> str | None:
        """Check if the message violates risk rules."""
        risk_config = agent.risk_config or {}
        forbidden_keywords = risk_config.get("forbidden_keywords", [])
        for kw in forbidden_keywords:
            if kw in message:
                return "您的问题涉及敏感内容，无法回答。如有需要请联系人工客服。"
        return None

    def _parse_llm_output(self, content: str) -> dict[str, Any]:
        """Try to parse structured JSON from LLM output, fallback to natural language."""
        if not content or not content.strip():
            return {
                "short_answer": "",
                "expanded_answer": None,
                "suggested_followups": [],
            }

        raw = content.strip()

        # Try JSON parsing first (backwards-compatible with models that output JSON)
        try:
            text = raw
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict) and "short_answer" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Natural language mode: use the raw response as-is
        return {
            "short_answer": raw,
            "expanded_answer": None,
            "suggested_followups": [],
        }

    @staticmethod
    def _generate_followups(retrieval, domains: list[str]) -> list[str]:
        """Generate suggested follow-up questions when LLM didn't provide any."""
        followups = []
        if retrieval.hits:
            followups.append("需要更多详细信息吗？")
            followups.append("还有其他问题吗？")
        elif domains:
            followups.append("换个方式描述试试？")
            followups.append("需要人工客服帮助吗？")
        else:
            followups.append("有什么可以帮助你的吗？")
        return followups[:3]

    def _workflow_followups(self, result) -> list[str]:
        if result.status == "waiting_input":
            return ["如何填写？", "需要准备什么材料？"]
        elif result.status == "completed":
            return ["查看办理结果", "还有其他问题"]
        elif result.status == "escalated":
            return ["人工客服工作时间？", "还能自助办理吗？"]
        return []

    def _error_response(self, message: str, req: InvokeRequest) -> InvokeResponse:
        return InvokeResponse(
            session_id=req.session_id or "",
            trace_id=new_trace_id(),
            short_answer=message,
        )

    def _fallback_response(self, session_id: str, trace_id: str, error: str) -> InvokeResponse:
        """Classified error fallback response."""
        error_lower = error.lower()
        if "timeout" in error_lower or "timed out" in error_lower:
            msg = "请求超时，模型处理较慢，请稍后重试。"
        elif "connection" in error_lower or "connect" in error_lower:
            msg = "无法连接到语言模型服务，请检查服务状态后重试。"
        elif "rate" in error_lower or "429" in error_lower:
            msg = "请求频率过高，请稍后重试。"
        else:
            msg = "系统暂时无法响应，请稍后重试或联系人工客服。"

        return InvokeResponse(
            session_id=session_id,
            trace_id=trace_id,
            short_answer=msg,
            suggested_followups=["转人工客服", "稍后重试"],
            metadata={"error_detail": error},
        )

    def _retrieval_only_response(
        self, session_id: str, trace_id: str, retrieval, message: str,
    ) -> InvokeResponse:
        """Return retrieval data directly when LLM is unavailable."""
        short_answer = retrieval.fast_answer or (
            retrieval.hits[0].content if retrieval.hits else "暂无相关资料。"
        )
        citations = []
        for hit in retrieval.hits:
            citations.append(Citation(
                source_id=hit.source_id,
                source_name=hit.source_name,
                content_snippet=hit.content[:100],
                page=hit.page,
                paragraph=hit.paragraph,
                line_start=hit.line_start,
                line_end=hit.line_end,
                score=hit.score,
            ))
        return InvokeResponse(
            session_id=session_id,
            trace_id=trace_id,
            short_answer=short_answer,
            citations=citations,
            suggested_followups=["需要更多信息吗？"],
            metadata={"mode": "retrieval_only"},
        )

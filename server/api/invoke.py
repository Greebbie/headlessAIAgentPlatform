"""The /invoke endpoint — single Headless API entry point."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import get_db
from server.schemas.invoke import InvokeRequest, InvokeResponse
from server.engine.agent_runtime import AgentRuntime

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest, db: AsyncSession = Depends(get_db)):
    """Unified entry point: send a message to any configured agent.

    This is the single API that all client systems integrate with.
    Handles both QA (knowledge retrieval) and workflow (process execution) scenarios.
    """
    runtime = AgentRuntime(db)
    return await runtime.invoke(req)


@router.post("/invoke/stream")
async def invoke_stream(req: InvokeRequest, db: AsyncSession = Depends(get_db)):
    """SSE streaming entry point: same logic as /invoke, with real-time progress events.

    Emits Server-Sent Events:
      - event: status   -> {"stage": "processing"} / {"stage": "retrieval"}
      - event: answer   -> {"content": "<final answer>"}
      - event: done     -> {"session_id": "...", "trace_id": "...", "citations": [...], "followups": [...]}
      - event: error    -> {"detail": "...", "error_type": "...", "error_msg": "..."}
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _sse_event(event: str, data: dict) -> str:
        """Format a single SSE event string."""
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n"

    async def _run_pipeline() -> None:
        """Execute the agent pipeline and push SSE events onto the queue."""
        try:
            # Signal that processing has started
            await queue.put(_sse_event("status", {"stage": "processing"}))

            runtime = AgentRuntime(db)

            # Signal retrieval phase
            await queue.put(_sse_event("status", {"stage": "retrieval"}))

            # Run the full orchestration pipeline
            response: InvokeResponse = await runtime.invoke(req)

            # Emit the final answer
            await queue.put(_sse_event("answer", {"content": response.short_answer}))

            # Emit the done event with metadata
            await queue.put(_sse_event("done", {
                "session_id": response.session_id,
                "trace_id": response.trace_id,
                "citations": [c.model_dump() for c in response.citations],
                "followups": response.suggested_followups,
                "expanded_answer": response.expanded_answer,
                "workflow_card": response.workflow_card.model_dump() if response.workflow_card else None,
                "workflow_status": response.workflow_status,
                "escalated": response.escalated,
                "escalation_reason": response.escalation_reason,
                "metadata": response.metadata,
            }))

        except Exception as exc:
            logger.error(f"SSE pipeline error: {exc}", exc_info=True)
            error_str = str(exc).lower()

            if "timeout" in error_str or "timed out" in error_str:
                error_type = "timeout"
                error_msg = "请求超时，模型处理较慢，请稍后重试。"
            elif "connection" in error_str or "connect" in error_str:
                error_type = "connection"
                error_msg = "无法连接到语言模型服务，请检查服务状态后重试。"
            elif "rate" in error_str or "429" in error_str:
                error_type = "rate_limit"
                error_msg = "请求频率过高，请稍后重试。"
            else:
                error_type = "internal"
                error_msg = "系统暂时无法响应，请稍后重试或联系人工客服。"

            await queue.put(_sse_event("error", {
                "detail": str(exc),
                "error_type": error_type,
                "error_msg": error_msg,
            }))
        finally:
            # Sentinel: signals the generator to stop
            await queue.put(None)

    async def _event_generator():
        """Async generator that yields SSE events from the queue."""
        # Start the pipeline as a background task
        task = asyncio.create_task(_run_pipeline())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            # Ensure the task is cleaned up if the client disconnects
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

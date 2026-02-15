"""Audit logger — records every pipeline event with trace_id."""

from __future__ import annotations

import uuid
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from server.models.audit import AuditTrace


def new_trace_id() -> str:
    return str(uuid.uuid4())


class AuditLogger:
    """Accumulates audit events and flushes them to DB."""

    def __init__(self, session: AsyncSession, trace_id: str, session_id: str, agent_id: str, tenant_id: str = "default"):
        self.db = session
        self.trace_id = trace_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.tenant_id = tenant_id
        self._events: list[AuditTrace] = []
        self._timers: dict[str, float] = {}

    def start_timer(self, key: str):
        self._timers[key] = time.perf_counter()

    def elapsed_ms(self, key: str) -> float:
        start = self._timers.get(key)
        if start is None:
            return 0.0
        return (time.perf_counter() - start) * 1000

    def log(
        self,
        event_type: str,
        event_data: dict[str, Any] | None = None,
        *,
        retrieval_hits: dict | None = None,
        llm_meta: dict | None = None,
        tool_meta: dict | None = None,
        workflow_meta: dict | None = None,
        escalation_reason: str | None = None,
        latency_ms: float | None = None,
    ):
        trace = AuditTrace(
            trace_id=self.trace_id,
            session_id=self.session_id,
            agent_id=self.agent_id,
            tenant_id=self.tenant_id,
            event_type=event_type,
            event_data=event_data,
            retrieval_hits=retrieval_hits,
            llm_meta=llm_meta,
            tool_meta=tool_meta,
            workflow_meta=workflow_meta,
            escalation_reason=escalation_reason,
            latency_ms=latency_ms,
        )
        self._events.append(trace)

    async def flush(self):
        """Write all buffered events to DB."""
        if not self._events:
            return
        self.db.add_all(self._events)
        await self.db.commit()
        self._events.clear()

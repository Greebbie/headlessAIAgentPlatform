"""Audit trace model — full pipeline traceability."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, DateTime, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column

from server.db import Base


class AuditTrace(Base):
    """A single trace record capturing one event in the pipeline."""

    __tablename__ = "audit_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # Correlation id for end-to-end tracking
    trace_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(36), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(36), default="default")

    # Event classification
    event_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # user_input | intent | retrieval | llm_call | tool_call | workflow_step | response | escalation | error

    # Event data (flexible JSON)
    event_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # For retrieval events: which chunks were hit
    retrieval_hits: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # For LLM events: model, prompt tokens, completion tokens
    llm_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # For tool events: tool_id, input, output, success
    tool_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # For workflow events: step_id, status, collected fields
    workflow_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # For escalation: reason
    escalation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timing
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

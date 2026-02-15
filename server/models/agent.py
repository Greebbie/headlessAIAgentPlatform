"""Agent model — the top-level configurable entity."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, Boolean, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column

from server.db import Base


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    # System prompt / persona
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    # Which LLM model to use (override global)
    llm_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Associated workflow id (optional — pure-QA agents have no workflow)
    workflow_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    # Multi-workflow scope: {"workflow_ids": ["wf-1", "wf-2"], "descriptions": {"wf-1": "desc"}}
    workflow_scope: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Knowledge domain ids (list of knowledge source ids this agent can access)
    knowledge_scope: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Tool ids this agent may call
    tool_scope: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Response style config
    response_config: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=lambda: {
        "default_mode": "short",       # short | expanded
        "enable_citations": True,
        "enable_followups": True,
        "max_short_tokens": 120,
        "no_citation_policy": "refuse",  # refuse | escalate | create_ticket
    })
    # Risk / compliance config
    risk_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Tenant isolation
    tenant_id: Mapped[str] = mapped_column(String(36), default="default")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

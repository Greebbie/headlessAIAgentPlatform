"""Agent model — the top-level configurable entity."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, Boolean, DateTime, JSON, ForeignKey
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
    # Reference to a saved LLM configuration (takes priority over llm_model)
    llm_config_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("llm_configs.id", ondelete="SET NULL"), nullable=True,
    )
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
    # Routing mode: "conversational" (default) — LLM-driven with skills as tools
    skill_routing_mode: Mapped[str] = mapped_column(String(16), default="conversational")
    # Tenant isolation
    tenant_id: Mapped[str] = mapped_column(String(36), default="default")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

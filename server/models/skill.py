"""Skill model — first-class capability abstraction for agents."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, Boolean, DateTime, JSON, Integer
from sqlalchemy.orm import Mapped, mapped_column

from server.db import Base


class Skill(Base):
    """A reusable capability that can be bound to agents.

    skill_type determines the execution path:
      - workflow:     wraps a workflow definition
      - tool_call:    wraps one or more tool definitions with function calling
      - knowledge_qa: wraps knowledge sources for RAG retrieval
      - delegate:     delegates to another agent
      - composite:    chains multiple sub-skills sequentially
    """

    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    skill_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Trigger conditions: {"keywords": [], "examples": [], "trigger_description": ""}
    trigger_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Execution config (structure depends on skill_type)
    execution_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Optional input/output schemas for validation
    input_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Routing priority (lower = higher priority)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    tenant_id: Mapped[str] = mapped_column(String(36), default="default")
    # None = manually created; "agent:<uuid>" = auto-managed by capabilities API
    managed_by: Mapped[str | None] = mapped_column(String(128), nullable=True, default=None)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

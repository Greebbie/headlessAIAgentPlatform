"""AgentSkill model — many-to-many binding between agents and skills."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Boolean, DateTime, JSON, Integer
from sqlalchemy.orm import Mapped, mapped_column

from server.db import Base


class AgentSkill(Base):
    """Binds a Skill to an Agent with optional overrides."""

    __tablename__ = "agent_skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    skill_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    # Override the skill's default priority for this agent
    priority_override: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Override execution_config for this agent-skill binding
    config_override: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

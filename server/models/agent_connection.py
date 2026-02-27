"""AgentConnection model — inter-agent collaboration relationships."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, Boolean, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column

from server.db import Base


class AgentConnection(Base):
    """Defines a collaboration relationship between two agents.

    connection_type:
      - delegate:     source can delegate tasks to target
      - orchestrate:  source orchestrates target as a sub-agent
      - peer:         bidirectional peer collaboration
    """

    __tablename__ = "agent_connections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_agent_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    target_agent_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    connection_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Context fields shared during delegation
    shared_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    tenant_id: Mapped[str] = mapped_column(String(36), default="default")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

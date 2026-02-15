"""Conversation session and message models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, DateTime, JSON, Integer
from sqlalchemy.orm import Mapped, mapped_column

from server.db import Base


class ConversationSession(Base):
    """A conversation session between a user and an agent."""

    __tablename__ = "conversation_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(36), default="default")
    # Current workflow execution state (if in a workflow)
    workflow_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Collected data across workflow steps
    collected_data: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    # Context / memory
    context: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    # Status: active | paused | escalated | completed
    status: Mapped[str] = mapped_column(String(16), default="active")
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    """A single message in a conversation."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant | system | tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Structured response parts
    short_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    expanded_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    suggested_followups: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Metadata
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

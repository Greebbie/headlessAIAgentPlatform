"""LLM configuration model — per-tenant LLM provider settings."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, Boolean, DateTime, JSON, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from server.db import Base


class LLMConfig(Base):
    __tablename__ = "llm_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)  # openai_compatible | dashscope | zhipu | local
    base_url: Mapped[str] = mapped_column(Text, nullable=False)
    api_key: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    temperature: Mapped[float] = mapped_column(Float, default=0.3)
    top_p: Mapped[float] = mapped_column(Float, default=1.0)
    max_tokens: Mapped[int] = mapped_column(Integer, default=2048)
    timeout_ms: Mapped[int] = mapped_column(Integer, default=60000)
    extra_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    tenant_id: Mapped[str] = mapped_column(String(36), default="default")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

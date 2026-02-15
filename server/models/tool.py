"""Tool definition model — unified tool gateway registry."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, Boolean, Integer, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column

from server.db import Base


class ToolDefinition(Base):
    """A registered tool (local API, internal service, 3rd-party, etc.)."""

    __tablename__ = "tool_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    # Tool category: api | function | webhook | rpc
    category: Mapped[str] = mapped_column(String(32), default="api")
    # Endpoint URL or function path
    endpoint: Mapped[str] = mapped_column(Text, default="", server_default="")
    method: Mapped[str] = mapped_column(String(8), default="POST")  # GET | POST | PUT | DELETE
    # Input/output JSON schema
    input_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Auth config: {type: "bearer"|"api_key"|"basic"|"none", credentials_ref: "vault_key"}
    auth_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Execution config
    timeout_ms: Mapped[int] = mapped_column(Integer, default=30000)
    max_retries: Mapped[int] = mapped_column(Integer, default=2)
    retry_backoff_ms: Mapped[int] = mapped_column(Integer, default=1000)
    # Sync or async
    is_async: Mapped[bool] = mapped_column(Boolean, default=False)
    # Callback URL for async tools
    callback_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Permission / scope
    required_permission: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Risk level
    risk_level: Mapped[str] = mapped_column(String(16), default="info")  # info | warning | critical
    tenant_id: Mapped[str] = mapped_column(String(36), default="default")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

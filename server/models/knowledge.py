"""Knowledge source and chunk models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, Integer, DateTime, JSON, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from server.db import Base


class KnowledgeSource(Base):
    """A knowledge source (document, FAQ sheet, structured table)."""

    __tablename__ = "knowledge_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # document | faq | structured_table | kv_entity
    # For document: original file path / URL
    source_uri: Mapped[str] = mapped_column(Text, default="")
    # Domain tag for index isolation
    domain: Mapped[str] = mapped_column(String(64), default="default")
    tenant_id: Mapped[str] = mapped_column(String(36), default="default")
    # Processing status: pending | processing | ready | error
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # Metadata (page count, file size, etc.)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeChunk(Base):
    """A chunk from a knowledge source, indexed for retrieval."""

    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id: Mapped[str] = mapped_column(String(36), ForeignKey("knowledge_sources.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Citation info
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paragraph_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Chunk order within source
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    # Vector embedding id (external reference in vector store)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # For structured/KV data: key for fast lookup
    entity_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # Relevance score cache
    cached_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    domain: Mapped[str] = mapped_column(String(64), default="default")
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

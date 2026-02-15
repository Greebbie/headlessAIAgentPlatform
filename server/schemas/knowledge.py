"""Schemas for Knowledge management."""

from __future__ import annotations

from typing import Any
from datetime import datetime

from pydantic import BaseModel


class KnowledgeSourceCreate(BaseModel):
    name: str
    source_type: str  # document | faq | structured_table | kv_entity
    domain: str = "default"
    tenant_id: str = "default"
    metadata: dict[str, Any] | None = None


class KnowledgeSourceOut(BaseModel):
    id: str
    name: str
    source_type: str
    source_uri: str
    domain: str
    tenant_id: str
    status: str
    metadata_: Any
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KVEntityCreate(BaseModel):
    """For fast-answer channel: structured key-value entities."""
    source_id: str
    entity_key: str  # e.g. "市民服务中心电话"
    content: str     # e.g. "0571-12345678"
    domain: str = "default"
    metadata: dict[str, Any] | None = None


class FAQCreate(BaseModel):
    """For FAQ entries."""
    source_id: str
    question: str
    answer: str
    domain: str = "default"
    metadata: dict[str, Any] | None = None


class RetrievalRequest(BaseModel):
    query: str
    domain: str | None = None
    top_k: int = 5
    use_fast_channel: bool = True
    use_rag_channel: bool = True


class RetrievalHit(BaseModel):
    chunk_id: str
    source_id: str
    source_name: str
    content: str
    score: float
    page: int | None = None
    paragraph: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    channel: str  # fast | rag


class RetrievalResponse(BaseModel):
    hits: list[RetrievalHit] = []
    fast_answer: str | None = None
    query: str = ""
    latency_ms: float = 0.0

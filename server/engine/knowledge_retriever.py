"""Knowledge retrieval engine — dual-channel: fast KV + standard RAG."""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from server.models.knowledge import KnowledgeChunk, KnowledgeSource
from server.schemas.knowledge import RetrievalHit, RetrievalResponse


class KnowledgeRetriever:
    """Two-channel retrieval: fast answer + RAG.

    Fast channel: exact / fuzzy match on entity_key (KV/FAQ) → <50ms
    RAG channel:  vector similarity + optional keyword re-rank → <200ms
    """

    def __init__(self, db: AsyncSession, vector_store=None):
        self.db = db
        self.vector_store = vector_store  # injected; None = skip vector search

    # ── Fast Channel ─────────────────────────────────────────────
    async def fast_lookup(self, query: str, domain: str | None = None, top_k: int = 3) -> list[RetrievalHit]:
        """Look up structured KV / FAQ entries by entity_key match."""
        t0 = time.perf_counter()
        stmt = (
            select(KnowledgeChunk, KnowledgeSource.name)
            .join(KnowledgeSource, KnowledgeChunk.source_id == KnowledgeSource.id)
            .where(KnowledgeChunk.entity_key.isnot(None))
        )
        if domain:
            stmt = stmt.where(KnowledgeChunk.domain == domain)

        # Simple keyword containment match on entity_key
        stmt = stmt.where(
            or_(
                KnowledgeChunk.entity_key.contains(query),
                KnowledgeChunk.content.contains(query),
            )
        ).limit(top_k)

        result = await self.db.execute(stmt)
        rows = result.all()

        hits = []
        for chunk, source_name in rows:
            hits.append(RetrievalHit(
                chunk_id=chunk.id,
                source_id=chunk.source_id,
                source_name=source_name,
                content=chunk.content,
                score=1.0,  # exact match
                page=chunk.page_number,
                paragraph=chunk.paragraph_index,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                channel="fast",
            ))
        return hits

    # ── RAG Channel ──────────────────────────────────────────────
    async def rag_search(self, query: str, domain: str | None = None, top_k: int = 5) -> list[RetrievalHit]:
        """Vector similarity search + keyword fallback."""
        hits: list[RetrievalHit] = []

        # 1. Vector search (if vector store is available)
        if self.vector_store is not None:
            vector_results = await self._vector_search(query, domain, top_k)
            hits.extend(vector_results)

        # 2. Keyword fallback / supplementary (SQL LIKE)
        if len(hits) < top_k:
            remaining = top_k - len(hits)
            keyword_hits = await self._keyword_search(query, domain, remaining)
            # Deduplicate
            seen_ids = {h.chunk_id for h in hits}
            for h in keyword_hits:
                if h.chunk_id not in seen_ids:
                    hits.append(h)

        return hits[:top_k]

    async def _vector_search(self, query: str, domain: str | None, top_k: int) -> list[RetrievalHit]:
        """Placeholder: vector similarity via FAISS/Milvus."""
        # In production, this would:
        # 1. Embed the query via embedding model
        # 2. Search the vector index (filtered by domain)
        # 3. Fetch chunk metadata from DB
        # 4. Return scored hits
        return []

    async def _keyword_search(self, query: str, domain: str | None, top_k: int) -> list[RetrievalHit]:
        """SQL keyword search as fallback."""
        stmt = (
            select(KnowledgeChunk, KnowledgeSource.name)
            .join(KnowledgeSource, KnowledgeChunk.source_id == KnowledgeSource.id)
            .where(KnowledgeChunk.content.contains(query))
        )
        if domain:
            stmt = stmt.where(KnowledgeChunk.domain == domain)
        stmt = stmt.limit(top_k)

        result = await self.db.execute(stmt)
        rows = result.all()

        return [
            RetrievalHit(
                chunk_id=chunk.id,
                source_id=chunk.source_id,
                source_name=source_name,
                content=chunk.content,
                score=0.5,  # keyword match gets lower score
                page=chunk.page_number,
                paragraph=chunk.paragraph_index,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                channel="rag",
            )
            for chunk, source_name in rows
        ]

    # ── Unified Retrieve ─────────────────────────────────────────
    async def retrieve(
        self, query: str, domain: str | None = None, top_k: int = 5,
        use_fast: bool = True, use_rag: bool = True,
    ) -> RetrievalResponse:
        t0 = time.perf_counter()
        all_hits: list[RetrievalHit] = []
        fast_answer: str | None = None

        # Fast channel first
        if use_fast:
            fast_hits = await self.fast_lookup(query, domain, top_k=3)
            if fast_hits:
                fast_answer = fast_hits[0].content
                all_hits.extend(fast_hits)

        # RAG channel
        if use_rag and len(all_hits) < top_k:
            rag_hits = await self.rag_search(query, domain, top_k=top_k - len(all_hits))
            all_hits.extend(rag_hits)

        # Sort by score descending
        all_hits.sort(key=lambda h: h.score, reverse=True)

        return RetrievalResponse(
            hits=all_hits[:top_k],
            fast_answer=fast_answer,
            query=query,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

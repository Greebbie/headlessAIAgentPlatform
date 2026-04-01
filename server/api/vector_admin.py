"""Vector store administration API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import get_db
from server.middleware.auth import get_current_user
from server.engine.vector_store import get_vector_store
from server.models.knowledge import KnowledgeChunk

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/stats")
async def vector_stats():
    """Get vector index statistics (count, dimension, memory estimate)."""
    vs = get_vector_store()
    if vs is None:
        return {
            "index_count": 0,
            "dimension": 0,
            "memory_usage_mb": 0.0,
            "status": "unavailable",
        }
    return {
        "index_count": vs.count(),
        "dimension": vs.dimension,
        "memory_usage_mb": round(vs.memory_usage_mb(), 3) if hasattr(vs, "memory_usage_mb") else 0.0,
        "index_type": "hnsw" if getattr(vs, "is_hnsw", False) else "pgvector" if type(vs).__name__ == "PgVectorStore" else "flat",
        "status": "ready",
    }


@router.post("/rebuild")
async def rebuild_index(db: AsyncSession = Depends(get_db)):
    """Trigger a full vector index rebuild from all knowledge chunks."""
    vs = get_vector_store()
    if vs is None:
        return {"message": "Vector store not available", "status": "error"}

    # Load all chunks from DB
    result = await db.execute(
        select(KnowledgeChunk.id, KnowledgeChunk.content, KnowledgeChunk.domain)
    )
    rows = result.all()

    if not rows:
        return {"message": "No chunks to index", "status": "empty"}

    batch = [
        {"chunk_id": row.id, "text": row.content, "domain": row.domain}
        for row in rows
    ]

    try:
        # For FAISS: reset to a fresh HNSW index before rebuilding
        # For pgvector: delete all existing vectors first
        if hasattr(vs, "_create_hnsw_index"):
            vs._create_hnsw_index()
        else:
            # pgvector or other DB-backed store: delete all existing chunk IDs
            all_chunk_ids = [row.id for row in rows]
            vs.delete(all_chunk_ids)

        vs.add_batch(batch)
        vs.save()
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        return {"message": f"Rebuild failed: {e}", "status": "error"}

    backend_type = type(vs).__name__
    return {
        "message": f"Index rebuilt with {len(batch)} vectors",
        "status": "completed",
        "count": len(batch),
        "index_type": "hnsw" if getattr(vs, "is_hnsw", False) else "pgvector" if "PgVector" in backend_type else "flat",
    }


@router.get("/health")
async def vector_health():
    """Check whether the vector store is initialized and operational."""
    vs = get_vector_store()
    if vs is None:
        return {
            "initialized": False,
            "backend": "faiss",
            "status": "unavailable",
        }
    backend_name = type(vs).__name__
    return {
        "initialized": True,
        "backend": "pgvector" if "PgVector" in backend_name else "faiss",
        "index_count": vs.count(),
        "dimension": vs.dimension,
        "index_type": "hnsw" if getattr(vs, "is_hnsw", False) else "pgvector" if "PgVector" in backend_name else "flat",
        "status": "healthy",
    }

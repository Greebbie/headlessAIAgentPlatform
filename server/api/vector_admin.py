"""Vector store administration API."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/stats")
async def vector_stats():
    """Get vector index statistics (count, dimension, memory estimate)."""
    return {
        "index_count": 0,
        "dimension": 768,
        "memory_usage_mb": 0.0,
        "status": "placeholder",
    }


@router.post("/rebuild")
async def rebuild_index():
    """Trigger a full vector index rebuild."""
    return {
        "message": "Index rebuild queued",
        "status": "placeholder",
    }


@router.get("/health")
async def vector_health():
    """Check whether the vector store is initialized and operational."""
    return {
        "initialized": False,
        "backend": "faiss",
        "status": "placeholder",
    }

"""Knowledge management API — upload, manage, search."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import get_db
from server.models.knowledge import KnowledgeSource, KnowledgeChunk
from server.schemas.knowledge import (
    KnowledgeSourceCreate, KnowledgeSourceOut,
    KVEntityCreate, FAQCreate,
    RetrievalRequest, RetrievalResponse,
)
from server.engine.knowledge_retriever import KnowledgeRetriever

router = APIRouter()


# ── Knowledge Sources ────────────────────────────────────────────

@router.get("/sources", response_model=list[KnowledgeSourceOut])
async def list_sources(tenant_id: str = "default", domain: str | None = None, db: AsyncSession = Depends(get_db)):
    stmt = select(KnowledgeSource).where(KnowledgeSource.tenant_id == tenant_id)
    if domain:
        stmt = stmt.where(KnowledgeSource.domain == domain)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/sources", response_model=KnowledgeSourceOut, status_code=201)
async def create_source(body: KnowledgeSourceCreate, db: AsyncSession = Depends(get_db)):
    source = KnowledgeSource(
        name=body.name,
        source_type=body.source_type,
        domain=body.domain,
        tenant_id=body.tenant_id,
        metadata_=body.metadata,
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


@router.delete("/sources/{source_id}", status_code=204)
async def delete_source(source_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(KnowledgeSource).where(KnowledgeSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(404, "Source not found")
    # Delete chunks
    chunks = await db.execute(select(KnowledgeChunk).where(KnowledgeChunk.source_id == source_id))
    for chunk in chunks.scalars().all():
        await db.delete(chunk)
    await db.delete(source)
    await db.commit()


# ── KV Entities (fast-answer channel) ────────────────────────────

@router.post("/kv", status_code=201)
async def add_kv_entity(body: KVEntityCreate, db: AsyncSession = Depends(get_db)):
    chunk = KnowledgeChunk(
        source_id=body.source_id,
        content=body.content,
        entity_key=body.entity_key,
        domain=body.domain,
        metadata_=body.metadata,
    )
    db.add(chunk)
    # Update source chunk count
    result = await db.execute(select(KnowledgeSource).where(KnowledgeSource.id == body.source_id))
    source = result.scalar_one_or_none()
    if source:
        source.chunk_count = (source.chunk_count or 0) + 1
    await db.commit()
    return {"id": chunk.id, "entity_key": body.entity_key}


# ── FAQ entries ──────────────────────────────────────────────────

@router.post("/faq", status_code=201)
async def add_faq(body: FAQCreate, db: AsyncSession = Depends(get_db)):
    chunk = KnowledgeChunk(
        source_id=body.source_id,
        content=body.answer,
        entity_key=body.question,
        domain=body.domain,
        metadata_={"question": body.question, **(body.metadata or {})},
    )
    db.add(chunk)
    result = await db.execute(select(KnowledgeSource).where(KnowledgeSource.id == body.source_id))
    source = result.scalar_one_or_none()
    if source:
        source.chunk_count = (source.chunk_count or 0) + 1
    await db.commit()
    return {"id": chunk.id, "question": body.question}


# ── Retrieval test ───────────────────────────────────────────────

@router.post("/search", response_model=RetrievalResponse)
async def search(body: RetrievalRequest, db: AsyncSession = Depends(get_db)):
    """Test knowledge retrieval — used from the console for debugging."""
    retriever = KnowledgeRetriever(db)
    return await retriever.retrieve(
        query=body.query,
        domain=body.domain,
        top_k=body.top_k,
        use_fast=body.use_fast_channel,
        use_rag=body.use_rag_channel,
    )


# ── Document Upload ─────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".txt", ".md"}


def _sanitize_filename(filename: str) -> str:
    """Strip directory components and reject path-traversal attempts."""
    import os
    # Take only the basename to block directory traversal
    name = os.path.basename(filename)
    # Reject anything that still looks suspicious
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid filename")
    return name


def _split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks: by paragraphs first, then by chunk_size."""
    # Split into paragraphs (double newline or more)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If a single paragraph exceeds chunk_size, split it further
        if len(paragraph) > chunk_size:
            # Flush any accumulated content first
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # Split the long paragraph by chunk_size with overlap
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end])
                start = end - chunk_overlap
            continue

        # Check if adding this paragraph would exceed chunk_size
        candidate = (current_chunk + "\n\n" + paragraph).strip() if current_chunk else paragraph
        if len(candidate) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Start new chunk with overlap: carry over the tail of the previous chunk
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _extract_entity_key(content: str) -> str:
    """Auto-generate entity_key from the first sentence of a chunk."""
    # Try splitting by common sentence terminators
    for delimiter in ["。", ".", "！", "!", "？", "?", "\n"]:
        idx = content.find(delimiter)
        if 0 < idx < 200:
            return content[: idx + 1].strip()
    # Fallback: first 100 characters
    return content[:100].strip()


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    source_id: str = Form(...),
    domain: str = Form("default"),
    chunk_size: int = Form(500, ge=50, le=10000),
    chunk_overlap: int = Form(50, ge=0, le=5000),
    db: AsyncSession = Depends(get_db),
):
    """Upload a text document, split into chunks, and store in the knowledge base."""
    import os

    # ── Validate chunk_overlap < chunk_size ──────────────────────
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            400,
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})",
        )

    # ── Validate file extension ──────────────────────────────────
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    safe_name = _sanitize_filename(file.filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # ── Verify source exists ─────────────────────────────────────
    result = await db.execute(
        select(KnowledgeSource).where(KnowledgeSource.id == source_id)
    )
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(404, f"Knowledge source '{source_id}' not found")

    # ── Read file content ────────────────────────────────────────
    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 encoded text")

    if not text.strip():
        raise HTTPException(400, "Uploaded file is empty")

    # ── Split into chunks ────────────────────────────────────────
    chunks = _split_into_chunks(text, chunk_size, chunk_overlap)

    # ── Store chunks ─────────────────────────────────────────────
    created_ids: list[str] = []
    for idx, chunk_content in enumerate(chunks):
        entity_key = _extract_entity_key(chunk_content)
        chunk = KnowledgeChunk(
            source_id=source_id,
            content=chunk_content,
            entity_key=entity_key,
            domain=domain,
            chunk_index=idx,
            metadata_={"filename": safe_name, "chunk_size": chunk_size},
        )
        db.add(chunk)
        created_ids.append(chunk.id)

    # ── Update source chunk_count and status ─────────────────────
    source.chunk_count = (source.chunk_count or 0) + len(chunks)
    source.status = "ready"

    await db.commit()

    return {
        "source_id": source_id,
        "filename": safe_name,
        "chunk_count": len(chunks),
        "chunk_ids": created_ids,
    }

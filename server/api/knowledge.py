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


@router.get("/sources/{source_id}/chunks")
async def list_chunks(source_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(KnowledgeChunk).where(KnowledgeChunk.source_id == source_id)
        .order_by(KnowledgeChunk.chunk_index)
    )
    chunks = result.scalars().all()
    return [
        {
            "id": c.id,
            "entity_key": c.entity_key,
            "content": c.content,
            "domain": c.domain,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]


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

    # Embed into vector store (non-fatal)
    try:
        from server.engine.vector_store import get_vector_store
        vs = get_vector_store()
        if vs:
            vs.add(chunk.id, body.content, domain=body.domain)
            vs.save()
    except Exception:
        pass  # vector embedding failure should not block API response

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

    # Embed into vector store (non-fatal)
    try:
        from server.engine.vector_store import get_vector_store
        vs = get_vector_store()
        if vs:
            vs.add(chunk.id, body.answer, domain=body.domain)
            vs.save()
    except Exception:
        pass

    return {"id": chunk.id, "question": body.question}


# ── Retrieval test ───────────────────────────────────────────────

@router.post("/search", response_model=RetrievalResponse)
async def search(body: RetrievalRequest, db: AsyncSession = Depends(get_db)):
    """Test knowledge retrieval — used from the console for debugging."""
    from server.engine.vector_store import get_vector_store
    from server.runtime_config import runtime_config
    retriever = KnowledgeRetriever(
        db, vector_store=get_vector_store(),
        runtime_cfg=runtime_config.all(),
    )
    return await retriever.retrieve(
        query=body.query,
        domain=body.domain,
        top_k=body.top_k,
        use_fast=body.use_fast_channel,
        use_rag=body.use_rag_channel,
    )


# ── Document Upload ─────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def _sanitize_filename(filename: str) -> str:
    """Strip directory components and reject path-traversal attempts."""
    import os
    # Take only the basename to block directory traversal
    name = os.path.basename(filename)
    # Reject anything that still looks suspicious
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid filename")
    return name


def _extract_text_from_pdf(raw_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise HTTPException(
            400,
            "PDF support requires pypdf. Install with: pip install 'hlab-agent-builder[rag]'",
        )
    import io
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_text_from_docx(raw_bytes: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        raise HTTPException(
            400,
            "DOCX support requires python-docx. Install with: pip install 'hlab-agent-builder[rag]'",
        )
    import io
    doc = DocxDocument(io.BytesIO(raw_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


# Sentence-ending punctuation for recursive splitting
_SENTENCE_DELIMITERS = "。.！!？?；;"


def _find_sentence_boundary(text: str, target: int, window: int = 50) -> int:
    """Find the nearest sentence boundary around `target` position.

    Looks within [target - window, target + window] for sentence delimiters.
    Falls back to target if no boundary found.
    """
    start = max(0, target - window)
    end = min(len(text), target + window)
    region = text[start:end]

    # Search for the closest delimiter to the midpoint of the region
    mid = target - start
    best = -1
    best_dist = window + 1

    for i, ch in enumerate(region):
        if ch in _SENTENCE_DELIMITERS:
            dist = abs(i - mid)
            if dist < best_dist:
                best_dist = dist
                best = start + i + 1  # split after the delimiter

    # Also try jieba word boundaries
    if best == -1:
        try:
            import jieba
            tokens = jieba.lcut(region)
            pos = start
            for tok in tokens:
                pos += len(tok)
                dist = abs(pos - target)
                if dist < best_dist:
                    best_dist = dist
                    best = pos
        except ImportError:
            pass

    return best if best != -1 else target


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursively split text using a hierarchy of separators.

    Separator hierarchy: paragraph (\\n\\n) -> line (\\n) -> sentence -> character.
    Overlap cuts prefer sentence boundaries via _find_sentence_boundary.
    """
    # Level 1: Split by paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            # Flush accumulated content
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # Level 2: Split long paragraph by lines
            lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
            line_chunk = ""
            for line in lines:
                if len(line) > chunk_size:
                    if line_chunk:
                        chunks.append(line_chunk)
                        line_chunk = ""
                    # Level 3: Split long line at sentence boundaries
                    start = 0
                    while start < len(line):
                        end = start + chunk_size
                        if end < len(line):
                            boundary = _find_sentence_boundary(line, end)
                            end = boundary
                        chunks.append(line[start:end].strip())
                        # Overlap: step back but align to sentence boundary
                        next_start = end - chunk_overlap
                        if chunk_overlap > 0 and next_start > start:
                            next_start = _find_sentence_boundary(
                                line, next_start, window=chunk_overlap // 2 or 20,
                            )
                        start = max(next_start, start + 1)  # ensure progress
                    continue

                candidate = (line_chunk + "\n" + line).strip() if line_chunk else line
                if len(candidate) > chunk_size and line_chunk:
                    chunks.append(line_chunk)
                    line_chunk = line
                else:
                    line_chunk = candidate

            if line_chunk:
                # Check if it fits with current_chunk
                chunks.append(line_chunk)
            continue

        # Normal-sized paragraph: accumulate
        candidate = (current_chunk + "\n\n" + paragraph).strip() if current_chunk else paragraph
        if len(candidate) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Overlap from previous chunk at sentence boundary
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                overlap_start = len(current_chunk) - chunk_overlap
                overlap_start = _find_sentence_boundary(
                    current_chunk, overlap_start, window=chunk_overlap // 2 or 20,
                )
                current_chunk = current_chunk[overlap_start:] + "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if c.strip()]


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

    if ext == ".pdf":
        text = _extract_text_from_pdf(raw_bytes)
    elif ext == ".docx":
        text = _extract_text_from_docx(raw_bytes)
    else:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "File must be UTF-8 encoded text")

    if not text.strip():
        raise HTTPException(400, "Uploaded file is empty")

    # ── Split into chunks ────────────────────────────────────────
    chunks = _recursive_split(text, chunk_size, chunk_overlap)

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

    # Embed all chunks into vector store (non-fatal)
    try:
        from server.engine.vector_store import get_vector_store
        vs = get_vector_store()
        if vs and created_ids:
            batch = [
                {"chunk_id": cid, "text": ct, "domain": domain}
                for cid, ct in zip(created_ids, [c for c in chunks])
            ]
            vs.add_batch(batch)
            vs.save()
    except Exception:
        pass  # vector embedding failure should not block upload response

    return {
        "source_id": source_id,
        "filename": safe_name,
        "chunk_count": len(chunks),
        "chunk_ids": created_ids,
    }

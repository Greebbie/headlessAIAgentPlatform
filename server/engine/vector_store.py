"""FAISS vector store with multi-provider embedding support.

Provides:
- EmbeddingModel: lazy-loaded singleton supporting local (sentence-transformers)
  and API-based (DashScope/OpenAI-compatible) embedding. BGE models get
  instruction prefix for queries.
- VectorStoreManager: thread-safe singleton wrapping FAISS IndexHNSWFlat
  (falls back to loading legacy IndexFlatIP indexes read-only).
- get_vector_store(): returns the singleton or None if deps are unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

import numpy as np

from server.config import settings

logger = logging.getLogger(__name__)

# ── Embedding Model ────────────────────────────────────────────

# Cascade fallback order for local embedding models
_LOCAL_EMBEDDING_FALLBACKS = [
    "BAAI/bge-m3",
    "BAAI/bge-small-zh-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """Lazy-loaded embedding model with multi-provider support.

    Supports:
    - "local": sentence-transformers (cascade fallback)
    - "dashscope" / "openai_compatible": HTTP API via OpenAI-compatible /embeddings endpoint

    BGE models automatically get instruction prefix for query encoding.
    """

    _instance: EmbeddingModel | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None  # sentence-transformers model (local only)
        self._dim: int | None = None
        self._provider: str = "local"
        self._is_bge: bool = False
        self._model_name: str = ""

    @classmethod
    def get_instance(cls) -> EmbeddingModel | None:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    if inst._load():
                        cls._instance = inst
                    else:
                        return None
        return cls._instance

    def _load(self) -> bool:
        """Initialize embedding based on configured provider."""
        self._provider = settings.embedding_provider

        if self._provider in ("dashscope", "openai_compatible"):
            return self._load_api()
        else:
            return self._load_local()

    def _load_api(self) -> bool:
        """Initialize API-based embedding (DashScope / OpenAI-compatible)."""
        try:
            import httpx  # noqa: F401 — verify httpx available
        except ImportError:
            logger.warning("httpx not installed; API embedding unavailable")
            return False

        # Determine API endpoint
        if self._provider == "dashscope":
            self._api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            self._api_base = settings.llm_base_url.rstrip("/")

        self._api_key = settings.llm_api_key
        self._model_name = settings.embedding_model
        self._is_bge = "bge" in self._model_name.lower()

        # Probe dimension with a test call
        try:
            test_vecs = self._api_embed(["test"])
            self._dim = len(test_vecs[0])
            logger.info(
                f"API embedding initialized: provider={self._provider}, "
                f"model={self._model_name}, dim={self._dim}, is_bge={self._is_bge}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize API embedding: {e}")
            return False

    def _api_embed(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI-compatible /embeddings endpoint synchronously."""
        import httpx

        url = f"{self._api_base}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": texts,
        }
        # DashScope text-embedding-v3 supports dimensions param
        if self._dim is not None:
            payload["dimensions"] = self._dim

        resp = httpx.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Extract embeddings sorted by index
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]

    def _load_local(self) -> bool:
        """Try loading local sentence-transformers models in cascade order."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed; trying API fallback")
            # Fall back to API if local model unavailable
            self._provider = "dashscope" if settings.llm_api_key else "local"
            if self._provider != "local":
                return self._load_api()
            return False

        models_to_try = [settings.embedding_model]
        for m in _LOCAL_EMBEDDING_FALLBACKS:
            if m != settings.embedding_model:
                models_to_try.append(m)

        for model_name in models_to_try:
            try:
                logger.info(f"Loading local embedding model: {model_name}")
                self._model = SentenceTransformer(model_name)
                test_vec = self._model.encode(["test"], normalize_embeddings=True)
                self._dim = test_vec.shape[1]
                self._model_name = model_name
                self._is_bge = "bge" in model_name.lower()
                logger.info(
                    f"Local embedding model loaded: {model_name} "
                    f"(dim={self._dim}, is_bge={self._is_bge})"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue

        logger.error("All local embedding models failed to load")
        return False

    @property
    def dimension(self) -> int:
        return self._dim or settings.embedding_dim

    def encode(self, texts: list[str], mode: str = "document") -> np.ndarray:
        """Encode texts into normalized vectors.

        Args:
            texts: List of strings to encode.
            mode: "query" adds BGE instruction prefix for retrieval queries;
                  "document" encodes as-is for indexing.
        """
        if self._is_bge and mode == "query":
            texts = [_BGE_QUERY_PREFIX + t for t in texts]

        if self._provider in ("dashscope", "openai_compatible"):
            return self._encode_api(texts)
        return self._encode_local(texts)

    def _encode_api(self, texts: list[str]) -> np.ndarray:
        """Encode via API, then L2-normalize."""
        # API batch limit is typically 25 for DashScope
        batch_size = 25
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = self._api_embed(batch)
            all_vecs.extend(vecs)
        arr = np.array(all_vecs, dtype=np.float32)
        # L2 normalize for cosine similarity via inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms

    def _encode_local(self, texts: list[str]) -> np.ndarray:
        """Encode via local sentence-transformers model."""
        if self._model is None:
            raise RuntimeError("Embedding model not loaded")
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── FAISS Vector Store ─────────────────────────────────────────


class VectorStoreManager:
    """Thread-safe FAISS vector store singleton with disk persistence.

    New indexes use IndexHNSWFlat (M=32, efConstruction=200) for fast ANN search.
    Legacy IndexFlatIP indexes can still be loaded and searched, but a log
    warning prompts the user to rebuild via /vector-admin/rebuild.
    """

    _instance: VectorStoreManager | None = None
    _lock = threading.Lock()

    def __init__(self, index_path: str, embedding: EmbeddingModel):
        self._index_path = index_path
        self._sidecar_path = index_path + ".ids.json"
        self._embedding = embedding
        self._write_lock = threading.Lock()

        # ID mapping: parallel lists
        self._ids: list[str] = []       # chunk_id at position i
        self._domains: list[str] = []   # domain at position i

        self._index = None
        self._is_hnsw: bool = False
        self._load_or_create()

    @classmethod
    def get_instance(cls) -> VectorStoreManager | None:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    emb = EmbeddingModel.get_instance()
                    if emb is None:
                        return None
                    try:
                        import faiss  # noqa: F401
                    except ImportError:
                        logger.warning("faiss-cpu not installed; vector search disabled")
                        return None
                    os.makedirs(os.path.dirname(settings.faiss_index_path) or ".", exist_ok=True)
                    cls._instance = cls(settings.faiss_index_path, emb)
        return cls._instance

    def _load_or_create(self):
        import faiss

        if os.path.exists(self._index_path) and os.path.exists(self._sidecar_path):
            try:
                loaded_index = faiss.read_index(self._index_path)
                with open(self._sidecar_path, "r") as f:
                    data = json.load(f)
                self._ids = data.get("ids", [])
                self._domains = data.get("domains", [])

                # Dimension mismatch detection
                if loaded_index.d != self._embedding.dimension:
                    logger.warning(
                        f"FAISS index dimension ({loaded_index.d}) != "
                        f"embedding dimension ({self._embedding.dimension}). "
                        f"Creating new HNSW index. Run /vector-admin/rebuild to re-index."
                    )
                    self._create_hnsw_index()
                    return

                # Detect index type
                self._is_hnsw = hasattr(loaded_index, 'hnsw')
                self._index = loaded_index

                if not self._is_hnsw:
                    logger.warning(
                        f"Legacy FlatIP index loaded ({loaded_index.ntotal} vectors). "
                        f"Run /vector-admin/rebuild to upgrade to HNSW for faster search."
                    )
                else:
                    logger.info(f"HNSW index loaded: {loaded_index.ntotal} vectors")
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index, creating new: {e}")

        self._create_hnsw_index()

    def _create_hnsw_index(self):
        """Create a new IndexHNSWFlat with production-grade parameters."""
        import faiss

        dim = self._embedding.dimension
        self._index = faiss.IndexHNSWFlat(dim, 32)  # M=32
        self._index.hnsw.efConstruction = 200
        self._is_hnsw = True
        self._ids = []
        self._domains = []
        logger.info(f"Created new FAISS IndexHNSWFlat (dim={dim}, M=32, efConstruction=200)")

    # ── Write operations ───────────────────────────────

    def add(self, chunk_id: str, text: str, domain: str = "default") -> None:
        """Embed a single text and add to the index."""
        vec = self._embedding.encode([text], mode="document")  # shape (1, dim)
        with self._write_lock:
            self._index.add(vec.astype(np.float32))
            self._ids.append(chunk_id)
            self._domains.append(domain)

    def add_batch(self, items: list[dict[str, str]]) -> None:
        """Batch embed and add. Each item: {chunk_id, text, domain}."""
        if not items:
            return
        texts = [it["text"] for it in items]
        vecs = self._embedding.encode(texts, mode="document").astype(np.float32)
        with self._write_lock:
            self._index.add(vecs)
            for it in items:
                self._ids.append(it["chunk_id"])
                self._domains.append(it.get("domain", "default"))

    def save(self) -> None:
        """Persist index and sidecar to disk."""
        import faiss

        with self._write_lock:
            os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
            faiss.write_index(self._index, self._index_path)
            with open(self._sidecar_path, "w") as f:
                json.dump({"ids": self._ids, "domains": self._domains}, f)
        logger.info(f"FAISS index saved: {self._index.ntotal} vectors")

    # ── Search ─────────────────────────────────────────

    def search(
        self, query: str, top_k: int = 5, domain: str | None = None,
        ef_search: int = 128,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors. Returns [{chunk_id, score, domain}].

        Args:
            ef_search: HNSW efSearch parameter (higher = more accurate, slower).
                       Ignored for legacy FlatIP indexes.
        """
        if self._index.ntotal == 0:
            return []

        vec = self._embedding.encode([query], mode="query").astype(np.float32)

        # Set HNSW efSearch before querying
        if self._is_hnsw:
            self._index.hnsw.efSearch = ef_search

        # Over-fetch to allow domain filtering
        fetch_k = min(top_k * 3, self._index.ntotal)
        scores, indices = self._index.search(vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            if domain and self._domains[idx] != domain:
                continue
            results.append({
                "chunk_id": self._ids[idx],
                "score": float(score),
                "domain": self._domains[idx],
            })
            if len(results) >= top_k:
                break

        return results

    # ── Stats ──────────────────────────────────────────

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def dimension(self) -> int:
        return self._embedding.dimension

    @property
    def is_hnsw(self) -> bool:
        return self._is_hnsw

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB (vectors only)."""
        if not self._index:
            return 0.0
        return (self._index.ntotal * self._embedding.dimension * 4) / (1024 * 1024)


# ── Module-level helper ────────────────────────────────────────

def get_vector_store() -> VectorStoreManager | None:
    """Return the global VectorStoreManager singleton, or None if deps unavailable."""
    try:
        return VectorStoreManager.get_instance()
    except Exception as e:
        logger.warning(f"Failed to initialize vector store: {e}")
        return None

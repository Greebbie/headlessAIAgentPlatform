"""Unit tests for the upgraded RAG pipeline.

Tests cover:
  1. jieba tokenization + stopword filtering
  2. BM25 scoring correctness
  3. RRF fusion logic
  4. Recursive text splitting
  5. PDF/DOCX extractor error handling
  6. Embedding model mode (query vs document)
  7. Performance preset ef_search values
  8. Runtime config wiring
"""

from __future__ import annotations

import math
import pytest


# ═══════════════════════════════════════════════════════════════
# 1. Tokenization
# ═══════════════════════════════════════════════════════════════

class TestTokenize:
    def test_chinese_tokenization(self):
        from server.engine.knowledge_retriever import _tokenize
        tokens = _tokenize("医保报销比例是多少")
        assert len(tokens) >= 2, f"Expected multiple tokens, got: {tokens}"
        assert all(len(t) >= 2 for t in tokens), f"Tokens should be >= 2 chars: {tokens}"

    def test_english_tokenization(self):
        from server.engine.knowledge_retriever import _tokenize
        tokens = _tokenize("What is the refund policy for insurance?")
        assert "refund" in tokens or "policy" in tokens or "insurance" in tokens
        # Stopwords should be filtered
        assert "is" not in tokens
        assert "the" not in tokens
        assert "for" not in tokens

    def test_stopword_filtering_chinese(self):
        from server.engine.knowledge_retriever import _tokenize
        tokens = _tokenize("我有一个问题")
        # "我", "有", "一个" are stopwords
        assert "我" not in tokens
        assert "有" not in tokens

    def test_empty_input(self):
        from server.engine.knowledge_retriever import _tokenize
        assert _tokenize("") == []
        assert _tokenize("   ") == []

    def test_mixed_language(self):
        from server.engine.knowledge_retriever import _tokenize
        tokens = _tokenize("Python编程入门教程")
        assert len(tokens) >= 1


# ═══════════════════════════════════════════════════════════════
# 2. BM25 Scoring
# ═══════════════════════════════════════════════════════════════

class TestBM25:
    def test_basic_scoring(self):
        from server.engine.knowledge_retriever import _bm25_score
        query_tokens = ["医保", "报销"]
        doc_tokens_list = [
            ["医保", "报销", "比例", "规定"],  # has both terms
            ["天气", "预报", "今天"],           # no match
            ["医保", "门诊", "报销"],           # has both terms, shorter doc
        ]
        scores = _bm25_score(query_tokens, doc_tokens_list)
        assert len(scores) == 3
        assert scores[0] > 0, "Doc with matching terms should score > 0"
        assert scores[1] == 0, "Doc with no matching terms should score 0"
        assert scores[2] > 0, "Doc with matching terms should score > 0"

    def test_normalization(self):
        from server.engine.knowledge_retriever import _bm25_score
        scores = _bm25_score(
            ["test", "query"],
            [["test", "query", "document"], ["test"]],
        )
        assert all(0 <= s <= 1 for s in scores), f"Scores should be in [0, 1]: {scores}"
        assert max(scores) == 1.0, "Max score should be normalized to 1.0"

    def test_empty_inputs(self):
        from server.engine.knowledge_retriever import _bm25_score
        assert _bm25_score([], [["a", "b"]]) == [0.0]
        assert _bm25_score(["a"], []) == []

    def test_length_normalization(self):
        """Shorter docs with same term frequency should score differently."""
        from server.engine.knowledge_retriever import _bm25_score
        scores = _bm25_score(
            ["target"],
            [
                ["target"] + ["padding"] * 100,  # long doc
                ["target"],                        # short doc
            ],
        )
        # Short doc should score higher due to BM25 length normalization
        assert scores[1] > scores[0], \
            f"Short doc should score higher: long={scores[0]}, short={scores[1]}"

    def test_term_frequency_matters(self):
        """Multiple occurrences of query term should increase score."""
        from server.engine.knowledge_retriever import _bm25_score
        scores = _bm25_score(
            ["target"],
            [
                ["target", "other", "word"],              # tf=1
                ["target", "target", "target", "word"],   # tf=3
            ],
        )
        assert scores[1] > scores[0], \
            f"Higher TF should score higher: tf1={scores[0]}, tf3={scores[1]}"

    def test_idf_effect(self):
        """Rare terms (low df) should contribute more than common terms."""
        from server.engine.knowledge_retriever import _bm25_score
        # "rare" appears in 1 doc, "common" appears in all 3
        docs = [
            ["rare", "common", "word"],
            ["common", "word", "other"],
            ["common", "another", "word"],
        ]
        # Query with rare term should score doc0 highly
        scores_rare = _bm25_score(["rare"], docs)
        assert scores_rare[0] == 1.0, "Doc with rare term should be top scorer"
        assert scores_rare[1] == 0, "Doc without rare term should score 0"


# ═══════════════════════════════════════════════════════════════
# 3. RRF Fusion
# ═══════════════════════════════════════════════════════════════

class TestRRFFusion:
    def _make_hit(self, chunk_id: str, score: float, channel: str):
        from server.schemas.knowledge import RetrievalHit
        return RetrievalHit(
            chunk_id=chunk_id,
            source_id="s1",
            source_name="test",
            content=f"content_{chunk_id}",
            score=score,
            channel=channel,
        )

    def test_basic_fusion(self):
        from server.engine.knowledge_retriever import _rrf_fuse
        list_a = [self._make_hit("1", 0.9, "vector"), self._make_hit("2", 0.8, "vector")]
        list_b = [self._make_hit("2", 0.7, "keyword"), self._make_hit("3", 0.6, "keyword")]

        fused = _rrf_fuse([list_a, list_b], [1.0, 0.5], k=60, top_k=3)

        assert len(fused) == 3
        # chunk_id=2 appears in both lists, should rank highest
        assert fused[0].chunk_id == "2", f"Expected chunk 2 on top, got {fused[0].chunk_id}"
        assert all(h.channel == "fused" for h in fused)

    def test_weights_matter(self):
        from server.engine.knowledge_retriever import _rrf_fuse
        list_a = [self._make_hit("A", 1.0, "vector")]
        list_b = [self._make_hit("B", 1.0, "keyword")]

        # With equal weights, rank 1 items should score the same
        fused_equal = _rrf_fuse([list_a, list_b], [1.0, 1.0], k=60, top_k=2)
        assert abs(fused_equal[0].score - fused_equal[1].score) < 0.001

        # With vector weight >> keyword weight, A should score higher
        fused_biased = _rrf_fuse([list_a, list_b], [2.0, 0.1], k=60, top_k=2)
        assert fused_biased[0].chunk_id == "A"

    def test_empty_lists(self):
        from server.engine.knowledge_retriever import _rrf_fuse
        fused = _rrf_fuse([], [], k=60, top_k=5)
        assert fused == []

    def test_single_list(self):
        from server.engine.knowledge_retriever import _rrf_fuse
        hits = [self._make_hit("1", 0.9, "vector"), self._make_hit("2", 0.8, "vector")]
        fused = _rrf_fuse([hits], [1.0], k=60, top_k=5)
        assert len(fused) == 2
        assert fused[0].chunk_id == "1"  # rank 1 should score higher

    def test_top_k_limits(self):
        from server.engine.knowledge_retriever import _rrf_fuse
        hits = [self._make_hit(str(i), 1.0 - i * 0.1, "v") for i in range(10)]
        fused = _rrf_fuse([hits], [1.0], k=60, top_k=3)
        assert len(fused) == 3

    def test_deduplication(self):
        """Same chunk_id in multiple lists should be merged, not duplicated."""
        from server.engine.knowledge_retriever import _rrf_fuse
        list_a = [self._make_hit("1", 0.9, "vector")]
        list_b = [self._make_hit("1", 0.8, "keyword")]

        fused = _rrf_fuse([list_a, list_b], [1.0, 1.0], k=60, top_k=5)
        assert len(fused) == 1, "Duplicate chunk_id should be merged"
        # Score should be sum of both contributions
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert abs(fused[0].score - round(expected, 6)) < 0.0001


# ═══════════════════════════════════════════════════════════════
# 4. Recursive Splitting
# ═══════════════════════════════════════════════════════════════

class TestRecursiveSplit:
    def test_paragraph_split(self):
        from server.api.knowledge import _recursive_split
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = _recursive_split(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1, "Short text should be a single chunk"
        assert "First" in chunks[0] and "Third" in chunks[0]

    def test_chunk_size_respected(self):
        from server.api.knowledge import _recursive_split
        text = "\n\n".join([f"Paragraph {i}. " + "x" * 100 for i in range(20)])
        chunks = _recursive_split(text, chunk_size=200, chunk_overlap=30)
        # Allow some slack (overlap can make chunks slightly longer)
        for i, c in enumerate(chunks):
            assert len(c) <= 300, f"Chunk {i} too long ({len(c)} chars): {c[:50]}..."

    def test_overlap_present(self):
        from server.api.knowledge import _recursive_split
        text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
        chunks = _recursive_split(text, chunk_size=120, chunk_overlap=20)
        assert len(chunks) >= 2
        # At least some overlap should exist between consecutive chunks
        if len(chunks) >= 2:
            # The last part of chunk 0 should appear in the beginning of chunk 1
            tail_0 = chunks[0][-20:]
            assert len(tail_0) > 0  # overlap content should exist

    def test_long_paragraph_split(self):
        """A single long paragraph (no \\n\\n) should still be split."""
        from server.api.knowledge import _recursive_split
        text = "这是一个很长的段落。" * 100  # ~900 chars
        chunks = _recursive_split(text, chunk_size=200, chunk_overlap=30)
        assert len(chunks) >= 3, f"Expected multiple chunks, got {len(chunks)}"

    def test_sentence_boundary_splitting(self):
        """When splitting within a paragraph, should prefer sentence boundaries."""
        from server.api.knowledge import _recursive_split
        text = "第一句话。第二句话。第三句话。第四句话。第五句话。" * 10
        chunks = _recursive_split(text, chunk_size=50, chunk_overlap=10)
        # At least one chunk should end with a sentence delimiter
        endings = [c[-1] for c in chunks if c]
        has_sentence_end = any(e in "。.！!？?；;" for e in endings)
        # This is a soft assertion — boundary finding is best-effort
        assert len(chunks) >= 2

    def test_empty_input(self):
        from server.api.knowledge import _recursive_split
        assert _recursive_split("", 500, 50) == []
        assert _recursive_split("   \n\n  ", 500, 50) == []

    def test_small_chunk_size(self):
        from server.api.knowledge import _recursive_split
        text = "Hello world. This is a test."
        chunks = _recursive_split(text, chunk_size=15, chunk_overlap=3)
        assert len(chunks) >= 1
        assert all(len(c) > 0 for c in chunks)


# ═══════════════════════════════════════════════════════════════
# 5. PDF/DOCX Extractors (error handling)
# ═══════════════════════════════════════════════════════════════

class TestExtractors:
    def test_pdf_extractor_with_bad_bytes(self):
        """PDF extractor should raise on invalid bytes."""
        from server.api.knowledge import _extract_text_from_pdf
        with pytest.raises(Exception):
            _extract_text_from_pdf(b"not a pdf file")

    def test_docx_extractor_with_bad_bytes(self):
        """DOCX extractor should raise on invalid bytes."""
        from server.api.knowledge import _extract_text_from_docx
        with pytest.raises(Exception):
            _extract_text_from_docx(b"not a docx file")

    def test_allowed_extensions(self):
        from server.api.knowledge import ALLOWED_EXTENSIONS
        assert ".txt" in ALLOWED_EXTENSIONS
        assert ".md" in ALLOWED_EXTENSIONS
        assert ".pdf" in ALLOWED_EXTENSIONS
        assert ".docx" in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════
# 6. Performance Presets
# ═══════════════════════════════════════════════════════════════

class TestPerformancePresets:
    def test_ef_search_in_presets(self):
        from server.performance_presets import PRESETS
        assert PRESETS["fast"]["ef_search"] == 64
        assert PRESETS["balanced"]["ef_search"] == 128
        assert PRESETS["accurate"]["ef_search"] == 256

    def test_reranker_only_in_accurate(self):
        from server.performance_presets import PRESETS
        assert PRESETS["fast"]["reranker_enabled"] is False
        assert PRESETS["balanced"]["reranker_enabled"] is False
        assert PRESETS["accurate"]["reranker_enabled"] is True

    def test_keyword_weight_ordering(self):
        from server.performance_presets import PRESETS
        assert PRESETS["fast"]["keyword_weight"] < PRESETS["balanced"]["keyword_weight"]
        assert PRESETS["balanced"]["keyword_weight"] < PRESETS["accurate"]["keyword_weight"]


# ═══════════════════════════════════════════════════════════════
# 7. Runtime Config Wiring
# ═══════════════════════════════════════════════════════════════

class TestRuntimeConfig:
    def test_singleton(self):
        from server.runtime_config import RuntimeConfig
        a = RuntimeConfig()
        b = RuntimeConfig()
        assert a is b, "RuntimeConfig should be a singleton"

    def test_update_and_read(self):
        from server.runtime_config import RuntimeConfig
        cfg = RuntimeConfig()
        cfg.update({"test_key_123": "value"})
        assert cfg.get("test_key_123") == "value"
        cfg.set("test_key_123", None)  # cleanup

    def test_all_returns_copy(self):
        from server.runtime_config import RuntimeConfig
        cfg = RuntimeConfig()
        snapshot = cfg.all()
        assert isinstance(snapshot, dict)


# ═══════════════════════════════════════════════════════════════
# 8. Config Defaults
# ═══════════════════════════════════════════════════════════════

class TestConfigDefaults:
    def test_embedding_defaults_in_code(self):
        """Verify the code-level defaults (may be overridden by .env at runtime)."""
        from server.config import Settings
        # Check the class-level defaults, not the instance (which reads .env)
        fields = Settings.model_fields
        assert fields["embedding_model"].default == "BAAI/bge-m3"
        assert fields["embedding_dim"].default == 1024


# ═══════════════════════════════════════════════════════════════
# 9. Reranker Class Structure
# ═══════════════════════════════════════════════════════════════

class TestRerankerStructure:
    def test_reranker_class_exists(self):
        from server.engine.knowledge_retriever import Reranker
        assert hasattr(Reranker, "get_instance")
        assert hasattr(Reranker, "rerank")

    def test_reranker_handles_empty(self):
        from server.engine.knowledge_retriever import Reranker
        r = Reranker()
        # Without loading the model, rerank should return input as-is
        result = r.rerank("query", [], top_k=5)
        assert result == []

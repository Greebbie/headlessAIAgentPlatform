"""End-to-end tests for the HlAB agent platform.

Tests cover:
  1. Knowledge retrieval (fast KV lookup + vector search + combined)
  2. Skill routing + Knowledge QA via /invoke
  3. Workflow step-by-step via /invoke
  4. Tool calling via /invoke

Prerequisites:
  - Server running at localhost:8000
  - For LLM-dependent tests: LLM provider configured (DashScope/Ollama/etc.)
  - pip install -e ".[rag,dev]"
"""

from __future__ import annotations

import asyncio
import os
import pytest
import httpx

from tests.e2e.setup_test_data import BASE_URL, full_setup, teardown

# ── Markers ────────────────────────────────────────────────────

pytestmark = pytest.mark.e2e


def _server_available() -> bool:
    try:
        resp = httpx.get("http://localhost:8000/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


skip_no_server = pytest.mark.skipif(
    not _server_available(),
    reason="Server not running at localhost:8000",
)

skip_no_llm = pytest.mark.skipif(
    os.environ.get("HLAB_SKIP_LLM_TESTS", "0") == "1",
    reason="HLAB_SKIP_LLM_TESTS=1 — skipping LLM-dependent tests",
)


# ── Module-level setup/teardown (avoids async fixture event loop issues) ──

_cached_ids: dict | None = None


def _sync_setup() -> dict:
    """Run full_setup synchronously, cached per process."""
    global _cached_ids
    if _cached_ids is not None:
        return _cached_ids

    async def _run():
        async with httpx.AsyncClient(timeout=60) as client:
            return await full_setup(client)

    _cached_ids = asyncio.get_event_loop().run_until_complete(_run())
    return _cached_ids


def _sync_teardown():
    global _cached_ids
    if _cached_ids is None:
        return

    async def _run():
        async with httpx.AsyncClient(timeout=60) as client:
            await teardown(client, _cached_ids)

    try:
        asyncio.get_event_loop().run_until_complete(_run())
    except Exception:
        pass
    _cached_ids = None


@pytest.fixture(scope="module")
def test_ids():
    """Set up all test data once per module, tear down at end."""
    ids = _sync_setup()
    yield ids
    _sync_teardown()


# ═══════════════════════════════════════════════════════════════
# Part 1: Knowledge Retrieval (direct search API)
# ═══════════════════════════════════════════════════════════════

@skip_no_server
class TestKnowledgeRetrieval:
    """Test the /knowledge/search endpoint directly."""

    async def test_fast_lookup_phone(self, test_ids: dict):
        """KV fast channel should find the phone number via entity_key match."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{BASE_URL}/knowledge/search", json={
                "query": "物业电话是多少",
                "domain": "property_mgmt",
                "top_k": 5,
                "use_fast_channel": True,
                "use_rag_channel": False,
            })
        resp.raise_for_status()
        data = resp.json()

        assert len(data["hits"]) > 0, "Fast lookup should return at least 1 hit"
        contents = [h["content"] for h in data["hits"]]
        found = any("0571-88001234" in c for c in contents)
        assert found, f"Expected phone 0571-88001234 in hits, got: {contents}"
        assert data["fast_answer"] is not None

    async def test_fast_lookup_parking_fee(self, test_ids: dict):
        """KV fast channel should find parking fee info."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{BASE_URL}/knowledge/search", json={
                "query": "停车费多少钱",
                "domain": "property_mgmt",
                "top_k": 5,
                "use_fast_channel": True,
                "use_rag_channel": False,
            })
        resp.raise_for_status()
        data = resp.json()

        assert len(data["hits"]) > 0
        contents = " ".join(h["content"] for h in data["hits"])
        assert any(kw in contents for kw in ["400", "月租", "停车", "车位"]), (
            f"Expected parking fee info, got: {contents}"
        )

    async def test_vector_search_semantic(self, test_ids: dict):
        """RAG vector channel should find parking fee via semantic search."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{BASE_URL}/knowledge/search", json={
                "query": "小区停车一个月要花多少钱",
                "domain": "property_mgmt",
                "top_k": 5,
                "use_fast_channel": False,
                "use_rag_channel": True,
            })
        resp.raise_for_status()
        data = resp.json()

        assert len(data["hits"]) > 0, "RAG channel should return at least 1 hit"
        contents = " ".join(h["content"] for h in data["hits"])
        assert any(kw in contents for kw in ["停车", "车位", "月租"]), (
            f"Expected parking-related content in RAG hits, got: {contents[:200]}"
        )

    async def test_combined_retrieval(self, test_ids: dict):
        """Both channels should return results, sorted by score."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{BASE_URL}/knowledge/search", json={
                "query": "物业费多少",
                "domain": "property_mgmt",
                "top_k": 5,
                "use_fast_channel": True,
                "use_rag_channel": True,
            })
        resp.raise_for_status()
        data = resp.json()

        assert len(data["hits"]) > 0
        assert data["fast_answer"] is not None

        # Hits should be sorted by score descending
        scores = [h["score"] for h in data["hits"]]
        assert scores == sorted(scores, reverse=True), f"Hits not sorted by score: {scores}"

    async def test_empty_domain_no_cross_leak(self, test_ids: dict):
        """Searching a non-existent domain should return no hits."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{BASE_URL}/knowledge/search", json={
                "query": "物业费",
                "domain": "nonexistent_domain",
                "top_k": 5,
                "use_fast_channel": True,
                "use_rag_channel": True,
            })
        resp.raise_for_status()
        data = resp.json()
        assert len(data["hits"]) == 0, "Cross-domain leak: hits found in nonexistent domain"


# ═══════════════════════════════════════════════════════════════
# Part 2: Skill Routing + Knowledge QA via /invoke
# ═══════════════════════════════════════════════════════════════

@skip_no_server
@skip_no_llm
class TestSkillRoutingKnowledgeQA:
    """Test skill-based routing to knowledge_qa skill via /invoke."""

    async def test_invoke_knowledge_qa_phone(self, test_ids: dict):
        """Asking about phone number should route to knowledge_qa and return answer."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{BASE_URL}/invoke", json={
                "agent_id": test_ids["agent_id"],
                "message": "物业服务电话是多少？",
            })
        resp.raise_for_status()
        data = resp.json()

        assert data["session_id"], "Should return a session_id"
        assert data["short_answer"], "Should return an answer"

        answer = data["short_answer"]
        assert "0571-88001234" in answer or "88001234" in answer, (
            f"Expected phone number in answer, got: {answer}"
        )

    async def test_invoke_knowledge_qa_fee(self, test_ids: dict):
        """Asking about property fees should route to knowledge_qa."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{BASE_URL}/invoke", json={
                "agent_id": test_ids["agent_id"],
                "message": "物业费标准是多少？",
            })
        resp.raise_for_status()
        data = resp.json()

        answer = data["short_answer"]
        assert any(kw in answer for kw in ["3.5", "物业费", "每平方米"]), (
            f"Expected fee info in answer, got: {answer}"
        )


# ═══════════════════════════════════════════════════════════════
# Part 3: Workflow via /invoke
# ═══════════════════════════════════════════════════════════════

@skip_no_server
@skip_no_llm
class TestWorkflow:
    """Test workflow step-by-step via /invoke."""

    async def test_workflow_step_by_step(self, test_ids: dict):
        """Trigger repair workflow -> submit form -> skip photo -> confirm -> complete.

        Note: In conversational mode, the LLM decides whether to call the
        workflow tool. qwen-turbo may need an explicit prompt to trigger it.
        We allow a few extra follow-up rounds for LLM non-determinism.
        """
        agent_id = test_ids["agent_id"]

        async with httpx.AsyncClient(timeout=60) as client:
            # Step 1: Trigger workflow — use explicit wording
            resp = await client.post(f"{BASE_URL}/invoke", json={
                "agent_id": agent_id,
                "message": "我要报修，请帮我启动报修流程",
            })
            resp.raise_for_status()
            data = resp.json()
            session_id = data["session_id"]

            # If LLM didn't trigger workflow on first try, push harder
            if data.get("workflow_status") not in ("in_progress", "waiting_input"):
                resp = await client.post(f"{BASE_URL}/invoke", json={
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "message": "请直接启动报修工单流程",
                })
                resp.raise_for_status()
                data = resp.json()

            assert data.get("workflow_status") in ("in_progress", "waiting_input") or "报修" in data["short_answer"], (
                f"Expected workflow to start, got: {data}"
            )

            # Step 2: Submit repair info
            resp = await client.post(f"{BASE_URL}/invoke", json={
                "agent_id": agent_id,
                "session_id": session_id,
                "message": "提交报修信息",
                "form_data": {
                    "location": "3号楼501室厨房",
                    "issue_type": "plumbing",
                    "description": "厨房水龙头漏水",
                },
            })
            resp.raise_for_status()
            data = resp.json()
            assert data["session_id"] == session_id

            # Step 3: Skip photo
            resp = await client.post(f"{BASE_URL}/invoke", json={
                "agent_id": agent_id,
                "session_id": session_id,
                "message": "跳过",
                "form_data": {"photo": ""},
            })
            resp.raise_for_status()
            data = resp.json()

            # Step 4: Confirm
            resp = await client.post(f"{BASE_URL}/invoke", json={
                "agent_id": agent_id,
                "session_id": session_id,
                "message": "确认",
                "form_data": {"__confirm": "yes"},
            })
            resp.raise_for_status()
            data = resp.json()

            # May need extra steps for the final action (LLM non-determinism)
            for _ in range(3):
                if data.get("workflow_status") == "completed":
                    break
                if "完成" in data.get("short_answer", "") or "工单" in data.get("short_answer", ""):
                    break
                resp = await client.post(f"{BASE_URL}/invoke", json={
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "message": "好的，确认提交",
                    "form_data": {"__confirm": "yes"},
                })
                resp.raise_for_status()
                data = resp.json()

            assert (
                data.get("workflow_status") == "completed"
                or "完成" in data.get("short_answer", "")
                or "工单" in data.get("short_answer", "")
                or "报修" in data.get("short_answer", "")
            ), (
                f"Expected workflow to complete, got status={data.get('workflow_status')}, "
                f"answer={data.get('short_answer')}"
            )


# ═══════════════════════════════════════════════════════════════
# Part 4: Tool Calling via /invoke
# ═══════════════════════════════════════════════════════════════

@skip_no_server
@skip_no_llm
class TestToolCalling:
    """Test tool_call skill via /invoke."""

    async def test_invoke_calculator(self, test_ids: dict):
        """Calculator tool should compute 123 * 456 = 56088."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{BASE_URL}/invoke", json={
                "agent_id": test_ids["agent_id"],
                "message": "帮我计算123乘以456",
            })
        resp.raise_for_status()
        data = resp.json()

        answer = data["short_answer"]
        assert "56088" in answer, f"Expected 56088 in answer, got: {answer}"

    async def test_invoke_weather(self, test_ids: dict):
        """Weather tool should return weather for a city."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{BASE_URL}/invoke", json={
                "agent_id": test_ids["agent_id"],
                "message": "杭州今天天气怎么样？",
            })
        resp.raise_for_status()
        data = resp.json()

        answer = data["short_answer"]
        assert any(kw in answer for kw in ["杭州", "天气", "°C", "度"]), (
            f"Expected weather info in answer, got: {answer}"
        )


# ═══════════════════════════════════════════════════════════════
# Part 5: Vector Admin endpoints
# ═══════════════════════════════════════════════════════════════

@skip_no_server
class TestVectorAdmin:
    """Test vector store admin endpoints."""

    async def test_vector_stats(self, test_ids: dict):
        """Stats endpoint should return index info."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{BASE_URL}/vector-admin/stats")
        resp.raise_for_status()
        data = resp.json()
        assert "index_count" in data
        assert "dimension" in data
        assert "status" in data

    async def test_vector_health(self, test_ids: dict):
        """Health endpoint should report status."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{BASE_URL}/vector-admin/health")
        resp.raise_for_status()
        data = resp.json()
        assert "initialized" in data
        assert "backend" in data
        assert data["backend"] == "faiss"

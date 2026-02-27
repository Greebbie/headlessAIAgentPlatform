"""E2E test data setup — creates knowledge, workflow, tools, skills, and agent via API."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "http://localhost:8000/api/v1"
TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
DOMAIN = "property_mgmt"
TENANT = "default"

# Unique suffix per test run to avoid name collisions
_RUN_ID = uuid.uuid4().hex[:8]


async def setup_knowledge(client: httpx.AsyncClient) -> dict[str, Any]:
    """Create a knowledge source, upload the property doc, and add KV entries.

    Returns {"source_id": str, "chunk_ids": list[str], "kv_ids": list[str]}.
    """
    # 1. Create knowledge source
    resp = await client.post(f"{BASE_URL}/knowledge/sources", json={
        "name": "物业管理知识库",
        "source_type": "document",
        "domain": DOMAIN,
        "tenant_id": TENANT,
    })
    resp.raise_for_status()
    source_id = resp.json()["id"]

    # 2. Upload document
    doc_path = TEST_DATA_DIR / "property_management.txt"
    with open(doc_path, "rb") as f:
        resp = await client.post(
            f"{BASE_URL}/knowledge/upload",
            files={"file": ("property_management.txt", f, "text/plain")},
            data={"source_id": source_id, "domain": DOMAIN, "chunk_size": "500", "chunk_overlap": "50"},
        )
    resp.raise_for_status()
    chunk_ids = resp.json()["chunk_ids"]

    # 3. Add KV entities for fast-lookup channel
    kv_entries = [
        {"entity_key": "物业电话", "content": "物业服务中心24小时值班电话：0571-88001234"},
        {"entity_key": "应急电话", "content": "紧急维修应急电话：0571-88005678"},
        {"entity_key": "物业费标准", "content": "住宅物业费每月每平方米3.5元，商铺每月每平方米6.8元，按季度缴纳"},
        {"entity_key": "停车费", "content": "地下车位月租400元，地面车位月租200元，临时停车前2小时免费，超时每小时5元"},
    ]
    kv_ids = []
    for kv in kv_entries:
        resp = await client.post(f"{BASE_URL}/knowledge/kv", json={
            "source_id": source_id,
            "entity_key": kv["entity_key"],
            "content": kv["content"],
            "domain": DOMAIN,
        })
        resp.raise_for_status()
        kv_ids.append(resp.json()["id"])

    return {"source_id": source_id, "chunk_ids": chunk_ids, "kv_ids": kv_ids}


async def setup_workflow(client: httpx.AsyncClient) -> dict[str, Any]:
    """Create a 4-step repair request workflow.

    Returns {"workflow_id": str}.
    """
    resp = await client.post(f"{BASE_URL}/workflows/", json={
        "name": "报修工单流程",
        "description": "业主报修流程：收集信息 → 上传照片(可选) → 确认 → 完成",
        "tenant_id": TENANT,
        "steps": [
            {
                "name": "收集报修信息",
                "order": 0,
                "step_type": "collect",
                "prompt_template": "请描述您的报修问题，包括：位置、问题类型、严重程度。",
                "fields": [
                    {"name": "location", "label": "位置", "field_type": "text", "required": True, "placeholder": "如：3号楼501室厨房"},
                    {"name": "issue_type", "label": "问题类型", "field_type": "select", "required": True, "options": [
                        {"value": "plumbing", "label": "水管/漏水"},
                        {"value": "electrical", "label": "电力/照明"},
                        {"value": "door_lock", "label": "门锁/门窗"},
                        {"value": "other", "label": "其他"},
                    ]},
                    {"name": "description", "label": "问题描述", "field_type": "text", "required": True, "placeholder": "请详细描述问题"},
                ],
            },
            {
                "name": "上传照片",
                "order": 1,
                "step_type": "collect",
                "prompt_template": "如有照片请上传，也可以输入'跳过'继续。",
                "fields": [
                    {"name": "photo", "label": "现场照片", "field_type": "text", "required": False, "placeholder": "输入跳过或上传照片"},
                ],
            },
            {
                "name": "确认信息",
                "order": 2,
                "step_type": "confirm",
                "prompt_template": "请确认以上报修信息是否正确？",
                "requires_human_confirm": True,
            },
            {
                "name": "提交完成",
                "order": 3,
                "step_type": "complete",
                "prompt_template": "报修工单已提交，工单编号：WO-{timestamp}。我们将在24小时内安排维修人员上门。",
            },
        ],
    })
    resp.raise_for_status()
    return {"workflow_id": resp.json()["id"]}


async def setup_tools(client: httpx.AsyncClient) -> dict[str, Any]:
    """Register calculator and weather mock tools.

    Returns {"tool_ids": [calculator_id, weather_id]}.
    """
    tools_data = [
        {
            "name": f"calculator_{_RUN_ID}",
            "description": "简单数学计算器，支持加减乘除。",
            "category": "function",
            "endpoint": "http://localhost:8000/api/v1/mock-tools/calculator",
            "method": "POST",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 123*456"},
                },
                "required": ["expression"],
            },
            "tenant_id": TENANT,
        },
        {
            "name": f"weather_query_{_RUN_ID}",
            "description": "查询城市天气信息。",
            "category": "function",
            "endpoint": "http://localhost:8000/api/v1/mock-tools/weather",
            "method": "POST",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 杭州"},
                },
                "required": ["city"],
            },
            "tenant_id": TENANT,
        },
    ]

    tool_ids = []
    for td in tools_data:
        resp = await client.post(f"{BASE_URL}/tools/", json=td)
        resp.raise_for_status()
        tool_ids.append(resp.json()["id"])

    return {"tool_ids": tool_ids}


async def setup_skills(
    client: httpx.AsyncClient,
    workflow_id: str,
    tool_ids: list[str],
) -> dict[str, Any]:
    """Create 3 skills: knowledge_qa, workflow, tool_call.

    Returns {"skill_ids": [knowledge_qa_id, workflow_id, tool_call_id]}.
    """
    skills_data = [
        {
            "name": "物业知识问答",
            "description": "回答物业相关问题（电话、费用、规定等）",
            "skill_type": "knowledge_qa",
            "trigger_config": {
                "keywords": ["电话", "费用", "物业费", "停车费", "停车", "垃圾", "装修", "门禁", "安防", "收费"],
                "trigger_description": "用户询问物业相关的知识问题",
            },
            "execution_config": {
                "domain": DOMAIN,
            },
            "priority": 10,
            "tenant_id": TENANT,
        },
        {
            "name": "报修工单",
            "description": "处理业主报修请求的工作流",
            "skill_type": "workflow",
            "trigger_config": {
                "keywords": ["报修", "维修", "漏水", "坏了", "修理", "故障"],
                "trigger_description": "用户要报修或维修",
            },
            "execution_config": {
                "workflow_id": workflow_id,
            },
            "priority": 20,
            "tenant_id": TENANT,
        },
        {
            "name": "工具调用",
            "description": "调用计算器或天气查询工具",
            "skill_type": "tool_call",
            "trigger_config": {
                "keywords": ["计算", "算一下", "天气", "几度", "乘", "加", "减", "除"],
                "trigger_description": "用户需要计算或查询天气",
            },
            "execution_config": {
                "tool_ids": tool_ids,
                "max_tool_rounds": 3,
            },
            "priority": 30,
            "tenant_id": TENANT,
        },
    ]

    skill_ids = []
    for sd in skills_data:
        resp = await client.post(f"{BASE_URL}/skills/", json=sd)
        resp.raise_for_status()
        skill_ids.append(resp.json()["id"])

    return {"skill_ids": skill_ids}


async def setup_agent(
    client: httpx.AsyncClient,
    skill_ids: list[str],
) -> dict[str, Any]:
    """Create an agent and bind all skills to it.

    Returns {"agent_id": str, "binding_ids": list[str]}.
    """
    # Create agent
    resp = await client.post(f"{BASE_URL}/agents/", json={
        "name": "E2E测试物业助手",
        "description": "端到端测试用物业服务智能助手",
        "system_prompt": "你是一个专业的物业服务助手。请用简洁的中文回答业主的问题。",
        "skill_routing_mode": "skill_based",
        "tenant_id": TENANT,
    })
    resp.raise_for_status()
    agent_id = resp.json()["id"]

    # Bind skills
    binding_ids = []
    for sid in skill_ids:
        resp = await client.post(f"{BASE_URL}/agents/{agent_id}/skills", json={
            "skill_id": sid,
        })
        resp.raise_for_status()
        binding_ids.append(resp.json()["id"])

    return {"agent_id": agent_id, "binding_ids": binding_ids}


async def full_setup(client: httpx.AsyncClient) -> dict[str, Any]:
    """Run all setup steps and return all created IDs."""
    knowledge = await setup_knowledge(client)
    workflow = await setup_workflow(client)
    tools = await setup_tools(client)
    skills = await setup_skills(client, workflow["workflow_id"], tools["tool_ids"])
    agent = await setup_agent(client, skills["skill_ids"])

    return {
        **knowledge,
        **workflow,
        **tools,
        **skills,
        **agent,
    }


async def teardown(client: httpx.AsyncClient, ids: dict[str, Any]) -> None:
    """Clean up all test data (best-effort, ignores 404s)."""
    agent_id = ids.get("agent_id")
    if agent_id:
        # Unbind skills
        for bid in ids.get("binding_ids", []):
            await client.delete(f"{BASE_URL}/agents/{agent_id}/skills/{bid}")
        # Delete agent
        await client.delete(f"{BASE_URL}/agents/{agent_id}")

    # Delete skills
    for sid in ids.get("skill_ids", []):
        await client.delete(f"{BASE_URL}/skills/{sid}")

    # Delete tools
    for tid in ids.get("tool_ids", []):
        await client.delete(f"{BASE_URL}/tools/{tid}")

    # Delete workflow
    wid = ids.get("workflow_id")
    if wid:
        await client.delete(f"{BASE_URL}/workflows/{wid}")

    # Delete knowledge source (cascades to chunks)
    src_id = ids.get("source_id")
    if src_id:
        await client.delete(f"{BASE_URL}/knowledge/sources/{src_id}")

"""Agent management CRUD API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import get_db
from server.models.agent import Agent
from server.schemas.agent import AgentCreate, AgentUpdate, AgentOut

router = APIRouter()


@router.get("/", response_model=list[AgentOut])
async def list_agents(tenant_id: str = "default", db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.tenant_id == tenant_id))
    return result.scalars().all()


@router.post("/", response_model=AgentOut, status_code=201)
async def create_agent(body: AgentCreate, db: AsyncSession = Depends(get_db)):
    agent = Agent(
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        llm_model=body.llm_model,
        workflow_id=body.workflow_id,
        workflow_scope=body.workflow_scope,
        knowledge_scope=body.knowledge_scope,
        tool_scope=body.tool_scope,
        response_config=body.response_config,
        risk_config=body.risk_config,
        tenant_id=body.tenant_id,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return agent


@router.get("/{agent_id}", response_model=AgentOut)
async def get_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, "Agent not found")
    return agent


@router.put("/{agent_id}", response_model=AgentOut)
async def update_agent(agent_id: str, body: AgentUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, "Agent not found")

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(agent, key, value)

    agent.version += 1
    await db.commit()
    await db.refresh(agent)
    return agent


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, "Agent not found")
    await db.delete(agent)
    await db.commit()

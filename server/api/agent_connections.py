"""Agent connection API — manage inter-agent collaboration relationships."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import get_db
from server.models.agent_connection import AgentConnection
from server.schemas.skill import AgentConnectionCreate, AgentConnectionUpdate, AgentConnectionOut

router = APIRouter()

VALID_CONNECTION_TYPES = {"delegate", "orchestrate", "peer"}


@router.get("/", response_model=list[AgentConnectionOut])
async def list_connections(
    tenant_id: str = "default",
    agent_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List connections, optionally filtered by agent (as source or target)."""
    query = select(AgentConnection).where(AgentConnection.tenant_id == tenant_id)
    if agent_id:
        query = query.where(
            or_(
                AgentConnection.source_agent_id == agent_id,
                AgentConnection.target_agent_id == agent_id,
            )
        )
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/", response_model=AgentConnectionOut, status_code=201)
async def create_connection(body: AgentConnectionCreate, db: AsyncSession = Depends(get_db)):
    if body.connection_type not in VALID_CONNECTION_TYPES:
        raise HTTPException(400, f"Invalid connection_type. Must be one of: {VALID_CONNECTION_TYPES}")

    if body.source_agent_id == body.target_agent_id:
        raise HTTPException(400, "Cannot connect an agent to itself")

    conn = AgentConnection(
        source_agent_id=body.source_agent_id,
        target_agent_id=body.target_agent_id,
        connection_type=body.connection_type,
        shared_context=body.shared_context,
        description=body.description,
        enabled=body.enabled,
        tenant_id=body.tenant_id,
    )
    db.add(conn)
    await db.commit()
    await db.refresh(conn)
    return conn


@router.get("/{connection_id}", response_model=AgentConnectionOut)
async def get_connection(connection_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AgentConnection).where(AgentConnection.id == connection_id))
    conn = result.scalar_one_or_none()
    if not conn:
        raise HTTPException(404, "Connection not found")
    return conn


@router.put("/{connection_id}", response_model=AgentConnectionOut)
async def update_connection(connection_id: str, body: AgentConnectionUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AgentConnection).where(AgentConnection.id == connection_id))
    conn = result.scalar_one_or_none()
    if not conn:
        raise HTTPException(404, "Connection not found")

    update_data = body.model_dump(exclude_unset=True)
    if "connection_type" in update_data and update_data["connection_type"] not in VALID_CONNECTION_TYPES:
        raise HTTPException(400, f"Invalid connection_type. Must be one of: {VALID_CONNECTION_TYPES}")

    for key, value in update_data.items():
        setattr(conn, key, value)

    await db.commit()
    await db.refresh(conn)
    return conn


@router.delete("/{connection_id}", status_code=204)
async def delete_connection(connection_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AgentConnection).where(AgentConnection.id == connection_id))
    conn = result.scalar_one_or_none()
    if not conn:
        raise HTTPException(404, "Connection not found")
    await db.delete(conn)
    await db.commit()

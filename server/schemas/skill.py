"""Schemas for Skill, AgentSkill, and AgentConnection CRUD."""

from __future__ import annotations

from typing import Any
from datetime import datetime

from pydantic import BaseModel


# ── Skill schemas ───────────────────────────────────

class SkillCreate(BaseModel):
    name: str
    description: str = ""
    skill_type: str  # workflow | tool_call | knowledge_qa | delegate | composite
    trigger_config: dict[str, Any] | None = None
    execution_config: dict[str, Any] | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    priority: int = 100
    enabled: bool = True
    tenant_id: str = "default"


class SkillUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    skill_type: str | None = None
    trigger_config: dict[str, Any] | None = None
    execution_config: dict[str, Any] | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    priority: int | None = None
    enabled: bool | None = None


class SkillOut(BaseModel):
    id: str
    name: str
    description: str
    skill_type: str
    trigger_config: Any
    execution_config: Any
    input_schema: Any
    output_schema: Any
    priority: int
    enabled: bool
    tenant_id: str
    managed_by: str | None = None
    version: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── AgentSkill schemas ──────────────────────────────

class AgentSkillCreate(BaseModel):
    skill_id: str
    priority_override: int | None = None
    config_override: dict[str, Any] | None = None
    enabled: bool = True


class AgentSkillOut(BaseModel):
    id: str
    agent_id: str
    skill_id: str
    priority_override: int | None
    config_override: Any
    enabled: bool
    created_at: datetime
    # Joined skill info (populated in API layer)
    skill_name: str | None = None
    skill_type: str | None = None
    skill_description: str | None = None

    model_config = {"from_attributes": True}


# ── AgentConnection schemas ────────────────────────

class AgentConnectionCreate(BaseModel):
    source_agent_id: str
    target_agent_id: str
    connection_type: str = "delegate"  # delegate | orchestrate | peer
    shared_context: dict[str, Any] | None = None
    description: str = ""
    enabled: bool = True
    tenant_id: str = "default"


class AgentConnectionUpdate(BaseModel):
    connection_type: str | None = None
    shared_context: dict[str, Any] | None = None
    description: str | None = None
    enabled: bool | None = None


class AgentConnectionOut(BaseModel):
    id: str
    source_agent_id: str
    target_agent_id: str
    connection_type: str
    shared_context: Any
    description: str
    enabled: bool
    tenant_id: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

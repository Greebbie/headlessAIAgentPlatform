"""Schemas for Agent CRUD."""

from __future__ import annotations

from typing import Any
from datetime import datetime

from pydantic import BaseModel


class AgentCreate(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    llm_model: str | None = None
    llm_config_id: str | None = None
    response_config: dict[str, Any] | None = None
    risk_config: dict[str, Any] | None = None
    enabled: bool = True
    tenant_id: str = "default"


class AgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    llm_model: str | None = None
    llm_config_id: str | None = None
    response_config: dict[str, Any] | None = None
    risk_config: dict[str, Any] | None = None
    enabled: bool | None = None


class AgentOut(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    llm_model: str | None
    llm_config_id: str | None
    response_config: Any
    risk_config: Any
    tenant_id: str
    enabled: bool
    version: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

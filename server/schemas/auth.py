"""Schemas for authentication endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ── Request schemas ─────────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=128)
    password: str = Field(..., min_length=8, max_length=256)
    role: str = Field(default="editor", pattern="^(admin|editor|viewer)$")
    tenant_id: str = Field(default="default", max_length=36)
    display_name: str = Field(default="", max_length=128)


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(default="", max_length=128)
    scopes: list[str] | None = None


# ── Response schemas ────────────────────────────────────────────


class UserInfo(BaseModel):
    id: str
    username: str
    role: str
    tenant_id: str
    display_name: str
    enabled: bool
    created_at: datetime | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserInfo


class APIKeyOut(BaseModel):
    id: str
    name: str
    tenant_id: str
    scopes: list[str] | None = None
    enabled: bool
    last_used_at: datetime | None = None
    created_at: datetime | None = None


class APIKeyCreatedResponse(BaseModel):
    """Returned only once when an API key is created. The raw key is never shown again."""

    id: str
    name: str
    key: str  # Raw key — shown only once
    tenant_id: str
    scopes: list[str] | None = None

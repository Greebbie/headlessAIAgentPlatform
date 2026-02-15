"""Schemas for the /invoke endpoint — the single Headless API entry point."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InvokeRequest(BaseModel):
    """Client sends this to talk to an agent."""

    agent_id: str
    session_id: str | None = None  # None = create new session
    user_id: str = "anonymous"
    message: str
    # Optional: pre-filled form data (for workflow steps)
    form_data: dict[str, Any] | None = None
    # Optional: explicit intent override
    intent: str | None = None
    # Whether the user wants expanded answer
    expand: bool = False
    # Client metadata (device, channel, etc.)
    client_meta: dict[str, Any] | None = None


class Citation(BaseModel):
    source_id: str
    source_name: str
    content_snippet: str
    page: int | None = None
    paragraph: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    score: float | None = None


class WorkflowCard(BaseModel):
    """A UI card for a workflow step."""
    step_name: str
    step_type: str
    prompt: str
    fields: list[dict[str, Any]] | None = None
    # Current step index & total
    current_step: int = 0
    total_steps: int = 0


class InvokeResponse(BaseModel):
    """Unified response from any agent invocation."""

    session_id: str
    trace_id: str
    # Core answer
    short_answer: str
    expanded_answer: str | None = None
    # Citations with traceability
    citations: list[Citation] = Field(default_factory=list)
    # Suggested follow-up questions
    suggested_followups: list[str] = Field(default_factory=list)
    # Workflow card (if in a workflow)
    workflow_card: WorkflowCard | None = None
    # Workflow status
    workflow_status: str | None = None  # None | in_progress | waiting_input | completed | escalated
    # Escalation info
    escalated: bool = False
    escalation_reason: str | None = None
    # Metadata
    metadata: dict[str, Any] | None = None

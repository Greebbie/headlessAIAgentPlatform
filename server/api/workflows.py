"""Workflow management CRUD API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from server.db import get_db
from server.models.workflow import Workflow, WorkflowStep
from server.schemas.workflow import WorkflowCreate, WorkflowUpdate, WorkflowOut, StepCreate, StepOut


async def _load_workflow(db: AsyncSession, workflow_id: str) -> Workflow | None:
    """Load a workflow with steps eagerly loaded (avoids async lazy-load issues)."""
    result = await db.execute(
        select(Workflow).where(Workflow.id == workflow_id).options(selectinload(Workflow.steps))
    )
    return result.scalar_one_or_none()

router = APIRouter()


@router.get("/", response_model=list[WorkflowOut])
async def list_workflows(tenant_id: str = "default", db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Workflow).where(Workflow.tenant_id == tenant_id).options(selectinload(Workflow.steps))
    )
    return result.scalars().all()


@router.post("/", response_model=WorkflowOut, status_code=201)
async def create_workflow(body: WorkflowCreate, db: AsyncSession = Depends(get_db)):
    wf = Workflow(
        name=body.name,
        description=body.description,
        tenant_id=body.tenant_id,
        config=body.config,
    )
    db.add(wf)
    await db.flush()

    # Create steps
    if body.steps:
        for step_data in body.steps:
            step = WorkflowStep(
                workflow_id=wf.id,
                name=step_data.name,
                order=step_data.order,
                step_type=step_data.step_type,
                prompt_template=step_data.prompt_template,
                fields=[f.model_dump() for f in step_data.fields] if step_data.fields else None,
                validation_rules=step_data.validation_rules,
                tool_id=step_data.tool_id,
                tool_config=step_data.tool_config,
                on_failure=step_data.on_failure,
                max_retries=step_data.max_retries,
                fallback_step_id=step_data.fallback_step_id,
                requires_human_confirm=step_data.requires_human_confirm,
                risk_level=step_data.risk_level,
                next_step_rules=step_data.next_step_rules,
            )
            db.add(step)

    await db.commit()
    wf = await _load_workflow(db, wf.id)
    return wf


@router.get("/{workflow_id}", response_model=WorkflowOut)
async def get_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    wf = await _load_workflow(db, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    return wf


@router.put("/{workflow_id}", response_model=WorkflowOut)
async def update_workflow(workflow_id: str, body: WorkflowUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(404, "Workflow not found")

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(wf, key, value)

    wf.version += 1
    await db.commit()
    wf = await _load_workflow(db, wf.id)
    return wf


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(404, "Workflow not found")
    # Delete steps first
    steps = await db.execute(select(WorkflowStep).where(WorkflowStep.workflow_id == wf.id))
    for step in steps.scalars().all():
        await db.delete(step)
    await db.delete(wf)
    await db.commit()


# ── Step management ──────────────────────────────────────────────

@router.post("/{workflow_id}/steps", response_model=StepOut, status_code=201)
async def add_step(workflow_id: str, body: StepCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(404, "Workflow not found")

    step = WorkflowStep(
        workflow_id=workflow_id,
        name=body.name,
        order=body.order,
        step_type=body.step_type,
        prompt_template=body.prompt_template,
        fields=[f.model_dump() for f in body.fields] if body.fields else None,
        validation_rules=body.validation_rules,
        tool_id=body.tool_id,
        tool_config=body.tool_config,
        on_failure=body.on_failure,
        max_retries=body.max_retries,
        fallback_step_id=body.fallback_step_id,
        requires_human_confirm=body.requires_human_confirm,
        risk_level=body.risk_level,
        next_step_rules=body.next_step_rules,
    )
    db.add(step)
    await db.commit()
    await db.refresh(step)
    return step


@router.put("/{workflow_id}/steps/{step_id}", response_model=StepOut)
async def update_step(workflow_id: str, step_id: str, body: StepCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WorkflowStep).where(WorkflowStep.id == step_id, WorkflowStep.workflow_id == workflow_id)
    )
    step = result.scalar_one_or_none()
    if not step:
        raise HTTPException(404, "Step not found")

    step.name = body.name
    step.order = body.order
    step.step_type = body.step_type
    step.prompt_template = body.prompt_template
    step.fields = [f.model_dump() for f in body.fields] if body.fields else None
    step.validation_rules = body.validation_rules
    step.tool_id = body.tool_id
    step.tool_config = body.tool_config
    step.on_failure = body.on_failure
    step.max_retries = body.max_retries
    step.fallback_step_id = body.fallback_step_id
    step.requires_human_confirm = body.requires_human_confirm
    step.risk_level = body.risk_level
    step.next_step_rules = body.next_step_rules

    await db.commit()
    await db.refresh(step)
    return step


@router.delete("/{workflow_id}/steps/{step_id}", status_code=204)
async def delete_step(workflow_id: str, step_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WorkflowStep).where(WorkflowStep.id == step_id, WorkflowStep.workflow_id == workflow_id)
    )
    step = result.scalar_one_or_none()
    if not step:
        raise HTTPException(404, "Step not found")
    await db.delete(step)
    await db.commit()

"""Workflow management CRUD API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from server.db import get_db
from server.middleware.auth import get_current_user
from server.models.workflow import Workflow, WorkflowStep
from server.schemas.workflow import WorkflowCreate, WorkflowUpdate, WorkflowOut, StepCreate, StepOut


async def _load_workflow(db: AsyncSession, workflow_id: str) -> Workflow | None:
    """Load a workflow with steps eagerly loaded (avoids async lazy-load issues)."""
    result = await db.execute(
        select(Workflow).where(Workflow.id == workflow_id).options(selectinload(Workflow.steps))
    )
    return result.scalar_one_or_none()

router = APIRouter(dependencies=[Depends(get_current_user)])


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


# ── Version management ──────────────────────────────────────────


@router.post("/{workflow_id}/publish")
async def publish_version(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Publish current workflow state as a new version snapshot."""
    from server.models.workflow import WorkflowVersion

    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Get current steps
    steps_result = await db.execute(
        select(WorkflowStep)
        .where(WorkflowStep.workflow_id == workflow_id)
        .order_by(WorkflowStep.order)
    )
    steps = list(steps_result.scalars().all())

    # Build snapshot
    snapshot = {
        "name": workflow.name,
        "description": workflow.description,
        "config": workflow.config,
        "steps": [
            {
                "name": s.name,
                "order": s.order,
                "step_type": s.step_type,
                "prompt_template": s.prompt_template,
                "fields": s.fields,
                "validation_rules": s.validation_rules,
                "tool_id": s.tool_id,
                "tool_config": s.tool_config,
                "on_failure": s.on_failure,
                "max_retries": s.max_retries,
                "next_step_rules": s.next_step_rules,
                "requires_human_confirm": s.requires_human_confirm,
                "risk_level": s.risk_level,
            }
            for s in steps
        ],
    }

    version = WorkflowVersion(
        workflow_id=workflow_id,
        version=workflow.version,
        snapshot=snapshot,
    )
    db.add(version)

    # Increment workflow version
    workflow.version += 1
    await db.commit()

    return {"workflow_id": workflow_id, "version": version.version, "id": version.id}


@router.get("/{workflow_id}/versions")
async def list_versions(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """List all published versions of a workflow."""
    from server.models.workflow import WorkflowVersion

    result = await db.execute(
        select(WorkflowVersion)
        .where(WorkflowVersion.workflow_id == workflow_id)
        .order_by(WorkflowVersion.version.desc())
    )
    versions = result.scalars().all()
    return [
        {
            "id": v.id,
            "workflow_id": v.workflow_id,
            "version": v.version,
            "published_by": v.published_by,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]


@router.get("/{workflow_id}/versions/{version_num}")
async def get_version(
    workflow_id: str, version_num: int, db: AsyncSession = Depends(get_db),
):
    """Get a specific version snapshot."""
    from server.models.workflow import WorkflowVersion

    result = await db.execute(
        select(WorkflowVersion).where(
            WorkflowVersion.workflow_id == workflow_id,
            WorkflowVersion.version == version_num,
        )
    )
    version = result.scalar_one_or_none()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    return {
        "id": version.id,
        "workflow_id": version.workflow_id,
        "version": version.version,
        "snapshot": version.snapshot,
        "published_by": version.published_by,
        "created_at": version.created_at.isoformat() if version.created_at else None,
    }

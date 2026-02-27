"""FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.config import settings

# Ensure data directory exists for SQLite
os.makedirs("./data", exist_ok=True)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from server.db import engine, Base
    import server.models  # noqa: F401 – register all models

    # Initialize default runtime config (balanced preset)
    from server.runtime_config import runtime_config
    from server.performance_presets import PRESETS
    runtime_config.update(PRESETS["balanced"])
    runtime_config.set("active_preset", "balanced")

    # Pre-initialize jieba to avoid ~1s cold start on first request
    try:
        import jieba
        jieba.initialize()
    except ImportError:
        pass

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Safe migration: add llm_config_id column for existing databases
        from sqlalchemy import text
        try:
            await conn.execute(text(
                "ALTER TABLE agents ADD COLUMN llm_config_id VARCHAR(36) REFERENCES llm_configs(id) ON DELETE SET NULL"
            ))
        except Exception:
            pass  # Column already exists
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ────────────────────────────────
from server.api.invoke import router as invoke_router
from server.api.agents import router as agents_router
from server.api.workflows import router as workflows_router
from server.api.knowledge import router as knowledge_router
from server.api.tools import router as tools_router
from server.api.audit import router as audit_router
from server.api.mock_tools import router as mock_tools_router
from server.api.llm_configs import router as llm_configs_router
from server.api.performance import router as performance_router
from server.api.vector_admin import router as vector_admin_router
from server.api.skills import router as skills_router
from server.api.agent_skills import router as agent_skills_router
from server.api.agent_connections import router as agent_connections_router
from server.api.agent_capabilities import router as agent_capabilities_router

prefix = settings.api_prefix

app.include_router(invoke_router, prefix=prefix, tags=["invoke"])
app.include_router(agents_router, prefix=prefix + "/agents", tags=["agents"])
app.include_router(workflows_router, prefix=prefix + "/workflows", tags=["workflows"])
app.include_router(knowledge_router, prefix=prefix + "/knowledge", tags=["knowledge"])
app.include_router(tools_router, prefix=prefix + "/tools", tags=["tools"])
app.include_router(audit_router, prefix=prefix + "/audit", tags=["audit"])
app.include_router(mock_tools_router, prefix=prefix + "/mock-tools", tags=["mock-tools"])
app.include_router(llm_configs_router, prefix=prefix + "/llm-configs", tags=["llm-configs"])
app.include_router(performance_router, prefix=prefix + "/performance", tags=["performance"])
app.include_router(vector_admin_router, prefix=prefix + "/vector-admin", tags=["vector-admin"])
app.include_router(skills_router, prefix=prefix + "/skills", tags=["skills"])
app.include_router(agent_skills_router, prefix=prefix + "/agents", tags=["agent-skills"])
app.include_router(agent_connections_router, prefix=prefix + "/agent-connections", tags=["agent-connections"])
app.include_router(agent_capabilities_router, prefix=prefix + "/agents", tags=["agent-capabilities"])


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# ── Serve frontend SPA (production) ─────────────────
if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(request: Request, full_path: str):
        """Serve index.html for all non-API routes (SPA client-side routing)."""
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))

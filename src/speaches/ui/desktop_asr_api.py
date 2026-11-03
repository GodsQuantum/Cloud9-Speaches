# Copyright (c) 2026 GodsQuantum
"""Desktop-only custom GGUF discovery/downloads, reusing native Hugging Face cache."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

router = APIRouter(tags=["desktop ASR models"])


class DesktopAsrLookup(BaseModel):
    source: str = Field(min_length=3, max_length=2048)


class DesktopAsrInstall(DesktopAsrLookup):
    filename: str = Field(min_length=5, max_length=255)


def require_native_loopback(request: Request) -> None:
    """Block server deployments, DNS rebinding, and cross-origin browser calls."""
    if os.environ.get("CLOUD9_PORTABLE_MODE") != "1":
        raise HTTPException(status_code=404, detail="Model management is desktop-only")
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Only local desktop callers may manage models")
    port = os.environ.get("UVICORN_PORT", "8005")
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if request.headers.get("host", "") not in allowed_hosts:
        raise HTTPException(status_code=403, detail="Unexpected desktop Host header")
    origin = request.headers.get("origin")
    if origin is not None and origin not in {f"http://{host}" for host in allowed_hosts}:
        raise HTTPException(status_code=403, detail="Cross-origin model changes refused")


@router.post("/v1/desktop-asr/variants")
async def desktop_asr_variants(request: Request, body: DesktopAsrLookup) -> dict:
    require_native_loopback(request)
    from speaches.ui.custom_asr import list_hf_gguf

    try:
        choices = await run_in_threadpool(list_hf_gguf, body.source)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid or unavailable Hugging Face GGUF source") from exc
    return {"variants": [{"label": label, "filename": filename} for label, filename in choices]}


@router.post("/v1/desktop-asr/install")
async def desktop_asr_install(request: Request, body: DesktopAsrInstall) -> dict:
    require_native_loopback(request)
    from speaches.ui.custom_asr import install_hf_gguf

    try:
        detail = await run_in_threadpool(install_hf_gguf, body.source, body.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid or unavailable Hugging Face GGUF file") from exc
    return {"status": "installed", "detail": detail}

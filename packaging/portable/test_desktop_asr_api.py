# Copyright (c) 2026 GodsQuantum
"""The embedded ScribeWatch UI may manage models via verified local-only APIs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

from fastapi import HTTPException, Request
import pytest

from speaches.ui.desktop_asr_api import (
    DesktopAsrInstall,
    DesktopAsrLookup,
    desktop_asr_install,
    desktop_asr_variants,
    require_native_loopback,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def request_for(*, host: str = "127.0.0.1", origin: str | None = None, remote: str = "127.0.0.1") -> Request:
    headers = [(b"host", host.encode("utf-8"))]
    if origin is not None:
        headers.append((b"origin", origin.encode("utf-8")))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/desktop-asr/variants",
            "headers": headers,
            "client": (remote, 52345),
            "server": ("127.0.0.1", 38005),
            "scheme": "http",
            "http_version": "1.1",
        }
    )


def test_desktop_authorized_native_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_PORTABLE_MODE", "1")
    monkeypatch.setenv("UVICORN_PORT", "38005")
    require_native_loopback(request_for(host="127.0.0.1:38005"))
    require_native_loopback(request_for(host="localhost:38005", origin="http://localhost:38005"))


@pytest.mark.parametrize(
    ("host", "origin", "remote"),
    [
        ("evil.example:38005", None, "127.0.0.1"),
        ("127.0.0.1:38005", "https://evil.example", "127.0.0.1"),
        ("127.0.0.1:38005", None, "172.20.30.10"),
        ("127.0.0.1:8005", None, "127.0.0.1"),
    ],
)
def test_desktop_blocked_untrusted_callers(
    monkeypatch: pytest.MonkeyPatch, host: str, origin: str | None, remote: str
) -> None:
    monkeypatch.setenv("CLOUD9_PORTABLE_MODE", "1")
    monkeypatch.setenv("UVICORN_PORT", "38005")
    with pytest.raises(HTTPException) as error:
        require_native_loopback(request_for(host=host, origin=origin, remote=remote))
    assert error.value.status_code == 403


def test_desktop_api_disabled_in_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUD9_PORTABLE_MODE", raising=False)
    with pytest.raises(HTTPException) as error:
        require_native_loopback(request_for(host="127.0.0.1:38005"))
    assert error.value.status_code == 404


def test_discovers_only_native_gguf_variants_without_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_PORTABLE_MODE", "1")
    monkeypatch.setenv("UVICORN_PORT", "38005")

    async def run_sync(func: Callable[..., object], *args: object) -> object:
        return func(*args)

    with (
        patch("speaches.ui.custom_asr.list_hf_gguf", return_value=[("Quantification Q8_0", "test-Q8_0.gguf")]),
        patch("speaches.ui.desktop_asr_api.run_in_threadpool", side_effect=run_sync),
    ):
        response = asyncio.run(
            desktop_asr_variants(
                request_for(host="127.0.0.1:38005"),
                DesktopAsrLookup(source="handy-computer/parakeet-ultra-gguf"),
            )
        )
    assert response["variants"] == [{"label": "Quantification Q8_0", "filename": "test-Q8_0.gguf"}]


def test_import_delegates_to_existing_hf_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_PORTABLE_MODE", "1")
    monkeypatch.setenv("UVICORN_PORT", "38005")

    async def run_sync(func: Callable[..., object], *args: object) -> object:
        return func(*args)

    with (
        patch("speaches.ui.custom_asr.install_hf_gguf", return_value="Model installed") as download,
        patch("speaches.ui.desktop_asr_api.run_in_threadpool", side_effect=run_sync),
    ):
        response = asyncio.run(
            desktop_asr_install(
                request_for(host="127.0.0.1:38005"),
                DesktopAsrInstall(source="handy-computer/parakeet-ultra-gguf", filename="test-Q8_0.gguf"),
            )
        )
    assert response["status"] == "installed"
    download.assert_called_once_with("handy-computer/parakeet-ultra-gguf", "test-Q8_0.gguf")

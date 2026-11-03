# Copyright (c) 2026 GodsQuantum
"""Offline checks for desktop ASR model discovery and optional imports."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from speaches.ui.custom_asr import (
    GGUF_MODELS,
    _safe_public_url,
    custom_model_file,
    install_github_gguf,
    list_hf_gguf,
    list_local_custom_ggufs,
    parse_hf_source,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("handy-computer/parakeet-ultra-gguf", ("handy-computer/parakeet-ultra-gguf", "main", None)),
        (
            "https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf",
            ("handy-computer/Qwen3-ASR-0.6B-gguf", "main", None),
        ),
        (
            "https://huggingface.co/handy-computer/parakeet-ultra-gguf/blob/main/parakeet-ultra-Q8_0.gguf",
            ("handy-computer/parakeet-ultra-gguf", "main", "parakeet-ultra-Q8_0.gguf"),
        ),
    ],
)
def test_hf_source_valid(source: str, expected: tuple[str, str, str | None]) -> None:
    assert parse_hf_source(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "http://huggingface.co/handy-computer/parakeet-ultra-gguf",
        "https://localhost/model.gguf",
        "https://huggingface.co.evil.com/handy-computer/model",
        "https://huggingface.co/handy-computer/../secret",
        "https://huggingface.co/handy-computer/model/blob/main/../other.gguf",
        "https://huggingface.co/handy-computer/model/blob/main/weights.bin",
        "file:///tmp/weights.gguf",
        "https://user:password@huggingface.co/owner/model",
    ],
)
def test_untrusted_hf_sources_rejected(source: str) -> None:
    with pytest.raises(ValueError, match=r"invalide|acceptées|attendu|refusée|Utiliser|doit|GGUF|Seules"):
        parse_hf_source(source)


def test_curated_models_use_native_gguf_conversions() -> None:
    assert len(GGUF_MODELS) >= 3
    assert all(repo.startswith("handy-computer/") and repo.endswith("-gguf") for _, repo in GGUF_MODELS)


def test_gguf_choices_exclude_huge_or_unrecognized_files() -> None:
    fake = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="parakeet-ultra-Q8_0.gguf", size=734003200),
            SimpleNamespace(rfilename="parakeet-ultra-Q4_K_M.gguf", size=470000000),
            SimpleNamespace(rfilename="parakeet-ultra-F32.gguf", size=2800000000),
            SimpleNamespace(rfilename="weights.bin", size=80),
            SimpleNamespace(rfilename="huge-Q8_0.gguf", size=13 * 1024**3),
        ]
    )
    hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *_a, **_k: fake))
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        found = list_hf_gguf("handy-computer/parakeet-ultra-gguf")
    assert [item[1] for item in found] == ["parakeet-ultra-Q8_0.gguf", "parakeet-ultra-Q4_K_M.gguf"]


def test_external_url_must_be_public_github_release() -> None:
    for bad in (
        "http://github.com/a/b/releases/download/v1/a.gguf",
        "https://127.0.0.1/a.gguf",
        "https://evil.com/m.gguf",
        "https://github.com/a/b/blob/main/weights.gguf",
        "https://github.com.evil.com/a/b/releases/download/v1/a.gguf",
    ):
        with pytest.raises(ValueError, match=r"invalide|acceptées|attendu|refusée|Utiliser|doit|GGUF|Seules"):
            _safe_public_url(bad)


class _FakeResponse(BytesIO):
    headers: dict[str, str]

    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        super().__init__(payload)
        self.headers = headers or {}


def test_external_gguf_sha256_install_and_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    payload = b"GGUF" + b"synthetic-fixture" * 5
    response = _FakeResponse(payload, {"Content-Length": str(len(payload))})
    opener = SimpleNamespace(open=lambda _request, **_kwargs: response)
    source = "https://github.com/org/repo/releases/download/v1/fake-Q8_0.gguf"
    with patch("speaches.ui.custom_asr.build_opener", return_value=opener):
        installed = install_github_gguf(source, sha256(payload).hexdigest())
    assert "custom/" in installed
    known = list_local_custom_ggufs()
    assert len(known) == 1
    assert custom_model_file(known[0][0]) == known[0][1]
    assert known[0][1].read_bytes() == payload
    assert not list(tmp_path.rglob("*.partial"))
    # Existing file is checked and not fetched again.
    with patch("speaches.ui.custom_asr.build_opener", side_effect=AssertionError("redownload")):
        assert "déjà installé" in install_github_gguf(source, sha256(payload).hexdigest())


def test_bad_checksum_does_not_publish_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    response = _FakeResponse(b"wrong")
    opener = SimpleNamespace(open=lambda _request, **_kwargs: response)
    with patch("speaches.ui.custom_asr.build_opener", return_value=opener), pytest.raises(ValueError, match="SHA-256"):
        install_github_gguf("https://github.com/org/repo/releases/download/v1/fake-Q8_0.gguf", "a" * 64)
    assert list_local_custom_ggufs() == []
    assert not list(tmp_path.rglob("*.partial"))


def test_hf_selection_is_remembered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from speaches.ui.custom_asr import preferred_hf_gguf, set_preferred_hf_gguf

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    repo = "handy-computer/parakeet-ultra-gguf"
    assert preferred_hf_gguf(repo) is None
    set_preferred_hf_gguf(repo, "parakeet-ultra-Q4_K_M.gguf")
    assert preferred_hf_gguf(repo) == "parakeet-ultra-Q4_K_M.gguf"
    assert list_local_custom_ggufs() == []


def test_hf_import_uses_cache_without_bulk_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from speaches.ui.custom_asr import install_hf_gguf, preferred_hf_gguf

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    choices = [("qwen-Q8_0.gguf · 1 Mio", "qwen-Q8_0.gguf")]
    calls: list[str] = []

    def download(*, repo_id: str, filename: str, revision: str) -> str:
        assert repo_id == "test/model"
        assert revision == "f" * 40
        calls.append(filename)
        return str(tmp_path / filename)

    info = SimpleNamespace(sha="f" * 40, siblings=[SimpleNamespace(rfilename="qwen-Q8_0.gguf", size=1048576)])
    hf = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda *_args, **_kwargs: info), hf_hub_download=download
    )
    with (
        patch.dict(sys.modules, {"huggingface_hub": hf}),
        patch("speaches.ui.custom_asr.list_hf_gguf", return_value=choices),
    ):
        result = install_hf_gguf("test/model", "qwen-Q8_0.gguf")
    assert "qwen-Q8_0.gguf" in result
    assert calls == ["README.md", "qwen-Q8_0.gguf"]
    assert preferred_hf_gguf("test/model") == "qwen-Q8_0.gguf"

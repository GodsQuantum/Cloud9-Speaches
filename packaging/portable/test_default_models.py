# Copyright (c) 2026 GodsQuantum
"""Deterministic offline tests for the native model bootstrap."""

import hashlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from default_models import PinnedModel, download_aligner, verify_model


def model_fixture(data: bytes, *, sha: str | None = None) -> PinnedModel:
    return PinnedModel(
        "tiny fixture",
        "aligner",
        "cstr/mock-model",
        "a" * 40,
        "test.gguf",
        sha or hashlib.sha256(data).hexdigest(),
        len(data),
        "test only",
    )


def test_download_is_atomic_and_checked() -> None:
    data = b"test GGUF payload"
    with TemporaryDirectory(prefix="trio-model-test-") as tmp:
        dest = Path(tmp) / "model" / "test.gguf"
        with patch("urllib.request.urlopen", return_value=BytesIO(data)) as fetch:
            result = download_aligner(model_fixture(data), dest)
        assert result == dest
        assert dest.read_bytes() == data
        assert not dest.is_symlink()
        assert not list(dest.parent.glob("*.download.*"))
        assert fetch.call_count == 1
        verify_model(dest, model_fixture(data))


def test_bad_download_does_not_replace_existing_model() -> None:
    data = b"expected"
    wrong = b"mismatch"
    with TemporaryDirectory(prefix="trio-model-test-") as tmp:
        dest = Path(tmp) / "models" / "test.gguf"
        with patch("urllib.request.urlopen", return_value=BytesIO(wrong)):
            try:
                download_aligner(model_fixture(data), dest)
            except ValueError:
                pass
            else:
                raise AssertionError("Hash mismatch was accepted")
        assert not dest.exists()
        assert not list(dest.parent.glob("*.download.*"))


def test_existing_valid_model_is_idempotent_and_not_redownloaded() -> None:
    data = b"installed"
    with TemporaryDirectory(prefix="trio-model-test-") as tmp:
        dest = Path(tmp) / "test.gguf"
        dest.write_bytes(data)
        with patch("urllib.request.urlopen") as fetch:
            assert download_aligner(model_fixture(data), dest) == dest
        fetch.assert_not_called()

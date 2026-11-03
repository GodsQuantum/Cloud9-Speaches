# Copyright (c) 2026 GodsQuantum
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_windows_frozen_stdout_and_stderr_are_restored_before_start() -> None:
    entry = (ROOT / "packaging/portable/entry.py").read_text()
    assert "ensure_logging_streams()" in entry
    assert "if sys.stderr is None:" in entry
    assert "if sys.stdout is None:" in entry


def test_smoke_child_is_bounded_and_finally_reaped() -> None:
    entry = (ROOT / "packaging/portable/entry.py").read_text()
    assert "child.wait(timeout=15)" in entry
    assert "time.monotonic() + 110" in entry
    assert "stdout=output" in entry
    assert "stderr=subprocess.STDOUT" in entry


def test_packaging_smoke_exercises_real_executable_and_empty_cache() -> None:
    source = (ROOT / "packaging/portable/build.py").read_text()
    assert "smoke_frozen_app(frozen, Path(independent))" in source
    assert 'env["CLOUD9_PORTABLE_DATA_DIR"] = str(isolated)' in source
    assert 'urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"' in source
    assert "process.wait(timeout=15)" in source


def test_windowed_fatal_errors_are_logged_not_modal() -> None:
    entry = (ROOT / "packaging/portable/entry.py").read_text()
    assert 'logger.exception("Cloud9-Speaches failed to start")' in entry
    assert "raise SystemExit(1) from None" in entry

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from build import create_dmg


def test_dmg_resource_busy_retries_and_recovers(tmp_path: Path) -> None:
    app = tmp_path / "Cloud9-Speaches.app"
    app.mkdir()
    result = tmp_path / "Speaches.dmg"
    busy = subprocess.CompletedProcess(["hdiutil"], 1, stdout="", stderr="hdiutil: create failed - Resource busy")
    good = subprocess.CompletedProcess(["hdiutil"], 0, stdout="created", stderr="")
    with patch("build.subprocess.run", side_effect=[busy, good]) as runner, patch("build.time.sleep") as sleeper:
        create_dmg(app, result)
    assert runner.call_count == 2
    sleeper.assert_called_once()


def test_dmg_other_error_fails_without_retry(tmp_path: Path) -> None:
    app = tmp_path / "Cloud9-Speaches.app"
    app.mkdir()
    result = tmp_path / "Speaches.dmg"
    failed = subprocess.CompletedProcess(["hdiutil"], 1, stdout="", stderr="hdiutil: image too large")
    with (
        patch("build.subprocess.run", return_value=failed) as runner,
        patch("build.time.sleep") as sleeper,
        pytest.raises(subprocess.CalledProcessError),
    ):
        create_dmg(app, result)
    runner.assert_called_once()
    sleeper.assert_not_called()


def test_dmg_busy_exhausts_max_attempts(tmp_path: Path) -> None:
    app = tmp_path / "Cloud9-Speaches.app"
    app.mkdir()
    result = tmp_path / "Speaches.dmg"
    failed = subprocess.CompletedProcess(["hdiutil"], 1, stdout="", stderr="Resource busy")
    with (
        patch("build.subprocess.run", return_value=failed) as runner,
        patch("build.time.sleep") as sleeper,
        pytest.raises(subprocess.CalledProcessError),
    ):
        create_dmg(app, result)
    assert runner.call_count == 5
    assert sleeper.call_count == 4

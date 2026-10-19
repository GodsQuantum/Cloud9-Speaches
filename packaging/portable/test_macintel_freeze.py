# Copyright (c) 2026 GodsQuantum
from __future__ import annotations

from build import intel_legacy_collect_flags


def test_mac_intel_collects_physical_speechbrain_source() -> None:
    assert intel_legacy_collect_flags("macos-amd64") == ["--collect-all", "speechbrain"]


def test_other_platforms_remain_unchanged() -> None:
    for name in ("macos-arm64", "windows-amd64", "linux-amd64", "linux-arm64"):
        assert intel_legacy_collect_flags(name) == []

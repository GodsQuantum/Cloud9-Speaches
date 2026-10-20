from __future__ import annotations

from pathlib import Path
import sys


def realtime_console_dist() -> Path:
    """Locate immutable UI assets independent of the launch directory."""
    frozen_root = getattr(sys, "_MEIPASS", None) if getattr(sys, "frozen", False) else None
    root = Path(frozen_root) if isinstance(frozen_root, str) else Path(__file__).resolve().parents[2]
    return root / "realtime-console" / "dist"

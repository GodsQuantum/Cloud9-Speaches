# Copyright (c) 2026 GodsQuantum
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

logger = logging.getLogger("speaches.portable")
APP = "Cloud9-Speaches"


def app_dir() -> Path:
    if platform.system() == "Windows":
        base = Path(os.environ["LOCALAPPDATA"])
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP


def runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def configure(port: int) -> Path:
    if not 0 < port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    base = app_dir()
    models = base / "models"
    cache = base / "cache"
    for path in (base, models, cache):
        path.mkdir(parents=True, exist_ok=True)
    bundle = runtime_dir()
    binary = bundle / "bin" / ("crispasr.exe" if os.name == "nt" else "crispasr")
    if binary.is_file():
        os.environ.setdefault("CRISPASR_BIN", str(binary))
    path_entry = str(bundle / "bin")
    os.environ["PATH"] = path_entry + os.pathsep + os.getenv("PATH", "")
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("ALIGN_MODEL_ROOT", str(models / "aligners"))
    os.environ.setdefault("CLOUD9_NATIVE_CTC_ENABLED", "1")
    os.environ.setdefault("CLOUD9_CTC_BACKEND", "cpu")
    os.environ.setdefault("TRANSCRIBE_BACKEND", "auto")
    os.environ["UVICORN_HOST"] = "127.0.0.1"
    os.environ["UVICORN_PORT"] = str(port)
    return binary


def check_install(port: int) -> None:
    binary = configure(port)
    if not binary.is_file():
        raise FileNotFoundError(f"CrispASR CPU/Metal binary missing from {binary}")
    result = subprocess.run([str(binary), "--help"], check=False, capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"CrispASR failed to start: {result.stderr[-400:]}")
    from speaches.main import create_app

    application = create_app()
    paths = {getattr(route, "path", "") for route in application.routes}
    if "/health" not in paths:
        raise RuntimeError("Bundled real FastAPI routes not found")
    logger.info("SPEACHES_BUNDLE_READY os=%s backend=%s", platform.system(), os.environ["CLOUD9_CTC_BACKEND"])


def port_is_busy(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def run_smoke(port: int) -> None:
    if port_is_busy(port):
        raise RuntimeError(f"Port {port} already in use; refusing to validate another process")
    child = subprocess.Popen([sys.executable, "--no-browser", "--port", str(port)])
    try:
        for _ in range(180):
            if child.poll() is not None:
                raise RuntimeError(f"Cloud9-Speaches process exited: {child.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/v1/audio/alignment/health", timeout=2
                        ) as alignment:
                            if alignment.status == 200:
                                logger.info("SPEACHES_PORTABLE_API_SMOKE_OK port=%d", port)
                                return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.5)
        raise RuntimeError("FastAPI/Gradio server did not become healthy")
    finally:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("UVICORN_PORT", "8005")))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check-install", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.smoke:
        check_install(args.port)
        run_smoke(args.port)
        return
    if args.check_install:
        check_install(args.port)
        return
    configure(args.port)
    if port_is_busy(args.port):
        raise RuntimeError(f"Port {args.port} in use; do not open another app's server")
    import uvicorn

    from speaches.main import create_app

    if not args.no_browser:
        webbrowser.open(f"http://127.0.0.1:{args.port}/")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()

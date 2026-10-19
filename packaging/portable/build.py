# Copyright (c) 2026 GodsQuantum
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from fetch_crispasr import acquire

logger = logging.getLogger(__name__)
NAME = "Cloud9-Speaches"
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(*args: str | Path, timeout: int = 1800) -> None:
    logger.info("Running %s", [str(arg) for arg in args])
    subprocess.run([str(a) for a in args], check=True, timeout=timeout)


def variant() -> str:
    system = platform.system()
    arch = platform.machine().lower()
    if system == "Windows":
        if arch not in ("amd64", "x86_64"):
            raise RuntimeError(f"Windows architecture unsupported: {arch}")
        return "windows-amd64"
    if system == "Darwin":
        return "macos-arm64" if arch in ("arm64", "aarch64") else "macos-amd64"
    if system == "Linux":
        return "linux-arm64" if arch in ("arm64", "aarch64") else "linux-amd64"
    raise RuntimeError(f"Unsupported OS: {system}")


def build_candidate() -> Path:
    target = variant()
    stage = HERE / "dist" / target
    build = HERE / "build"
    stage.mkdir(parents=True, exist_ok=True)
    ffmpeg = stage / NAME
    if ffmpeg.exists():
        raise RuntimeError(f"Refusing to overwrite existing frozen distribution: {ffmpeg}")
    run(
        sys.executable,
        "-m",
        "PyInstaller",
        "--onedir",
        "--noconfirm",
        "--clean",
        "--name",
        NAME,
        "--paths",
        ROOT / "src",
        "--collect-all",
        "speaches",
        "--collect-all",
        "gradio",
        "--collect-all",
        "gradio_client",
        "--collect-all",
        "language_tags",
        "--collect-all",
        "pyannote.audio",
        "--collect-data",
        "safehttpx",
        "--collect-submodules",
        "onnx_asr",
        "--collect-submodules",
        "faster_whisper",
        "--distpath",
        stage,
        "--workpath",
        build / target,
        "--specpath",
        build / target,
        HERE / "entry.py",
        timeout=5400,
    )
    frozen = ffmpeg / (NAME + (".exe" if os.name == "nt" else ""))
    assert frozen.is_file(), frozen
    bin_dir = ffmpeg / "bin"
    # The external acoustic alignment engine is a real upstream executable.
    # Models remain in per-user storage and are never written into the signed app.
    acquire(target, bin_dir)
    for source in bin_dir.glob("*.zip"):
        source.unlink()
    for source in bin_dir.glob("*.tar.gz"):
        source.unlink()
    run(frozen, "--check-install", timeout=180)
    if os.getenv("PORTABLE_E2E_SMOKE", "1") == "1":
        run(frozen, "--smoke", "--port", "39085", timeout=180)
    logger.info("SPEACHES_PORTABLE_CHECK_PASS %s", target)
    if target.startswith("macos"):
        app = stage / (NAME + ".app")
        macos = app / "Contents" / "MacOS"
        resource = app / "Contents" / "Resources"
        macos.mkdir(parents=True)
        resource.mkdir(parents=True)
        shutil.move(str(ffmpeg), str(resource / NAME))
        script = macos / NAME
        script.write_text(
            '#!/bin/sh\nROOT="$(CDPATH= cd -- "$(dirname "$0")/../Resources" && pwd)"\n'
            f'exec "$ROOT/{NAME}/{NAME}" "$@"\n'
        )
        script.chmod(0o755)
        plist = {
            "CFBundleIdentifier": "io.github.GodsQuantum.Cloud9-Speaches",
            "CFBundleExecutable": NAME,
            "CFBundleName": NAME,
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "1",
            "CFBundlePackageType": "APPL",
            "LSMinimumSystemVersion": "11.0",
        }
        with (app / "Contents" / "Info.plist").open("wb") as file:
            plistlib.dump(plist, file)
        run(app / "Contents" / "MacOS" / NAME, "--check-install", timeout=180)
        run("codesign", "--force", "--deep", "--sign", "-", "--timestamp=none", app)
        run("codesign", "--verify", "--deep", "--strict", app)
        result = stage / f"{NAME}-{target}.dmg"
        run("hdiutil", "create", "-volname", NAME, "-format", "UDZO", "-srcfolder", app, result)
    elif target.startswith("windows"):
        result = stage / f"{NAME}-{target}.zip"
        with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in ffmpeg.rglob("*"):
                if path.is_file():
                    archive.write(path, (Path(NAME) / path.relative_to(ffmpeg)).as_posix())
        run("choco", "install", "nsis", "--yes", "--no-progress")
        makensis = next(
            (
                candidate
                for candidate in (
                    shutil.which("makensis"),
                    r"C:\Program Files (x86)\NSIS\makensis.exe",
                    r"C:\Program Files\NSIS\makensis.exe",
                )
                if candidate and Path(candidate).is_file()
            ),
            None,
        )
        if makensis is None:
            raise RuntimeError("NSIS compiler is not available")
        installer = stage / f"{NAME}-{target}-setup.exe"
        run(
            makensis,
            "/V2",
            "/DSTAGING=" + str(ffmpeg),
            "/DOUTPUT=" + str(installer),
            HERE / "installer.nsi",
        )
        with tempfile.TemporaryDirectory(prefix="trio-speaches-installed-") as installed_parent:
            installed = Path(installed_parent) / NAME
            run(installer, "/S", "/D=" + str(installed))
            app_binary = installed / (NAME + ".exe")
            if not app_binary.is_file():
                raise RuntimeError("NSIS installer failed to install real frozen binary")
            run(app_binary, "--check-install", timeout=180)
            run(app_binary, "--smoke", "--port", "39086", timeout=180)
            run(installed / "Uninstall.exe", "/S")
            for _ in range(120):
                if not app_binary.exists():
                    break
                time.sleep(0.25)
            if app_binary.exists():
                raise RuntimeError("NSIS uninstall did not remove frozen binary after 30 seconds")
        logger.info("WINDOWS_SPEACHES_NSIS_INSTALL_SMOKE_UNINSTALL_OK")
        with installer.open("rb") as stream:
            install_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        installer.with_suffix(installer.suffix + ".sha256").write_text(f"{install_digest}  {installer.name}\n")
    else:
        result = stage / f"{NAME}-{target}.zip"
        with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in ffmpeg.rglob("*"):
                if path.is_file():
                    archive.write(path, (Path(NAME) / path.relative_to(ffmpeg)).as_posix())
    with result.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    result.with_suffix(result.suffix + ".sha256").write_text(f"{checksum}  {result.name}\n")
    logger.info("SPEACHES_PORTABLE_ARTIFACT_SHA256 %s %s", result, checksum)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_candidate()

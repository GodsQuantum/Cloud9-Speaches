# Copyright (c) 2026 GodsQuantum
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import zipfile

logger = logging.getLogger(__name__)
RELEASE = "v0.8.43"
BASE = f"https://github.com/CrispStrobe/CrispASR/releases/download/{RELEASE}"
ASSETS: dict[str, tuple[str, str]] = {
    "windows-amd64": (
        "crispasr-windows-x86_64-cpu.zip",
        "997bc39f541e5d5d0bd14144159c038bd95d6ab8b523718c34a54e03ab404f47",
    ),
    "macos-arm64": ("crispasr-macos-arm64.tar.gz", "5c13fbbc12eb45c3f01823a6bbfbbb22e0682bc76c6b78a0d9511da1aaf88831"),
    "macos-amd64": (
        "crispasr-macos-x86_64-cpu-legacy.tar.gz",
        "f632510b55552545ff0c18d6fe716f5c890c948fe96c5ef25cec49aeba9d275a",
    ),
    "linux-amd64": (
        "crispasr-linux-x86_64-cpu-legacy.tar.gz",
        "c00f01bf6395fe0b6aa6b6b8313f8da7a434768a5bdb292a63bcb549d8ff77b0",
    ),
    "linux-arm64": ("crispasr-linux-arm64.tar.gz", "ae6e3f13e795fa6dec55d20c8bf1c96fca0bd581eab1495f344cb9f9195ec2d2"),
}


def safe_relative_path(member: str) -> Path:
    if "\\" in member or ":" in member:
        raise ValueError(f"Unsafe archive entry {member}")
    value = PurePosixPath(member)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Unsafe archive entry {member}")
    parts = value.parts[1:]
    if not parts:
        return Path()
    return Path(*parts)


def acquire(variant: str, folder: Path) -> None:
    name, sha = ASSETS[variant]
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / name
    if not archive.exists():
        url = f"{BASE}/{name}"
        if not url.startswith("https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/"):
            raise ValueError("CrispASR source must use pinned GitHub HTTPS endpoint")
        request = urllib.request.Request(url, headers={"User-Agent": "Cloud9-Speaches-Desktop/1.0"})  # noqa: S310
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as dest:  # noqa: S310
            while chunk := response.read(1024 * 1024):
                dest.write(chunk)
    with archive.open("rb") as stream:
        got = hashlib.file_digest(stream, "sha256").hexdigest()
    if got != sha:
        archive.unlink(missing_ok=True)
        raise ValueError(f"Bad digest {name}: {got}")
    suffix = ".exe" if variant.startswith("windows") else ""
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as entries:
            for member in entries.infolist():
                path = safe_relative_path(member.filename)
                if path == Path() or member.is_dir():
                    continue
                target = folder / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with entries.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    else:
        with tarfile.open(archive, "r:gz") as entries:
            for member in entries:
                path = safe_relative_path(member.name)
                if path == Path() or member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Non-regular archive member {member.name}")
                target = folder / path
                target.parent.mkdir(parents=True, exist_ok=True)
                source = entries.extractfile(member)
                if source is None:
                    raise ValueError(f"Missing data {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    binary = folder / f"crispasr{suffix}"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    binary.chmod(0o755)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.txt"):
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    logger.info("CRISPASR_ASSET_SHA256_OK %s %s", variant, got)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=ASSETS.keys())
    parser.add_argument("--dest", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    acquire(args.variant, args.dest)

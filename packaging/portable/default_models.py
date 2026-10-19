# Copyright (c) 2026 GodsQuantum
"""Optional, atomic, checksum-verified local model bootstrap for native Speaches.

Large acoustic models are NOT hidden in installers or automatically fetched
during a smoke test. The user runs --install-default-models explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
from typing import Literal
import urllib.request


@dataclass(frozen=True)
class PinnedModel:
    name: str
    kind: Literal["aligner", "transcriber"]
    repo: str
    revision: str
    filename: str
    sha256: str
    size: int
    license: str


MODELS = (
    PinnedModel(
        name="French FastConformer CTC Q4_K",
        kind="aligner",
        repo="cstr/stt-fr-fastconformer-hybrid-ctc-large-GGUF",
        revision="02059e0cd6b0d3aea6f220921d3ccd880d12666d",
        filename="stt-fr-fastconformer-hybrid-ctc-large-q4_k.gguf",
        sha256="05c42c40547fdcdcd04bd216970463d927f641a3064f30330f64ad65ee0fa4dd",
        size=69651168,
        license="CC-BY-4.0, derived from NVIDIA FastConformer CTC",
    ),
    PinnedModel(
        name="Whisper Large-v3-Turbo GGUF Q6_K",
        kind="transcriber",
        repo="handy-computer/whisper-large-v3-turbo-gguf",
        revision="18ada752005f5fddef499256f9a458d0b37a203a",
        filename="whisper-large-v3-turbo-Q6_K.gguf",
        sha256="b8b4d57e1a6fc49bcd7843faadf224c27cb27707e13db7e6dbfbfffe39878103",
        size=692536896,
        license="MIT (OpenAI Whisper; transcribe.cpp conversion)",
    ),
)


def verify_model(path: Path, model: PinnedModel) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != model.size:
        raise ValueError(f"{model.name}: size {size}, expected {model.size}")
    with path.open("rb") as data:
        got = hashlib.file_digest(data, "sha256").hexdigest()
    if got != model.sha256:
        raise ValueError(f"{model.name}: SHA256 mismatch")


def model_path(model: PinnedModel, align_root: Path, hf_home: Path) -> Path:
    if model.kind == "aligner":
        return align_root / model.repo.replace("/", "--") / model.filename
    # Cache discovery is delegated to huggingface_hub rather than replicating
    # its snapshot-directory/symlink protocol in application code.
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(
        repo_id=model.repo,
        filename=model.filename,
        revision=model.revision,
        cache_dir=hf_home / "hub",
    )
    if not isinstance(cached, str):
        raise FileNotFoundError(f"Whisper model is not cached: {model.name}")
    return Path(cached)


def download_aligner(model: PinnedModel, target: Path) -> Path:
    if model.kind != "aligner":
        raise ValueError("Internal error: aligner only")
    if target.exists():
        verify_model(target, model)
        return target
    url = f"https://huggingface.co/{model.repo}/resolve/{model.revision}/{model.filename}"
    if not url.startswith("https://huggingface.co/cstr/"):
        raise ValueError("Unexpected aligner source host")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download.{os.getpid()}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Cloud9-Speaches-Installer/1.0"})  # noqa: S310
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("xb") as output:  # noqa: S310
            shutil.copyfileobj(source, output, length=1024 * 1024)
        verify_model(temporary, model)
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def download_transcriber(model: PinnedModel, hf_home: Path) -> Path:
    if model.kind != "transcriber":
        raise ValueError("Internal error: transcriber only")
    from huggingface_hub import hf_hub_download

    # This creates an upstream-managed HF cache (no application-generated
    # filesystem aliases or symlinks).
    kwargs = {
        "repo_id": model.repo,
        "revision": model.revision,
        "cache_dir": hf_home / "hub",
    }
    path = Path(hf_hub_download(filename=model.filename, **kwargs))
    verify_model(path, model)
    hf_hub_download(filename="README.md", **kwargs)
    return path


def install_defaults(align_root: Path, hf_home: Path, *, only: str = "both") -> None:
    if only not in ("both", "aligner", "transcriber"):
        raise ValueError("Unknown default model category")
    for model in MODELS:
        if only not in ("both", model.kind):
            continue
        print(f"Installing {model.name} from {model.repo}@{model.revision} ({model.size} bytes)")
        if model.kind == "aligner":
            target = download_aligner(model, model_path(model, align_root, hf_home))
        else:
            target = download_transcriber(model, hf_home)
        verify_model(target, model)
        print(f"MODEL_INSTALLED_AND_SHA256_VERIFIED {model.name} {target}")
    # Preserve a human-readable attribution record alongside model storage.
    attribution = align_root.parent / "MODEL_ATTRIBUTION.txt"
    attribution.parent.mkdir(parents=True, exist_ok=True)
    attribution.write_text(
        "\n".join(f"{m.name}\n{m.repo}@{m.revision}\nSHA256 {m.sha256}\nLicense {m.license}\n" for m in MODELS),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit("Use the Cloud9-Speaches app --install-default-models command")

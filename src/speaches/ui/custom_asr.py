# Copyright (c) 2026 GodsQuantum
"""Desktop-only custom ASR GGUF imports; reuse the native Hugging Face cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

# Only public HF repos and GitHub release assets; never fetch LAN/local addresses.
_ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "github.com",
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
        "github-production-release-asset-2e65be.s3.amazonaws.com",
    }
)
_QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "F16", "BF16")
_MAX_BYTES = 12 * 1024**3
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,90}")
GGUF_MODELS = (
    ("Parakeet Ultra · 25 langues · haute précision", "handy-computer/parakeet-ultra-gguf"),
    ("Qwen3-ASR 0.6B · multilingue", "handy-computer/Qwen3-ASR-0.6B-gguf"),
    ("Nemotron 3.5 ASR 0.6B · streaming multilingue", "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf"),
    ("Qwen3-ASR 1.7B · qualité, plus lourd", "handy-computer/Qwen3-ASR-1.7B-gguf"),
)


def parse_hf_source(value: str) -> tuple[str, str, str | None]:
    """Accept a repo ID or HF web URL; reject API endpoints and unknown hosts."""
    value = value.strip()
    if not value or len(value) > 2048:
        raise ValueError("Saisir un dépôt Hugging Face ou son URL publique.")
    if "://" not in value:
        if not _REPO.fullmatch(value):
            raise ValueError("Format attendu : organisation/nom-du-modele.")
        return value, "main", None
    url = urlparse(value)
    if (
        url.scheme != "https"
        or url.hostname not in {"huggingface.co", "hf.co"}
        or url.username
        or url.password
        or url.port
    ):
        raise ValueError("Seules les URLs HTTPS de huggingface.co ou hf.co sont acceptées ici.")
    parts = [unquote(piece) for piece in url.path.strip("/").split("/")]
    if len(parts) < 2 or not _REPO.fullmatch("/".join(parts[:2])):
        raise ValueError("URL de dépôt Hugging Face invalide.")
    repo = "/".join(parts[:2])
    if len(parts) == 2:
        return repo, "main", None
    if len(parts) < 4 or parts[2] not in {"blob", "resolve", "tree"}:
        raise ValueError("URL attendue : /organisation/depot ou /blob/branche/fichier.gguf.")
    revision = parts[3]
    if revision in {".", ".."} or not _NAME.fullmatch(revision):
        raise ValueError("Révision Hugging Face invalide.")
    filename = "/".join(parts[4:]) if len(parts) > 4 else None
    if filename and (
        len(filename) > 255 or any(not _NAME.fullmatch(p) or p in {".", ".."} for p in filename.split("/"))
    ):
        raise ValueError("Nom de fichier invalide.")
    if filename and not filename.lower().endswith(".gguf"):
        raise ValueError("Le fichier sélectionné doit être un GGUF.")
    return repo, revision, filename


def list_hf_gguf(source: str) -> list[tuple[str, str]]:
    """List only GGUF files, bounded by size; no weight downloads on discovery."""
    from huggingface_hub import HfApi

    repo, revision, file_hint = parse_hf_source(source)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    choices: list[tuple[str, str]] = []
    for file in info.siblings or []:
        name = file.rfilename
        if not name.lower().endswith(".gguf") or (not file.size or file.size > _MAX_BYTES):
            continue
        if file_hint and name != file_hint:
            continue
        if not any(name.upper().endswith("-" + q + ".GGUF") for q in _QUANTS):
            continue
        size = f" · {file.size / 1024**2:.0f} Mio" if file.size is not None else ""
        choices.append((name + size, name))
    return sorted(choices, key=lambda t: (not t[1].upper().endswith("-Q8_0.GGUF"), t[1]))[:48]


def install_hf_gguf(source: str, filename: str) -> str:
    """HF Hub downloader is atomic, resumes safely and checks repository blobs."""
    from huggingface_hub import hf_hub_download

    repo, revision, file_hint = parse_hf_source(source)
    matches = {name for _, name in list_hf_gguf(source)}
    if filename not in matches or (file_hint and filename != file_hint):
        raise ValueError("Le fichier GGUF n'appartient pas au dépôt ou dépasse 12 Gio.")
    # README is required for native registry's model-card filtering.
    hf_hub_download(repo_id=repo, filename="README.md", revision=revision)
    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
    set_preferred_hf_gguf(repo, filename)
    return f"{repo} · {Path(path).name} installé dans le cache HF ; sélectionnez ce modèle dans Transcription."


def _preferences_path() -> Path:
    return custom_model_root() / "selected-quantizations.json"


def preferred_hf_gguf(repo: str) -> str | None:
    try:
        selected = json.loads(_preferences_path().read_text(encoding="utf-8")).get(repo)
    except (OSError, ValueError, AttributeError):
        return None
    return selected if isinstance(selected, str) and selected.endswith(".gguf") else None


def set_preferred_hf_gguf(repo: str, filename: str) -> None:
    if not _REPO.fullmatch(repo) or not _NAME.fullmatch(Path(filename).name):
        raise ValueError("Préférence de modèle invalide.")
    path = _preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            current = {}
    except (OSError, ValueError):
        current = {}
    current[repo] = filename
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(current, ensure_ascii=False, sort_keys=True) + chr(10), encoding="utf-8")
    temp.replace(path)


def custom_model_root() -> Path:
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return root / "custom-asr"


def list_local_custom_ggufs() -> list[tuple[str, Path]]:
    root = custom_model_root()
    if not root.is_dir():
        return []
    models = []
    for folder in root.iterdir():
        if not folder.is_dir() or not _SLUG.fullmatch(folder.name) or folder.is_symlink():
            continue
        for file in folder.glob("*.gguf"):
            if file.is_file() and not file.is_symlink():
                models.append((f"custom/{folder.name}", file))
                break
    return models


def custom_model_file(model_id: str) -> Path | None:
    for candidate, file in list_local_custom_ggufs():
        if candidate == model_id:
            return file
    return None


def _safe_public_url(value: str) -> str:
    p = urlparse(value.strip())
    if p.scheme != "https" or p.hostname not in _ALLOWED_DOWNLOAD_HOSTS or p.username or p.password or p.port:
        raise ValueError(
            "URL refusée : seuls les GGUF de GitHub Releases en HTTPS sont pris en charge hors Hugging Face."
        )
    if p.hostname == "github.com" and not re.fullmatch(
        r"/[^/]+/[^/]+/releases/download/[^/]+/[^/]+\.gguf", p.path, re.IGNORECASE
    ):
        raise ValueError("Utiliser le lien direct vers un fichier .gguf d'une release GitHub.")
    if p.hostname != "github.com" and not p.path:
        raise ValueError("Lien de téléchargement invalide.")
    return value.strip()


class _SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl) -> Request | None:  # noqa: ANN001
        _safe_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def install_github_gguf(source: str, sha256: str) -> str:
    """Optional public GitHub release import; only ready-to-use transcribe.cpp GGUF."""
    source = _safe_public_url(source)
    if urlparse(source).hostname != "github.com":
        raise ValueError("Le lien initial doit pointer vers une release GitHub publique.")
    sha256 = sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("SHA-256 du fichier obligatoire (64 caractères hexadécimaux).")
    name = unquote(Path(urlparse(source).path).name)
    if not _NAME.fullmatch(name) or not name.lower().endswith(".gguf"):
        raise ValueError("Nom de fichier GGUF invalide.")
    safe_name = re.sub(r"[^a-z0-9]+", "-", name[:-5].lower()).strip("-")[:48] or "gguf"
    slug = safe_name + "-" + hashlib.sha256(source.encode()).hexdigest()[:10]
    target = custom_model_root() / slug / name
    if target.is_file():
        with target.open("rb") as existing:
            actual_sha = hashlib.file_digest(existing, "sha256").hexdigest()
        if actual_sha != sha256:
            raise ValueError("Le modèle déjà présent ne correspond pas au SHA-256 fourni.")
        return f"custom/{slug} : déjà installé."
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name("." + name + "." + str(os.getpid()) + ".partial")
    digest = hashlib.sha256()
    total = 0
    try:
        request = Request(source, headers={"User-Agent": "Cloud9-Speaches/1.0"})  # noqa: S310 (HTTPS validated)
        with build_opener(_SafeRedirect()).open(request, timeout=90) as response, tmp.open("xb") as output:
            if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) > _MAX_BYTES:
                raise ValueError("Modèle trop volumineux (>12 Gio).")
            while chunk := response.read(1024**2):
                total += len(chunk)
                if total > _MAX_BYTES:
                    raise ValueError("Modèle trop volumineux (>12 Gio).")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError("SHA-256 incorrect : téléchargement refusé.")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return (
        f"custom/{slug} : GGUF vérifié et installé ({total / 1024**2:.0f} Mio). Compatibilité transcribe.cpp à tester."
    )

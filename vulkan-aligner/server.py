"""Cloud9 Vulkan-only CTC forced alignment API, backed by upstream CrispASR.

Model weights live in the shared read-only NVMe registry. Per-request subprocess
keeps ggml Vulkan contexts isolated; no fabricated word boundaries or CPU fallback.
"""
from __future__ import annotations
import asyncio
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata
import wave

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from alignment_groups import groups_for_alignment

ROOT = Path(os.getenv("ALIGN_MODEL_ROOT", "/ai-models/audio/asr/aligners"))
BIN = os.getenv("CRISPASR_BIN", "/opt/crispasr/bin/crispasr")
LANGUAGES = frozenset("bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk".split())
MODELS = {
    "fastconformer-fr-q4": ("cstr--stt-fr-fastconformer-hybrid-ctc-large-GGUF",
                            "stt-fr-fastconformer-hybrid-ctc-large-q4_k.gguf"),
    "fastconformer-en-q4": ("cstr--stt-en-fastconformer-hybrid-ctc-large-GGUF",
                            "stt-en-fastconformer-hybrid-ctc-large-q4_k.gguf"),
    "wav2vec2-zh-q4": ("cstr--wav2vec2-large-xlsr-53-chinese-zh-cn-GGUF",
                        "wav2vec2-large-xlsr-53-chinese-zh-cn-q4_k.gguf"),
    "canary-ctc-multilingual-q4": ("cstr--canary-ctc-aligner-GGUF",
                                    "canary-ctc-aligner-q4_k.gguf"),
}
app = FastAPI(title="Cloud9 Vulkan CTC Aligner", version="1.0.0")
gate = asyncio.Semaphore(1)


def detect_language(text: str) -> str:
    if re.search(r"[\u3400-\u9fff]", text):
        return "zh"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        raise ValueError("Automatic language detection unavailable or inconclusive; specify language")


def choose_model(model: str | None, language: str, transcript: str) -> tuple[str, str]:
    code = language.strip().lower().split("-")[0]
    if code in ("", "auto", "unknown", "und"):
        code = detect_language(transcript)
    selected = (model or "").strip().lower()
    if selected in ("", "auto", "default"):
        selected = {"fr": "fastconformer-fr-q4", "en": "fastconformer-en-q4",
                    "zh": "wav2vec2-zh-q4"}.get(code, "canary-ctc-multilingual-q4")
    if selected not in MODELS:
        raise ValueError(f"Unknown alignment model: {selected}")
    if selected == "canary-ctc-multilingual-q4" and code not in LANGUAGES:
        raise ValueError(f"Canary CTC does not support {code}; Chinese requires wav2vec2-zh-q4")
    directory, filename = MODELS[selected]
    path = ROOT / directory / filename
    if not path.is_file():
        raise ValueError(f"Model not present in central registry: {selected}")
    return selected, code


def normalized(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in stripped if c.isalnum() and not unicodedata.combining(c))


def attach_standalone_punctuation(text: str) -> str:
    """Punctuation has no acoustic onset; attach it to a spoken word."""
    out = []
    prefix = ""
    for token in text.split():
        if not any(c.isalnum() for c in token):
            if out:
                out[-1] += token
            else:
                prefix += token
        else:
            out.append(prefix + token)
            prefix = ""
    if prefix:
        raise ValueError("Only punctuation found in transcript")
    return " ".join(out)


def acoustic_tokens(text: str) -> list[str]:
    """Keep Latin words intact; use CJK ideograph onsets as acoustic units."""
    result: list[str] = []
    for piece in text.split():
        buf = ""
        for ch in piece:
            if 0x3400 <= ord(ch) <= 0x9FFF:
                if buf:
                    result.append(buf)
                    buf = ""
                result.append(ch)
            elif ch in "，。！？、；：":
                if buf:
                    result.append(buf)
                    buf = ""
                if result:
                    result[-1] += ch
                else:
                    buf += ch
            else:
                buf += ch
        if buf:
            result.append(buf)
    return result


def restore_original_tokens(raw: list[dict], text: str) -> list[dict]:
    """Group only measured CTC spans into input tokens (not time interpolation).

    CrispASR may split hyphenated words; return exactly the expected lexical
    tokens using the first/last acoustic boundaries of their constituent spans.
    """
    expected = acoustic_tokens(text)
    aligned: list[dict] = []
    i = 0
    for token in expected:
        goal = normalized(token)
        if not goal:
            raise ValueError("Punctuation-only token has no acoustic alignment")
        spans = []
        assembled = ""
        while i < len(raw) and len(assembled) < len(goal):
            word = raw[i]
            spans.append(word)
            assembled += normalized(str(word.get("word", "")))
            i += 1
            if not goal.startswith(assembled):
                raise ValueError(f"CTC lexical mismatch near token {len(aligned)}")
        if assembled != goal or not spans:
            raise ValueError(f"CTC omitted or changed token {len(aligned)}")
        start = float(spans[0]["start"])
        end = float(spans[-1]["end"])
        if not (math.isfinite(start) and math.isfinite(end) and end > start):
            raise ValueError("Nonfinite or empty acoustic boundaries")
        aligned.append({"word": token, "start": start, "end": end})
    if i != len(raw):
        raise ValueError("CTC returned extra words")
    return aligned


def run_alignment(wav_path: Path, text: str, language: str,
                  model: str | None, segment_json: str | None) -> dict:
    with wave.open(str(wav_path), "rb") as f:
        if f.getnchannels() != 1 or f.getframerate() != 16000 or f.getsampwidth() != 2:
            raise ValueError("Expected 16kHz mono signed PCM16 WAV")
        duration = f.getnframes() / 16000
        if duration <= 0:
            raise ValueError("Audio is empty")
        original_tokens = len(text.split())
        text = attach_standalone_punctuation(text)
        if segment_json is not None:
            group_rows = json.loads(segment_json)
            for group in group_rows:
                group["text"] = attach_standalone_punctuation(group["text"])
            segment_json = json.dumps(group_rows, ensure_ascii=False)
        chosen, detected = choose_model(model, language, text)
        groups = groups_for_alignment(text, duration, segment_json)
        rows: list[dict] = []
        previous_start = -1.0
        for group_idx, group in enumerate(groups):
            start, end = float(group["start"]), float(group["end"])
            first, last = round(start * 16000), round(end * 16000)
            f.setpos(first)
            samples = f.readframes(last - first)
            if len(samples) < 320:
                raise ValueError(f"Empty audio window {group_idx}")
            directory, filename = MODELS[chosen]
            # All intermediates stay in RAM and are deleted after each request.
            with tempfile.TemporaryDirectory(prefix="cloud9-ctc-", dir="/tmp") as tempdir:
                chunk = Path(tempdir) / "chunk.wav"
                text_file = Path(tempdir) / "reference.txt"
                result_file = Path(tempdir) / "alignment.json"
                with wave.open(str(chunk), "wb") as target:
                    target.setnchannels(1)
                    target.setsampwidth(2)
                    target.setframerate(16000)
                    target.writeframes(samples)
                text_file.write_text(group["text"], encoding="utf8")
                args = [BIN, "--align-only", "--gpu-backend", "vulkan",
                        "-am", str(ROOT / directory / filename),
                        "-f", str(chunk), "--text-file", str(text_file),
                        "--align-format", "json", "--align-granularity", "word",
                        "--align-output", str(result_file)]
                proc = subprocess.run(args, capture_output=True, text=True,
                                      timeout=300, check=False)
                if proc.returncode or not result_file.exists():
                    raise RuntimeError(f"Vulkan CTC alignment failed for chunk {group_idx}")
                parsed = json.loads(result_file.read_text(encoding="utf8"))
                if not isinstance(parsed, list):
                    raise ValueError("CrispASR did not return a word-level JSON array")
                words = restore_original_tokens(parsed, group["text"])
                for word in words:
                    word["start"] = round(word["start"] + start, 6)
                    word["end"] = round(min(duration, word["end"] + start), 6)
                    if word["start"] < previous_start or word["end"] <= word["start"]:
                        raise ValueError("Overlapping/non-monotonic acoustic word bounds")
                    previous_start = word["start"]
                rows.extend(words)
    if len(rows) != len(acoustic_tokens(text)):
        raise ValueError("Full transcript is not acoustically aligned")
    return {"word_segments": rows,
            "quality": {"backend": "vulkan", "method": "crispasr-ctc-forced-alignment",
                        "interpolation": False, "model": chosen, "language": detected,
                        "alignment_chunks": len(groups), "frame_confidence_available": False,
                        "accuracy_certified": False, "audio_duration": duration,
                        "standalone_punctuation_attached": original_tokens - len(text.split())}}


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "vulkan", "models": [
        {"id": name, "available": (ROOT / directory / filename).is_file()}
        for name, (directory, filename) in MODELS.items()]}


@app.get("/models")
async def models():
    return {"default": "auto", "fallback": "canary-ctc-multilingual-q4",
            "language_defaults": {"fr": "fastconformer-fr-q4",
                                  "en": "fastconformer-en-q4",
                                  "zh": "wav2vec2-zh-q4"},
            "models": [{"id": key, "languages":
                        ["fr"] if key.startswith("fastconformer-fr")
                        else ["en"] if key.startswith("fastconformer-en")
                        else ["zh"] if key.startswith("wav2vec2-zh")
                        else sorted(LANGUAGES)}
                       for key in MODELS]}


@app.post("/align")
async def align(file: UploadFile = File(...), transcript: str = Form(...),
                language: str = Form("auto"), model: str | None = Form(None),
                segments_json: str | None = Form(None),
                start: float | None = Form(None), end: float | None = Form(None)):
    if not transcript.strip():
        raise HTTPException(status_code=422, detail="Empty transcript")
    async with gate:
        try:
            with tempfile.TemporaryDirectory(prefix="cloud9-align-request-", dir="/tmp") as directory:
                source = Path(directory) / "input.wav"
                total = 0
                with source.open("wb") as f:
                    while data := await file.read(1024 * 1024):
                        total += len(data)
                        if total > 250 * 1024 * 1024:
                            raise ValueError("Audio upload exceeds 250 MiB")
                        f.write(data)
                if (start is not None or end is not None) and segments_json is None:
                    # Segment bounds are passed by legacy clients; apply them
                    # only to <=38 s audio to avoid guessing long-form placement.
                    with wave.open(str(source), "rb") as f:
                        duration = f.getnframes() / f.getframerate()
                    if duration > 38:
                        raise ValueError("Long audio requires timestamped ASR segments")
                    segments_json = json.dumps([{"start": start or 0.0,
                                                 "end": end or duration,
                                                 "text": transcript}])
                return await asyncio.to_thread(run_alignment, source, transcript,
                                               language, model, segments_json)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

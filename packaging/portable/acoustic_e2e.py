# Copyright (c) 2026 GodsQuantum
"""True frozen-app FastConformer inference on a real French speech fixture.

Run only on native CI runners, after the frozen app is assembled. This is
separate from the faster HTTP/CLI smoke tests and downloads a checksum-pinned
public acoustic model into isolated temporary user data.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

from build import variant

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def encode_multipart(fields: dict[str, str], filename: str, audio: bytes) -> tuple[bytes, str]:
    boundary = f"----SpeachesAcoustic{uuid.uuid4().hex}"
    body = bytearray()
    for key, value in fields.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    body.extend(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
    )
    body.extend(audio)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def validate_alignment(response: dict[str, Any], transcript: str, duration: float) -> int:
    rows = response.get("word_segments")
    if not isinstance(rows, list) or len(rows) != len(transcript.split()):
        raise ValueError("Real CTC word count mismatched reference")
    quality = response.get("quality")
    if not isinstance(quality, dict) or quality.get("backend") != "cpu":
        raise ValueError("Acoustic CPU inference backend was not verified")
    if quality.get("interpolation") is not False or quality.get("method") != "crispasr-ctc-forced-alignment":
        raise ValueError("Results are not measured acoustic CTC alignment")
    previous_end = 0.0
    for i, segment in enumerate(rows):
        if not isinstance(segment, dict) or not segment.get("word"):
            raise ValueError(f"Missing acoustic word {i}")
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid measured acoustic word {i}") from exc
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and start >= previous_end - 0.03
            and end > start
            and end <= duration + 0.08
        ):
            raise ValueError(f"Non-monotonic acoustic word timestamps near {i}")
        previous_end = end
    return len(rows)


def frozen_executable() -> Path:
    target = variant()
    folder = ROOT / "packaging" / "portable" / "dist" / target
    name = "Cloud9-Speaches"
    if target.startswith("macos"):
        executable = folder / f"{name}.app" / "Contents" / "MacOS" / name
    else:
        executable = folder / name / (name + (".exe" if platform.system() == "Windows" else ""))
    if not executable.is_file():
        raise FileNotFoundError(f"Native frozen Speaches executable not found: {executable}")
    return executable


def convert_fixture(output: Path) -> float:
    from scipy.signal import resample_poly
    import soundfile as sf

    source = ROOT / "bench" / "fixtures" / "fr_regression_v1.wav"
    samples, rate = sf.read(source, dtype="float32")
    if samples.ndim != 1 or rate < 16000:
        raise ValueError("Real French fixture must contain mono speech at 16 kHz or higher")
    divisor = math.gcd(rate, 16000)
    resampled = resample_poly(samples, 16000 // divisor, rate // divisor)
    sf.write(output, resampled, 16000, subtype="PCM_16")
    return float(len(resampled) / 16000)


def wait_for_ready(child: subprocess.Popen[str], origin: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Frozen Speaches process exited with {child.returncode}")
        try:
            with urllib.request.urlopen(f"{origin}/health", timeout=3) as health:  # noqa: S310 - local loopback only
                if health.status != 200:
                    continue
            with urllib.request.urlopen(f"{origin}/v1/audio/alignment/health", timeout=3) as response:  # noqa: S310
                result = json.load(response)
            if any(
                m.get("id") == "fastconformer-fr-q4" and m.get("available") is True for m in result.get("models", [])
            ):
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.8)
    raise TimeoutError("Frozen Speaches server and installed FastConformer did not become healthy")


def run_real_acoustics() -> None:
    binary = frozen_executable()
    transcript = (ROOT / "bench" / "fixtures" / "fr_regression_v1.txt").read_text(encoding="utf-8").strip()
    if not transcript:
        raise ValueError("Empty source audio transcript")
    port = 39087
    origin = f"http://127.0.0.1:{port}"
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            raise RuntimeError(f"Refusing to validate other process: {origin} is already in use")
    except ConnectionRefusedError:
        pass

    with tempfile.TemporaryDirectory(prefix="trio-speaches-real-acoustics-") as scratch:
        temporary = Path(scratch)
        isolated_cwd = temporary / "unrelated-working-directory"
        isolated_cwd.mkdir()
        env = os.environ.copy()
        env["CLOUD9_NATIVE_CTC_ENABLED"] = "1"
        env["CLOUD9_CTC_BACKEND"] = "cpu"
        env["HF_HOME"] = str(temporary / "cache" / "huggingface")
        env["ALIGN_MODEL_ROOT"] = str(temporary / "user-data" / "models" / "aligners")
        if platform.system() == "Windows":
            env["LOCALAPPDATA"] = str(temporary / "user-data")
        elif platform.system() == "Darwin":
            env["HOME"] = str(temporary / "user-home")
        else:
            env["XDG_DATA_HOME"] = str(temporary / "user-data")
            env["XDG_CACHE_HOME"] = str(temporary / "cache")

        audio_file = temporary / "authentic-french-speech-16k.wav"
        duration = convert_fixture(audio_file)
        if not 10.0 < duration < 30.0:
            raise ValueError(f"Invalid authentic acoustic test fixture length {duration:.2f}s")
        logger.info("Downloading SHA256-pinned French Q4 GGUF using installed native binary")
        subprocess.run(
            [str(binary), "--install-default-models", "--model-group", "aligner"],
            env=env,
            cwd=isolated_cwd,
            check=True,
            timeout=900,
        )
        log_file = temporary / "server.log"
        with log_file.open("w+", encoding="utf-8") as logs:
            child = subprocess.Popen(
                [str(binary), "--no-browser", "--port", str(port)],
                env=env,
                cwd=isolated_cwd,
                stdout=logs,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_for_ready(child, origin, time.monotonic() + 240)
                audio = audio_file.read_bytes()
                body, content_type = encode_multipart(
                    {"transcript": transcript, "language": "fr", "model": "fastconformer-fr-q4"},
                    audio_file.name,
                    audio,
                )
                request = urllib.request.Request(  # noqa: S310 - constructed only from fixed 127.0.0.1 origin
                    origin + "/v1/audio/alignment/align",
                    data=body,
                    headers={"Content-Type": content_type},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=480) as response:  # noqa: S310 - loopback only
                    if response.status != 200:
                        raise ValueError(f"Authentic CTC HTTP status {response.status}")
                    actual = json.load(response)
                count = validate_alignment(actual, transcript, duration)
                logger.info(
                    "SPEACHES_NATIVE_REAL_ACOUSTIC_PASS os=%s arch=%s words=%d duration=%.2fs model=FastConformer-FR-Q4 backend=cpu",
                    platform.system(),
                    platform.machine(),
                    count,
                    duration,
                )
            except Exception:
                logs.flush()
                logger.exception(
                    "SPEACHES_NATIVE_REAL_ACOUSTIC_FAIL app_log_tail=%s", log_file.read_text(errors="replace")[-1800:]
                )
                raise
            finally:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_real_acoustics()

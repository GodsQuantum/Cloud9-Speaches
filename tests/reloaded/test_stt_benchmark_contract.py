"""CPU-only regression checks for the existing Cloud9-Speaches STT harness."""

import json
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch
import wave

BENCHMARK = Path(__file__).resolve().parents[2] / "scripts/cloud9_fr_stt_benchmark.py"
NS = runpy.run_path(str(BENCHMARK))


def test_custom_gguf_routes_not_replaced() -> None:
    models = NS["DEFAULT_MODELS"]
    assert models[0] == ("fr-fast", "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf")
    assert models[1] == ("fr-quality", "handy-computer/Qwen3-ASR-1.7B-gguf")
    assert models[2] == ("stt-whisper", "deepdml/faster-whisper-large-v3-turbo-ct2")


def test_wer_contract_keeps_french_accents_and_hyphens() -> None:
    normalize = NS["normalize"]
    assert normalize("À Paris, quand-même, vingt-trois.") == [
        "a",
        "paris",
        "quand",
        "meme",
        "vingt",
        "trois",
    ]
    assert NS["wer"]("Bonjour les amis", "Bonjour les amis") == (0.0, 0, 3)


def test_cli_json_contract_without_network_or_model_load() -> None:
    # Isolate all model/network paths; only run the JSON aggregation logic.
    with tempfile.TemporaryDirectory() as work:
        folder = Path(work)
        audio = folder / "clip.wav"
        reference = folder / "clip.txt"
        result = folder / "results.json"
        reference.write_text("Bonjour les amis", encoding="utf-8")
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 16000)
        args = [
            str(BENCHMARK),
            "--base-url",
            "http://127.0.0.1:9999",
            "--audio",
            str(audio),
            "--reference",
            str(reference),
            "--output",
            str(result),
            "--passes",
            "2",
            "--model",
            "fr-fast=handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
        ]
        # run_path creates an isolated namespace: rebind the function's globals.
        globals_ = NS["main"].__globals__
        with (
            patch.object(sys, "argv", args),
            patch.dict(
                globals_,
                {
                    "transcribe": lambda *_: ("Bonjour les amis", 0.25, {"gtt_used": 1024}),
                    "unload": lambda *_: {"rc": 0, "response": "ok"},
                },
            ),
            patch("time.sleep", return_value=None),
        ):
            NS["main"]()
        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["duration_s"] == 1.0
        assert data["models"][0]["model"].startswith("handy-computer/nemotron")
        assert len(data["models"][0]["runs"]) == 2
        assert data["models"][0]["runs"][0]["mode"] == "cold"
        assert data["models"][0]["runs"][1]["mode"] == "warm"
        assert data["models"][0]["runs"][0]["wer"] == 0.0
        assert data["models"][0]["runs"][0]["peak_gtt_bytes"] == 1024
        assert data["models"][0]["unload"] == {"rc": 0, "response": "ok"}

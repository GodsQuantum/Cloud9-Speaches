#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download

DEFAULTS = [
    ("handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf", ["*Q8_0.gguf", "README.md"]),
    ("handy-computer/Qwen3-ASR-1.7B-gguf", ["*Q8_0.gguf", "README.md"]),
    ("deepdml/faster-whisper-large-v3-turbo-ct2", None),
    ("Systran/faster-whisper-small", None),
    ("speaches-ai/Kokoro-82M-v1.0-ONNX-int8", None),
    ("speaches-ai/piper-fr_FR-tom-medium", None),
]

def main():
    parser = argparse.ArgumentParser(description="Populate the Hugging Face cache for offline/plug-and-play use.")
    parser.add_argument("models", nargs="*", help="HF model IDs. No arguments downloads the curated defaults.")
    args = parser.parse_args()
    requested = args.models or [m for m, _ in DEFAULTS]
    patterns = {m: p for m, p in DEFAULTS}
    for model in requested:
        print(f"==> {model}")
        snapshot_download(repo_id=model, allow_patterns=patterns.get(model))

if __name__ == "__main__":
    main()

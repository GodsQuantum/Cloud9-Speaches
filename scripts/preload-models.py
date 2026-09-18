#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download

PATTERNS = {
    "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf": ["*Q8_0.gguf", "README.md"],
    "handy-computer/Qwen3-ASR-0.6B-gguf": ["*Q8_0.gguf", "README.md"],
    "handy-computer/Qwen3-ASR-1.7B-gguf": ["*Q8_0.gguf", "README.md"],
    "handy-computer/parakeet-tdt-0.6b-v3-gguf": ["*Q8_0.gguf", "README.md"],
    "handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf": ["*Q8_0.gguf", "README.md"],
}

def main():
    parser = argparse.ArgumentParser(description="Preload model repositories into Speaches Reloaded.")
    parser.add_argument("models", nargs="+", help="Hugging Face model repository IDs")
    args = parser.parse_args()
    for model in args.models:
        print(f"==> {model}", flush=True)
        snapshot_download(repo_id=model, allow_patterns=PATTERNS.get(model))

if __name__ == "__main__":
    main()

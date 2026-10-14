"""Stable human-readable specialties for already-installed Cloud9 speech models.

Enhances OpenAI-compatible model *listing* only; model IDs and requests stay intact.
These are workload descriptions, not unsourced quality rankings.
"""

from typing import Any

SPECIALTIES: dict[str, str] = {
    "Systran/faster-whisper-large-v3": "ASR multilingue, qualité",
    "deepdml/faster-whisper-large-v3-turbo-ct2": "ASR multilingue, rapide",
    "Systran/faster-whisper-small": "STT léger et économique",
    "handy-computer/Qwen3-ASR-1.7B-gguf": "Transcription FR et multilingue",
    "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf": "Dictée française en streaming",
    "speaches-ai/piper-fr_FR-tom-medium": "TTS voix française Tom",
    "speaches-ai/Kokoro-82M-v1.0-ONNX-int8": "TTS voix naturelle multilingue",
    "handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf": "Diarisation multi-locuteurs",
    "snakers4/silero-vad": "Détection d'activité vocale",
}


def annotate_model_public_dict(model: dict[str, Any]) -> dict[str, Any]:
    """Return copied public metadata; never rename or mutate the original model id."""
    out = model.copy()
    model_id = out.get("id")
    if not isinstance(model_id, str):
        return out
    specialty = SPECIALTIES.get(model_id)
    if specialty is not None:
        base_name = out.get("name") or model_id
        out["name"] = f"{base_name} — {specialty}"
        out["cloud9"] = {"specialty": specialty}
    return out

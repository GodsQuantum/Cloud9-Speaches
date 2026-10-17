"""First-party FastConformer Vulkan alignment UI within Cloud9-Speaches."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

import gradio as gr

from speaches.config import Config
from speaches.ui.utils import http_client_from_gradio_req

ALIGN_MODELS = ["auto", "fastconformer-fr-q4", "fastconformer-en-q4",
                "wav2vec2-zh-q4", "canary-ctc-multilingual-q4"]


def create_alignment_tab(config: Config, api_key_input: gr.Textbox) -> None:
    async def run_ui_align(
        audio_path: str, transcript: str, language: str, model: str,
        segments_json: str, api_key: str, request: gr.Request,
    ) -> dict:
        if not audio_path or not transcript.strip():
            return {"error": "Audio and an exact transcript are required."}
        # The CTC decoder accepts only mono PCM16 WAV at 16 kHz.
        with tempfile.TemporaryDirectory(prefix="speaches-ui-align-") as tmp:
            wav = Path(tmp) / "input.wav"
            process = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-y", "-i", audio_path, "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(wav)],
                capture_output=True, text=True, timeout=90, check=False,
            )
            if process.returncode:
                return {"error": "Failed to prepare a 16-kHz mono PCM16 WAV."}
            data = {"transcript": transcript, "language": language, "model": model}
            if segments_json.strip():
                try:
                    rows = json.loads(segments_json)
                    if not isinstance(rows, list):
                        return {"error": "ASR segments must be a JSON array."}
                except ValueError:
                    return {"error": "Invalid ASR segment JSON."}
                data["segments_json"] = segments_json
            client = http_client_from_gradio_req(request, config, api_key or None)
            try:
                with wav.open("rb") as source:
                    response = await client.post(
                        "/v1/audio/alignment/align",
                        files={"file": ("input.wav", source, "audio/wav")},
                        data=data, timeout=7200,
                    )
                payload = response.json()
                return payload if response.is_success else {
                    "error": f"HTTP {response.status_code}", "detail": payload.get("detail", payload)
                }
            except Exception:
                return {"error": "Local Vulkan alignment request failed."}

    with gr.Tab(label="Acoustic Alignment"):
        gr.Markdown(
            "### FastConformer CTC Vulkan — embedded in Cloud9-Speaches\n"
            "Upload audio and its recognized transcript. Select FR/EN/ZH or a specific CTC model. "
            "Audio over 38 seconds requires timestamped ASR segments; no synthetic timing."
        )
        audio = gr.Audio(label="Audio", type="filepath")
        transcript = gr.Textbox(label="Exact transcript", lines=5)
        with gr.Row():
            language = gr.Dropdown(["fr", "en", "zh", "auto"], value="fr", label="Language")
            model = gr.Dropdown(ALIGN_MODELS, value="auto", label="Acoustic model")
        segments = gr.Textbox(
            label="ASR segments JSON (required for audio >38s)",
            placeholder='[{"start":0,"end":30,"text":"..."}]', lines=3,
        )
        button = gr.Button("Align speech word by word")
        results = gr.JSON(label="Measured timestamps / quality")
        button.click(
            run_ui_align, [audio, transcript, language, model, segments, api_key_input], results,
        )

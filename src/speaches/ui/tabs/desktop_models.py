# Copyright (c) 2026 GodsQuantum
"""Hardware-aware model chooser for native desktop builds only."""

from __future__ import annotations

import os
import shutil
import subprocess

import gradio as gr

# Verified CTranslate2/multilingual model repositories, not arbitrary URLs.
CATALOGUE = (
    ("Systran/faster-whisper-tiny", "Tiny — fastest / limited accuracy"),
    ("Systran/faster-whisper-base", "Base — lightweight"),
    ("Systran/faster-whisper-small", "Small — balanced"),
    ("Systran/faster-whisper-medium", "Medium — accuracy-focused"),
    ("Systran/faster-whisper-large-v3", "Large-v3 — highest accuracy / heavy"),
)


def hardware() -> tuple[float, bool, str, float | None]:
    """Inspect the actual CTranslate2 backend rather than Torch's unrelated MPS/CUDA capability."""
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / 1024**3
    except (ImportError, OSError):
        ram_gb = 0.0
    gpu = False
    gpu_name = "CPU (CTranslate2)"
    vram_gb: float | None = None
    try:
        import ctranslate2

        gpu = ctranslate2.get_cuda_device_count() > 0
        if gpu:
            gpu_name = "NVIDIA CUDA (CTranslate2)"
    except (ImportError, OSError, RuntimeError, AttributeError):
        pass
    if gpu:
        executable = shutil.which("nvidia-smi")
        if executable:
            try:
                response = subprocess.run(
                    [executable, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=3,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                vram_gb = float(response.stdout.strip().splitlines()[0].strip()) / 1024
                gpu_name += f", {vram_gb:.1f} Go VRAM"
            except (ValueError, IndexError, OSError, subprocess.SubprocessError):
                pass
    return ram_gb, gpu, gpu_name, vram_gb


def recommend(ram_gb: float, gpu: bool, goal: str, vram_gb: float | None = None) -> str:
    """Conservative capability heuristic, not a fabricated performance benchmark."""
    if ram_gb and ram_gb < 4:
        return CATALOGUE[0][0]
    if ram_gb and ram_gb < 8:
        return CATALOGUE[1][0] if goal == "Vitesse" else CATALOGUE[2][0]
    if ram_gb and ram_gb < 16:
        return CATALOGUE[2][0] if goal == "Vitesse" else CATALOGUE[3][0]
    if gpu and ram_gb >= 16:
        if goal == "Vitesse":
            return CATALOGUE[2][0]
        # Large-v3 is only auto-suggested when accelerator VRAM is known sufficient.
        return CATALOGUE[4][0] if vram_gb is not None and vram_gb >= 8 else CATALOGUE[3][0]
    return CATALOGUE[1][0] if goal == "Vitesse" else CATALOGUE[3][0]


def create_model_manager_tab() -> None:
    # Do not expose desktop model actions in container/network deployments.
    if os.getenv("CLOUD9_PORTABLE_MODE") != "1":
        return
    ram_gb, gpu, gpu_name, vram_gb = hardware()
    count = os.cpu_count() or 1
    with gr.Tab("Modèles à télécharger"):
        gr.Markdown("## Modèles de transcription locaux")
        gr.Markdown(
            f"**Matériel détecté :** {count} threads CPU · {ram_gb:.1f} Go RAM"
            if ram_gb
            else f"**Matériel détecté :** {count} threads CPU · RAM inconnue"
        )
        gr.Markdown(
            f"**Accélérateur :** {gpu_name}. La recommandation est indicative ; "
            "les performances réelles dépendent du backend et du matériel."
        )
        goal = gr.Radio(["Vitesse", "Précision"], value="Vitesse", label="Priorité")
        model = gr.Dropdown(
            choices=[(label, repo) for repo, label in CATALOGUE],
            value=recommend(ram_gb, gpu, "Vitesse", vram_gb),
            label="Modèle à installer",
        )
        advisory = gr.Markdown("Conseil : commencer avec le modèle proposé, puis comparer sur un audio réel.")

        def update(choice: str) -> tuple[object, str]:
            selected = recommend(ram_gb, gpu, choice, vram_gb)
            return gr.update(value=selected), f"Suggestion **{choice.lower()}** pour ce matériel : `{selected}`."

        goal.change(update, inputs=goal, outputs=[model, advisory])
        button = gr.Button("Télécharger le modèle sélectionné", variant="primary")
        result = gr.Textbox(label="État / téléchargement", interactive=False)

        def install(repo_id: str) -> str:
            if repo_id not in dict(CATALOGUE):
                raise gr.Error("Modèle non autorisé dans le catalogue")
            # The catalog contains only verified CTranslate2 Whisper repos.
            # Calling ExecutorRegistry.download_model_by_id would enumerate
            # every remote model across all engines on each click, which can
            # be slow and fail even when this exact public repo is available.
            from speaches.executors.whisper import whisper_model_registry

            try:
                downloaded = whisper_model_registry.download_model_files_if_not_exist(repo_id)
            except Exception as exc:
                # UI should report the error without logging private tokens.
                raise gr.Error(f"Téléchargement impossible : {type(exc).__name__}") from exc
            return f"{repo_id} : {'téléchargement terminé' if downloaded else 'déjà installé'}"

        button.click(install, inputs=model, outputs=result)
        gr.Markdown(
            "Téléchargements depuis Hugging Face, stockés dans le cache local utilisateur. "
            "Le modèle doit être sélectionné séparément dans l'onglet transcription. "
            "Ce catalogue est multilingue ; vérifier la précision sur vos enregistrements."
        )

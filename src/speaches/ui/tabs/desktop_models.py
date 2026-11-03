# Copyright (c) 2026 GodsQuantum
"""Hardware-aware model chooser for native desktop builds only."""

from __future__ import annotations

import os
import platform

import gradio as gr

# Verified CTranslate2/multilingual model repositories, not arbitrary URLs.
CATALOGUE = (
    ("Systran/faster-whisper-tiny", "Tiny — fastest / limited accuracy"),
    ("Systran/faster-whisper-base", "Base — lightweight"),
    ("Systran/faster-whisper-small", "Small — balanced"),
    ("Systran/faster-whisper-medium", "Medium — accuracy-focused"),
    ("Systran/faster-whisper-large-v3", "Large-v3 — highest accuracy / heavy"),
)


def hardware() -> tuple[float, bool, str]:
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / 1024**3
    except (ImportError, OSError):
        ram_gb = 0.0
    gpu = False
    gpu_name = "CPU"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = True
            gpu_name = torch.cuda.get_device_name(0)
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            gpu = True
            gpu_name = "Apple Metal"
    except (ImportError, RuntimeError, AttributeError):
        pass
    return ram_gb, gpu, gpu_name


def recommend(ram_gb: float, gpu: bool, goal: str) -> str:
    """Conservative capability heuristic, not a fabricated performance benchmark."""
    if ram_gb and ram_gb < 4:
        return CATALOGUE[0][0]
    if ram_gb and ram_gb < 8:
        return CATALOGUE[1][0] if goal == "Vitesse" else CATALOGUE[2][0]
    if ram_gb and ram_gb < 16:
        return CATALOGUE[2][0] if goal == "Vitesse" else CATALOGUE[3][0]
    if gpu and ram_gb >= 16:
        return CATALOGUE[2][0] if goal == "Vitesse" else CATALOGUE[4][0]
    return CATALOGUE[1][0] if goal == "Vitesse" else CATALOGUE[3][0]


def create_model_manager_tab() -> None:
    # Do not expose desktop model actions in container/network deployments.
    if os.getenv("CLOUD9_PORTABLE_MODE") != "1":
        return
    ram_gb, gpu, gpu_name = hardware()
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
            value=recommend(ram_gb, gpu, "Vitesse"),
            label="Modèle à installer",
        )
        advisory = gr.Markdown("Conseil : commencer avec le modèle proposé, puis comparer sur un audio réel.")

        def update(choice: str) -> tuple[object, str]:
            selected = recommend(ram_gb, gpu, choice)
            return gr.update(value=selected), f"Suggestion **{choice.lower()}** pour ce matériel : `{selected}`."

        goal.change(update, inputs=goal, outputs=[model, advisory])
        button = gr.Button("Télécharger le modèle sélectionné", variant="primary")
        result = gr.Textbox(label="État / téléchargement", interactive=False)

        def install(repo_id: str) -> str:
            if repo_id not in dict(CATALOGUE):
                raise gr.Error("Modèle non autorisé dans le catalogue")
            from speaches.dependencies import get_executor_registry

            registry = get_executor_registry()
            try:
                downloaded = registry.download_model_by_id(repo_id)
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

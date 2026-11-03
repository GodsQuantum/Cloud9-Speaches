# Copyright (c) 2026 GodsQuantum
"""Hardware-aware model chooser for native desktop builds only."""

from __future__ import annotations

import os
from pathlib import Path
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

        gr.Markdown("## Couplage direct avec ScribeWatch")
        gr.Markdown(
            "ScribeWatch demande actuellement **Whisper Large-v3 Turbo GGUF Q6_K (~693 Mo)** et "
            "**FastConformer français Q4_K (~70 Mo)** pour vérifier la transcription et alignement. "
            "Ce pack est facultatif si vous utilisez Speaches seul. Une connexion Internet est nécessaire "
            "uniquement pour le téléchargement initial. Vérifiez la vitesse réelle sur votre CPU."
        )
        bundle = gr.Button("Installer le pack ScribeWatch — transcription + alignement")
        bundle_status = gr.Textbox(label="Installation du pack ScribeWatch", interactive=False)

        def install_scribewatch_bundle() -> str:
            # Reuse pinned, atomic, SHA-256-verified native bootstrap; no
            # duplicated model URLs, no unpinned GGUF downloads.
            from default_models import install_defaults

            align_root = Path(os.environ["ALIGN_MODEL_ROOT"])
            hf_home = Path(os.environ["HF_HOME"])
            try:
                install_defaults(align_root, hf_home, only="both")
            except Exception as exc:
                raise gr.Error(f"Installation du pack impossible : {type(exc).__name__}") from exc
            return "Pack ScribeWatch installé et SHA-256 vérifié. Dans ScribeWatch > Providers, cliquez sur Connect Cloud9-Speaches (http://127.0.0.1:8005)."

        bundle.click(install_scribewatch_bundle, outputs=bundle_status)

        from speaches.ui.custom_asr import GGUF_MODELS, install_hf_gguf, list_hf_gguf

        gr.Markdown("## ASR GGUF : Parakeet Ultra, Qwen3, Nemotron")
        gr.Markdown("Sélectionnez un modèle transcribe.cpp compatible. Les GGUF llama.cpp ne conviennent pas.")
        curated = gr.Dropdown(
            choices=[(name, repo) for name, repo in GGUF_MODELS],
            value=GGUF_MODELS[0][1],
            label="Catalogue de modèles vocaux",
        )
        source = gr.Textbox(
            value=GGUF_MODELS[0][1],
            label="Dépôt Hugging Face ou URL personnalisée",
            placeholder="organisation/modele ou https://huggingface.co/organisation/modele",
        )
        curated.change(lambda repo: repo, inputs=curated, outputs=source)
        variants = gr.Dropdown(choices=[], label="Quantification / fichier GGUF", interactive=True)
        custom_status = gr.Textbox(label="Statut modèle GGUF", interactive=False)

        def discover(source_value: str) -> tuple[object, str]:
            try:
                candidates = list_hf_gguf(source_value)
            except Exception as exc:
                raise gr.Error(f"Lecture du dépôt impossible : {type(exc).__name__}") from exc
            if not candidates:
                raise gr.Error("Aucun fichier GGUF pris en charge dans le dépôt.")
            return gr.update(
                choices=candidates, value=candidates[0][1]
            ), f"{len(candidates)} quantification(s) trouvée(s)."

        gr.Button("Afficher les quantifications").click(discover, inputs=source, outputs=[variants, custom_status])

        def install_selected(source_value: str, filename: str) -> str:
            try:
                return install_hf_gguf(source_value, filename)
            except Exception as exc:
                raise gr.Error(f"Téléchargement impossible : {type(exc).__name__}") from exc

        gr.Button("Installer ce modèle GGUF", variant="primary").click(
            install_selected, inputs=[source, variants], outputs=custom_status
        )
        gr.Markdown(
            "Le modèle doit ensuite être sélectionné dans Transcription et testé avec un audio fictif. "
            "Import ne garantit pas compatibilité ni précision."
        )

        from speaches.ui.custom_asr import install_github_gguf

        with gr.Accordion("Fichier GGUF externe : GitHub Releases", open=False):
            gr.Markdown(
                "Pour les GGUF ASR transcribe.cpp seulement. Source HTTPS publique GitHub Releases, "
                "SHA-256 publié par l'éditeur obligatoire. Le fichier est limité à 12 Gio."
            )
            external_source = gr.Textbox(
                label="URL du fichier GGUF",
                placeholder="https://github.com/org/repo/releases/download/tag/model-Q8_0.gguf",
            )
            external_hash = gr.Textbox(label="SHA-256 du modèle", type="password")
            external_status = gr.Textbox(label="État de l'import externe", interactive=False)

            def install_external(url: str, digest: str) -> str:
                try:
                    return install_github_gguf(url, digest)
                except Exception as exc:
                    raise gr.Error(f"Import externe refusé : {type(exc).__name__}") from exc

            gr.Button("Importer le modèle externe").click(
                install_external, inputs=[external_source, external_hash], outputs=external_status
            )


# Extend the desktop-only model-manager UI without enabling it in containers.

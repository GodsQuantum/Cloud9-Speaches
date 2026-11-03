# Copyright (c) 2026 GodsQuantum
from __future__ import annotations

from speaches.ui.tabs.desktop_models import CATALOGUE, recommend


def test_recommendations_are_bounded_to_verified_catalogue() -> None:
    valid = {repo for repo, _ in CATALOGUE}
    for ram in (0, 2, 4, 8, 16, 64):
        for gpu in (True, False):
            for goal in ("Vitesse", "Précision"):
                assert recommend(ram, gpu, goal) in valid


def test_more_ram_does_not_force_heavy_model_for_speed() -> None:
    assert recommend(32, gpu=True, goal="Vitesse") == "Systran/faster-whisper-small"


def test_model_install_uses_direct_native_registry_not_remote_catalog_scan() -> None:
    from pathlib import Path

    code = (Path(__file__).resolve().parents[2] / "src/speaches/ui/tabs/desktop_models.py").read_text()
    assert "whisper_model_registry.download_model_files_if_not_exist(repo_id)" in code
    assert "registry.download_model_by_id(repo_id)" not in code


def test_unknown_or_low_gpu_vram_never_auto_recommends_large_v3() -> None:
    assert recommend(32, gpu=True, goal="Précision") == "Systran/faster-whisper-medium"
    assert recommend(32, gpu=True, goal="Précision", vram_gb=4) == "Systran/faster-whisper-medium"
    assert recommend(32, gpu=True, goal="Précision", vram_gb=8) == "Systran/faster-whisper-large-v3"


def test_scribewatch_bundle_is_single_verified_native_path() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "src/speaches/ui/tabs/desktop_models.py").read_text()
    assert "from default_models import install_defaults" in source
    assert 'install_defaults(align_root, hf_home, only="both")' in source
    assert "Installer le pack ScribeWatch" in source
    assert "FastConformer" in source

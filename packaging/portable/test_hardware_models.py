from __future__ import annotations
from speaches.ui.tabs.desktop_models import CATALOGUE, recommend


def test_recommendations_are_bounded_to_verified_catalogue():
    valid = {repo for repo, _ in CATALOGUE}
    for ram in (0, 2, 4, 8, 16, 64):
        for gpu in (True, False):
            for goal in ("Vitesse", "Précision"):
                assert recommend(ram, gpu, goal) in valid


def test_more_ram_does_not_force_heavy_model_for_speed():
    assert recommend(32, True, "Vitesse") == "Systran/faster-whisper-small"

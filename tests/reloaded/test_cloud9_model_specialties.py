"""Offline contract for descriptive model labels without altering OpenAI model IDs."""

import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/speaches/cloud9_model_specialty.py"
ROUTES = ROOT / "src/speaches/routers/models.py"
spec = importlib.util.spec_from_file_location("cloud9_model_specialty", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SpecialtyContract(unittest.TestCase):
    def test_all_nine_known_models_have_a_specialty(self) -> None:
        assert len(module.SPECIALTIES) == 9
        for model_id, specialty in module.SPECIALTIES.items():
            with self.subTest(model=model_id):
                value = {"id": model_id, "object": "model", "task": "test"}
                original = json.loads(json.dumps(value))
                result = module.annotate_model_public_dict(value)
                assert value == original
                assert result["id"] == model_id
                assert result["object"] == "model"
                assert result["task"] == "test"
                assert specialty in result["name"]
                assert result["cloud9"]["specialty"] == specialty

    def test_unknown_models_are_untouched(self) -> None:
        value = {"id": "arbitrary/other-model", "name": "Custom model"}
        assert module.annotate_model_public_dict(value) == value
        assert module.annotate_model_public_dict({}) == {}

    def test_route_decorates_all_three_local_listing_forms(self) -> None:
        text = ROUTES.read_text(encoding="utf-8")
        assert text.count("annotate_model_public_dict(model.model_dump())") == 3
        assert "from speaches.cloud9_model_specialty import annotate_model_public_dict" in text


if __name__ == "__main__":
    unittest.main()

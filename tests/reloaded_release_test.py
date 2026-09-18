import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class ReloadedReleaseContract(unittest.TestCase):
    def test_brand_and_translations_exist(self):
        for path in ("README.md", "README.fr.md", "README.zh-CN.md"):
            self.assertIn("Speaches Reloaded", read(path))

    def test_logo_is_a_real_svg(self):
        logo = read("docs/assets/speaches-reloaded-logo.svg")
        self.assertTrue(logo.lstrip().startswith("<svg"))
        self.assertGreater(len(logo), 500)

    def test_ci_tracks_main_and_does_not_publish_upstream_cname(self):
        for path in (".github/workflows/test.yaml", ".github/workflows/lint.yaml"):
            text = read(path)
            self.assertIn("- main", text)
            self.assertNotIn("- master", text)
        self.assertFalse((ROOT / "docs/CNAME").exists())

    def test_cpu_compose_uses_published_cpu_image(self):
        self.assertIn(
            "ghcr.io/godsquantum/speaches-reloaded:latest-cpu",
            read("compose.cpu.yaml"),
        )

    def test_vulkan_compose_exposes_dri_and_allows_cpu_override(self):
        text = read("compose.vulkan.yaml")
        self.assertIn("ghcr.io/godsquantum/speaches-reloaded:latest-vulkan", text)
        self.assertIn("/dev/dri", text)
        self.assertIn('TRANSCRIBE_BACKEND: "${TRANSCRIBE_BACKEND:-vulkan}"', text)
        self.assertIn('WHISPER__INFERENCE_DEVICE: "${WHISPER_DEVICE:-cpu}"', text)

    def test_cuda_compose_reserves_nvidia_gpu_and_allows_cpu_override(self):
        text = read("compose.cuda.yaml")
        self.assertIn("ghcr.io/godsquantum/speaches-reloaded:latest-cuda", text)
        self.assertIn("driver: nvidia", text)
        self.assertIn("capabilities: [gpu]", text)
        self.assertIn('TRANSCRIBE_BACKEND: "${TRANSCRIBE_BACKEND:-cuda}"', text)
        self.assertIn('WHISPER__INFERENCE_DEVICE: "${WHISPER_DEVICE:-cuda}"', text)

    def test_dockerfile_supports_three_backends(self):
        text = read("Dockerfile")
        self.assertIn("ARG TRANSCRIBE_BACKEND", text)
        self.assertIn("spirv-headers", text)
        for backend in ("cpu", "vulkan", "cuda"):
            self.assertIn(backend, text)
        self.assertFalse((ROOT / "Dockerfile.reloaded").exists())

    def test_release_workflow_builds_three_variants(self):
        text = read(".github/workflows/release-images.yml")
        for variant in ("cpu", "vulkan", "cuda"):
            self.assertIn(variant, text)
        self.assertIn("docker/build-push-action", text)

    def test_model_setup_helper_exists(self):
        text = read("scripts/models.sh")
        self.assertIn("recommended", text.lower())
        self.assertTrue("huggingface" in text.lower() or "model" in text.lower())

    def test_ui_points_to_reloaded_documentation(self):
        text = read("src/speaches/ui/app.py")
        self.assertIn("GodsQuantum/Speaches-Reloaded", text)
        self.assertNotIn("https://speaches.ai", text)


if __name__ == "__main__":
    unittest.main()

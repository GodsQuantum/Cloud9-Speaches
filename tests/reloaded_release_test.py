import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class ReloadedReleaseContract(unittest.TestCase):
    def test_brand_and_translations_exist(self):
        for path in ("README.md", "README.fr.md", "README.zh-CN.md"):
            self.assertIn("Speaches Reloaded", read(path))

    def test_cpu_compose_uses_published_cpu_image(self):
        self.assertIn(
            "ghcr.io/godsquantum/speaches-reloaded:latest-cpu",
            read("compose.cpu.yaml"),
        )

    def test_vulkan_compose_exposes_dri(self):
        text = read("compose.vulkan.yaml")
        self.assertIn("ghcr.io/godsquantum/speaches-reloaded:latest-vulkan", text)
        self.assertIn("/dev/dri", text)

    def test_cuda_compose_reserves_nvidia_gpu(self):
        text = read("compose.cuda.yaml")
        self.assertIn("ghcr.io/godsquantum/speaches-reloaded:latest-cuda", text)
        self.assertIn("driver: nvidia", text)
        self.assertIn("capabilities: [gpu]", text)

    def test_reloaded_dockerfile_supports_three_backends(self):
        text = read("Dockerfile.reloaded")
        self.assertIn("ARG TRANSCRIBE_BACKEND", text)
        self.assertIn("spirv-headers", text)
        for backend in ("cpu", "vulkan", "cuda"):
            self.assertIn(backend, text)

    def test_release_workflow_builds_three_variants(self):
        text = read(".github/workflows/release-images.yml")
        for variant in ("cpu", "vulkan", "cuda"):
            self.assertIn(variant, text)
        self.assertIn("docker/build-push-action", text)

    def test_model_setup_helper_exists(self):
        text = read("scripts/models.sh")
        self.assertIn("recommended", text.lower())
        self.assertTrue("huggingface" in text.lower() or "model" in text.lower())


if __name__ == "__main__":
    unittest.main()

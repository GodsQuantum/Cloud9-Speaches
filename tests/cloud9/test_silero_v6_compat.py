from pathlib import Path
import tempfile

from speaches.executors import silero_vad_v5 as vad

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    (root / "silero_vad_v6.onnx").write_bytes(b"x")
    vad.get_assets_path = lambda: str(root)
    files = vad.silero_vad_model_registry.get_model_files(vad.MODEL_ID)
    assert getattr(files, "model", None) == root / "silero_vad_v6.onnx", files
print("SILERO_V6_COMPAT=PASS")

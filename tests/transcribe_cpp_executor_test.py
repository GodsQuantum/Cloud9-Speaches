from speaches.config import Config
from speaches.executors.shared.registry import ExecutorRegistry


def test_executor_registry_exposes_transcribe_cpp_for_transcription() -> None:
    registry = ExecutorRegistry(Config())
    names = [executor.name for executor in registry.transcription]

    assert "transcribe.cpp" in names


def test_transcribe_backend_is_configurable_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "cuda")
    config = Config()
    registry = ExecutorRegistry(config)

    assert config.transcribe_backend == "cuda"
    assert registry._transcribe_cpp_executor.model_manager.backend == "cuda"
    assert registry._transcribe_cpp_diarization_executor.model_manager.backend == "cuda"


def test_transcribe_registry_prefers_q8_0(monkeypatch, tmp_path) -> None:
    from speaches.executors import transcribe_cpp

    f16 = tmp_path / "model-F16.gguf"
    q8 = tmp_path / "model-Q8_0.gguf"
    f16.touch()
    q8.touch()
    monkeypatch.setattr(
        transcribe_cpp,
        "list_model_files",
        lambda _model_id, _pattern="**/*": iter([f16, q8]),
        raising=False,
    )

    selected = transcribe_cpp.transcribe_cpp_model_registry.get_model_files("owner/model")

    assert selected == q8


def test_transcribe_manager_loads_selected_gguf_with_vulkan(monkeypatch, tmp_path) -> None:
    import sys
    from types import SimpleNamespace

    from speaches.executors import transcribe_cpp

    model_path = tmp_path / "model-Q8_0.gguf"
    model_path.touch()
    calls = []

    class FakeModel:
        def __init__(self, path, *, backend):
            calls.append((path, backend))

    monkeypatch.setitem(sys.modules, "transcribe_cpp", SimpleNamespace(Model=FakeModel))
    monkeypatch.setattr(transcribe_cpp.transcribe_cpp_model_registry, "get_model_files", lambda _id: model_path)

    manager = transcribe_cpp.TranscribeCppModelManager(ttl=120, backend="vulkan")
    loaded = manager._load_fn("owner/model")

    assert isinstance(loaded, FakeModel)
    assert calls == [(str(model_path), "vulkan")]


def test_transcribe_result_maps_to_openai_verbose_json() -> None:
    from types import SimpleNamespace

    from speaches.executors import transcribe_cpp

    result = SimpleNamespace(
        text="Bonjour le monde",
        language="fr-FR",
        segments=(SimpleNamespace(text="Bonjour le monde", t0_ms=100, t1_ms=900),),
        words=(
            SimpleNamespace(text="Bonjour", t0_ms=100, t1_ms=450),
            SimpleNamespace(text="le", t0_ms=460, t1_ms=550),
            SimpleNamespace(text="monde", t0_ms=560, t1_ms=900),
        ),
    )

    response = transcribe_cpp.transcribe_result_to_response(result, "verbose_json", duration=1.0)

    assert response.text == "Bonjour le monde"
    assert response.language == "fr-FR"
    assert response.segments[0].start == 0.1
    assert response.segments[0].end == 0.9
    assert [word.word for word in response.words] == ["Bonjour", "le", "monde"]


def test_language_hint_maps_fr_for_nemotron_and_disables_for_qwen() -> None:
    from types import SimpleNamespace

    from speaches.executors import transcribe_cpp

    caps = SimpleNamespace(languages=("en-US", "fr-FR", "de-DE"))
    nemotron = SimpleNamespace(arch="parakeet", capabilities=caps)
    qwen = SimpleNamespace(arch="qwen3_asr", capabilities=caps)

    assert transcribe_cpp.resolve_language_hint(nemotron, "fr") == "fr-FR"
    assert transcribe_cpp.resolve_language_hint(nemotron, "fr-FR") == "fr-FR"
    assert transcribe_cpp.resolve_language_hint(qwen, "fr") is None
    assert transcribe_cpp.resolve_language_hint(nemotron, None) is None


def test_transcribe_manager_runs_model_and_returns_verbose_json(monkeypatch) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np

    from speaches.audio import Audio
    from speaches.executors import transcribe_cpp
    from speaches.executors.shared.handler_protocol import TranscriptionRequest
    from speaches.executors.silero_vad_v5 import VadOptions

    calls = []
    result = SimpleNamespace(text="Bonjour", language="fr-FR", segments=(), words=())

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run(self, pcm, **kwargs):
            calls.append((len(pcm), kwargs))
            return result

    fake_model = SimpleNamespace(
        arch="parakeet",
        capabilities=SimpleNamespace(languages=("fr-FR",)),
        session=lambda: FakeSession(),
    )
    manager = transcribe_cpp.TranscribeCppModelManager(ttl=120)
    monkeypatch.setattr(manager, "load_model", lambda _id: nullcontext(fake_model))
    request = TranscriptionRequest(
        audio=Audio(np.zeros(16000, dtype=np.float32), sample_rate=16000),
        model="owner/model",
        stream=False,
        language="fr",
        response_format="verbose_json",
        temperature=0.0,
        timestamp_granularities=["word", "segment"],
        speech_segments=[],
        vad_options=VadOptions(),
        without_timestamps=False,
    )

    response = manager.handle_non_streaming_transcription_request(request)

    assert response.text == "Bonjour"
    assert calls[0][0] == 16000
    assert calls[0][1]["language"] == "fr-FR"
    assert calls[0][1]["timestamps"] == "word"


def test_verbose_json_can_use_resolved_language_when_runtime_omits_it() -> None:
    from types import SimpleNamespace

    from speaches.executors import transcribe_cpp

    result = SimpleNamespace(text="Bonjour", language=None, segments=(), words=())
    response = transcribe_cpp.transcribe_result_to_response(result, "verbose_json", duration=1.0, language="fr-FR")
    assert response.language == "fr-FR"


def test_qwen_long_audio_is_chunked_and_offsets_are_restored(monkeypatch) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np

    from speaches.audio import Audio
    from speaches.executors import transcribe_cpp
    from speaches.executors.shared.handler_protocol import TranscriptionRequest
    from speaches.executors.silero_vad_v5 import VadOptions

    calls = []
    outputs = ["A", "B", "C"]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run_batch(self, pcms, **kwargs):
            calls.append(([len(pcm) for pcm in pcms], kwargs))
            return [
                SimpleNamespace(
                    text=text,
                    language=None,
                    segments=(SimpleNamespace(text=text, t0_ms=0, t1_ms=1000),),
                    words=(SimpleNamespace(text=text, t0_ms=0, t1_ms=500),),
                )
                for text in outputs
            ]

    fake_model = SimpleNamespace(
        arch="qwen3_asr",
        capabilities=SimpleNamespace(languages=()),
        session=lambda: FakeSession(),
    )
    manager = transcribe_cpp.TranscribeCppModelManager(ttl=120)
    monkeypatch.setattr(manager, "load_model", lambda _id: nullcontext(fake_model))
    request = TranscriptionRequest(
        audio=Audio(np.zeros(65 * 10, dtype=np.float32), sample_rate=10),
        model="owner/qwen",
        stream=False,
        language="fr",
        response_format="verbose_json",
        temperature=0.0,
        timestamp_granularities=["word", "segment"],
        speech_segments=[],
        vad_options=VadOptions(),
        without_timestamps=False,
    )
    response = manager.handle_non_streaming_transcription_request(request)

    assert calls == [([300, 300, 50], {"language": None, "timestamps": "none"})]
    assert response.text == "A B C"
    assert [segment.start for segment in response.segments] == [0.0, 30.0, 60.0]
    assert [word.start for word in response.words] == [0.0, 30.0, 60.0]
    assert response.language == "fr"


def test_timestamp_mode_disables_native_timestamps_for_qwen() -> None:
    from types import SimpleNamespace

    from speaches.executors import transcribe_cpp

    qwen = SimpleNamespace(arch="qwen3_asr")
    nemotron = SimpleNamespace(arch="parakeet")
    assert transcribe_cpp.resolve_timestamp_mode(qwen, ["word", "segment"]) == "none"
    assert transcribe_cpp.resolve_timestamp_mode(nemotron, ["word", "segment"]) == "word"
    assert transcribe_cpp.resolve_timestamp_mode(nemotron, ["segment"]) == "segment"


def test_streaming_uses_native_stream_and_emits_committed_deltas(monkeypatch) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np

    from speaches.audio import Audio
    from speaches.executors import transcribe_cpp
    from speaches.executors.shared.handler_protocol import TranscriptionRequest
    from speaches.executors.silero_vad_v5 import VadOptions

    feeds = []
    stream_kwargs = {}

    class FakeStream:
        def __init__(self):
            self.step = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def feed(self, pcm):
            feeds.append(len(pcm))
            self.step += 1
            return SimpleNamespace(committed_changed=True)

        def finalize(self):
            self.step = 3
            return SimpleNamespace(committed_changed=True)

        def text(self):
            committed = {1: "Bon", 2: "Bonjour ", 3: "Bonjour monde"}[self.step]
            return SimpleNamespace(committed=committed, tentative="", display=committed)

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def stream(self, **kwargs):
            stream_kwargs.update(kwargs)
            return FakeStream()

    fake_model = SimpleNamespace(
        arch="parakeet",
        capabilities=SimpleNamespace(languages=("fr-FR",), supports_streaming=True),
        session=lambda: FakeSession(),
    )
    manager = transcribe_cpp.TranscribeCppModelManager(ttl=120)
    monkeypatch.setattr(manager, "load_model", lambda _id: nullcontext(fake_model))
    request = TranscriptionRequest(
        audio=Audio(np.zeros(40000, dtype=np.float32), sample_rate=16000),
        model="owner/nemotron",
        stream=True,
        language="fr",
        response_format="json",
        temperature=0.0,
        timestamp_granularities=["segment"],
        speech_segments=[],
        vad_options=VadOptions(),
        without_timestamps=True,
    )
    events = list(manager.handle_streaming_transcription_request(request))

    assert feeds == [17920, 17920, 4160]
    assert [event.delta for event in events[:-1]] == ["Bon", "jour ", "monde"]
    assert events[-1].text == "Bonjour monde"
    assert stream_kwargs["language"] == "fr-FR"
    assert stream_kwargs["timestamps"] == "segment"
    assert stream_kwargs["family"].att_context_right == 6


def test_executor_registry_exposes_transcribe_cpp_diarization() -> None:
    registry = ExecutorRegistry(Config())
    names = [executor.name for executor in registry.diarization]

    assert "transcribe.cpp-diarization" in names


def test_sortformer_manager_loads_q8_with_vulkan(monkeypatch, tmp_path) -> None:
    import sys
    from types import SimpleNamespace

    from speaches.executors import transcribe_cpp

    model_path = tmp_path / "sortformer-Q8_0.gguf"
    model_path.touch()
    calls = []

    class FakeModel:
        def __init__(self, path, *, backend):
            calls.append((path, backend))

    monkeypatch.setitem(sys.modules, "transcribe_cpp", SimpleNamespace(Model=FakeModel))
    monkeypatch.setattr(
        transcribe_cpp.transcribe_cpp_diarization_model_registry,
        "get_model_files",
        lambda _id: model_path,
    )
    manager = transcribe_cpp.TranscribeCppDiarizationModelManager(ttl=120)
    loaded = manager._load_fn(transcribe_cpp.SORTFORMER_MODEL_ID)
    assert isinstance(loaded, FakeModel)
    assert calls == [(str(model_path), "vulkan")]

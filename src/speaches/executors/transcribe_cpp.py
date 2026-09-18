from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import huggingface_hub

from speaches.api_types import Model, TimestampGranularities
from speaches.audio import Audio
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
)
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry

TASK_NAME = "automatic-speech-recognition"
DIARIZATION_TASK_NAME = "speaker-diarization"
SORTFORMER_MODEL_ID = "handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf"

hf_model_filter = HfModelFilter(
    library_name="transcribe.cpp",
    task=TASK_NAME,
)
diarization_hf_model_filter = HfModelFilter(model_name=SORTFORMER_MODEL_ID)


class TranscribeCppModelRegistry(ModelRegistry[Model, Path]):
    def list_remote_models(self) -> Generator[Model]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            if model.created_at is None or model.card_data is None:
                continue
            yield Model(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME,
            )

    def list_local_models(self) -> Generator[Model]:
        for cached_repo in get_cached_model_repos_info():
            card_data = get_model_card_data_from_cached_repo_info(cached_repo)
            if card_data is None:
                continue
            if not self.hf_model_filter.passes_filter(cached_repo.repo_id, card_data):
                continue
            yield Model(
                id=cached_repo.repo_id,
                created=int(cached_repo.last_modified),
                owned_by=cached_repo.repo_id.split("/")[0],
                language=extract_language_list(card_data),
                task=TASK_NAME,
            )

    def get_model_files(self, model_id: str) -> Path:
        files = [path for path in list_model_files(model_id, "**/*.gguf") if path.is_file()]
        for suffix in ("-Q8_0.gguf", "-F16.gguf", "-BF16.gguf", ".gguf"):
            match = next((path for path in files if path.name.endswith(suffix)), None)
            if match is not None:
                return match
        raise FileNotFoundError(f"No GGUF model file found for '{model_id}'")

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(
            repo_id=model_id,
            repo_type="model",
            allow_patterns=["*Q8_0.gguf", "README.md"],
        )


transcribe_cpp_model_registry = TranscribeCppModelRegistry(hf_model_filter=hf_model_filter)


class TranscribeCppDiarizationModelRegistry(ModelRegistry[Model, Path]):
    def list_remote_models(self) -> Generator[Model]:
        yield Model(
            id=SORTFORMER_MODEL_ID,
            created=0,
            owned_by="handy-computer",
            task=DIARIZATION_TASK_NAME,
        )

    def list_local_models(self) -> Generator[Model]:
        for cached_repo in get_cached_model_repos_info():
            if cached_repo.repo_id == SORTFORMER_MODEL_ID:
                yield Model(
                    id=SORTFORMER_MODEL_ID,
                    created=int(cached_repo.last_modified),
                    owned_by="handy-computer",
                    task=DIARIZATION_TASK_NAME,
                )

    def get_model_files(self, model_id: str) -> Path:
        files = [path for path in list_model_files(model_id, "**/*.gguf") if path.is_file()]
        match = next((path for path in files if path.name.endswith("-Q8_0.gguf")), None)
        if match is None:
            raise FileNotFoundError(f"No Q8_0 GGUF model file found for '{model_id}'")
        return match

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(
            repo_id=model_id,
            repo_type="model",
            allow_patterns=["*Q8_0.gguf", "README.md"],
        )


transcribe_cpp_diarization_model_registry = TranscribeCppDiarizationModelRegistry(
    hf_model_filter=diarization_hf_model_filter
)


def _offset_result(result: Any, offset_ms: int) -> SimpleNamespace:
    segments = tuple(
        SimpleNamespace(text=seg.text, t0_ms=seg.t0_ms + offset_ms, t1_ms=seg.t1_ms + offset_ms)
        for seg in getattr(result, "segments", ())
    )
    words = tuple(
        SimpleNamespace(text=word.text, t0_ms=word.t0_ms + offset_ms, t1_ms=word.t1_ms + offset_ms)
        for word in getattr(result, "words", ())
    )
    return SimpleNamespace(text=result.text, language=getattr(result, "language", None), segments=segments, words=words)


def run_qwen_chunked(
    session: Any,
    audio: Any,
    *,
    sample_rate: int,
    language: str | None,
    timestamps: str,
    max_seconds: float = 30.0,
) -> SimpleNamespace:
    max_samples = max(1, int(sample_rate * max_seconds))
    starts = list(range(0, len(audio), max_samples))
    chunks = [audio[start : min(len(audio), start + max_samples)] for start in starts]
    results = session.run_batch(chunks, language=language, timestamps=timestamps)
    parts = [
        _offset_result(result, round(start * 1000 / sample_rate)) for start, result in zip(starts, results, strict=True)
    ]
    return SimpleNamespace(
        text=" ".join(part.text.strip() for part in parts if part.text.strip()),
        language=next((part.language for part in parts if part.language), None),
        segments=tuple(seg for part in parts for seg in part.segments),
        words=tuple(word for part in parts for word in part.words),
    )


class TranscribeCppDiarizationModelManager(BaseModelManager):
    def __init__(self, ttl: int, backend: str = "vulkan") -> None:
        super().__init__(ttl)
        self.backend = backend

    def _load_fn(self, model_id: str) -> Any:
        import transcribe_cpp  # pyrefly: ignore[missing-import]

        model_path = transcribe_cpp_diarization_model_registry.get_model_files(model_id)
        return transcribe_cpp.Model(str(model_path), backend=self.backend)

    def diarize(self, model_id: str, audio: Audio) -> Any:
        import transcribe_cpp  # pyrefly: ignore[missing-import]

        with self.load_model(model_id) as model:
            family = transcribe_cpp.SortformerStreamOptions(preset="very_high_latency")
            with model.session() as session:
                return session.run(audio.data, timestamps="none", family=family)


class TranscribeCppModelManager(BaseModelManager):
    def __init__(self, ttl: int, backend: str = "vulkan") -> None:
        super().__init__(ttl)
        self.backend = backend

    def _load_fn(self, model_id: str) -> Any:
        import transcribe_cpp  # pyrefly: ignore[missing-import]

        model_path = transcribe_cpp_model_registry.get_model_files(model_id)
        return transcribe_cpp.Model(str(model_path), backend=self.backend)

    def handle_non_streaming_transcription_request(
        self, request: TranscriptionRequest, **_kwargs
    ) -> NonStreamingTranscriptionResponse:
        with self.load_model(request.model) as model:
            language = resolve_language_hint(model, request.language)
            timestamps = resolve_timestamp_mode(model, request.timestamp_granularities)
            with model.session() as session:
                if getattr(model, "arch", "") == "qwen3_asr" and request.audio.duration > 30.0:
                    result = run_qwen_chunked(
                        session,
                        request.audio.data,
                        sample_rate=request.audio.sample_rate,
                        language=language,
                        timestamps=timestamps,
                    )
                else:
                    result = session.run(request.audio.data, language=language, timestamps=timestamps)
        return transcribe_result_to_response(
            result,
            request.response_format,
            duration=request.audio.duration,
            language=language or request.language,
        )

    def handle_streaming_transcription_request(
        self, request: TranscriptionRequest, **_kwargs
    ) -> Generator[StreamingTranscriptionEvent]:
        import openai.types.audio
        import transcribe_cpp  # pyrefly: ignore[missing-import]

        with self.load_model(request.model) as model:
            if not getattr(model.capabilities, "supports_streaming", False):
                response = cast(
                    "openai.types.audio.Transcription",
                    self.handle_non_streaming_transcription_request(
                        request.model_copy(update={"stream": False, "response_format": "json"})
                    ),
                )
                text = response.text
                yield openai.types.audio.TranscriptionTextDeltaEvent(
                    type="transcript.text.delta", delta=text, logprobs=None
                )
                yield openai.types.audio.TranscriptionTextDoneEvent(
                    type="transcript.text.done", text=text, logprobs=None
                )
                return

            language = resolve_language_hint(model, request.language)
            timestamps = resolve_timestamp_mode(model, request.timestamp_granularities)
            family = None
            if getattr(model, "arch", "") == "parakeet":
                family = transcribe_cpp.ParakeetStreamOptions(att_context_right=6)
            stream_kwargs = {"language": language, "timestamps": timestamps}
            if family is not None:
                stream_kwargs["family"] = family

            chunk_samples = max(1, int(request.audio.sample_rate * 1.12))
            emitted = ""
            with model.session() as session, session.stream(**stream_kwargs) as stream:
                for start in range(0, len(request.audio.data), chunk_samples):
                    update = stream.feed(request.audio.data[start : start + chunk_samples])
                    if update.committed_changed:
                        committed = stream.text().committed
                        if committed.startswith(emitted):
                            delta = committed[len(emitted) :]
                            if delta:
                                yield openai.types.audio.TranscriptionTextDeltaEvent(
                                    type="transcript.text.delta", delta=delta, logprobs=None
                                )
                            emitted = committed
                stream.finalize()
                final_text = stream.text().display
                if final_text.startswith(emitted):
                    delta = final_text[len(emitted) :]
                    if delta:
                        yield openai.types.audio.TranscriptionTextDeltaEvent(
                            type="transcript.text.delta", delta=delta, logprobs=None
                        )
                yield openai.types.audio.TranscriptionTextDoneEvent(
                    type="transcript.text.done", text=final_text, logprobs=None
                )

    def handle_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        return self.handle_non_streaming_transcription_request(request, **kwargs)


def transcribe_result_to_response(
    result: Any,
    response_format: str,
    *,
    duration: float,
    language: str | None = None,
) -> NonStreamingTranscriptionResponse:
    import openai.types.audio

    from speaches.text_utils import format_as_srt, format_as_vtt

    if response_format == "text":
        return result.text, "text/plain"
    if response_format == "json":
        return openai.types.audio.Transcription(text=result.text)

    segments = [
        openai.types.audio.TranscriptionSegment(
            id=i,
            seek=0,
            start=seg.t0_ms / 1000,
            end=seg.t1_ms / 1000,
            text=seg.text,
            tokens=[],
            temperature=0.0,
            avg_logprob=0.0,
            compression_ratio=0.0,
            no_speech_prob=0.0,
        )
        for i, seg in enumerate(result.segments)
    ]
    words = [
        openai.types.audio.TranscriptionWord(
            start=word.t0_ms / 1000,
            end=word.t1_ms / 1000,
            word=word.text,
        )
        for word in result.words
    ]
    if response_format == "verbose_json":
        return openai.types.audio.TranscriptionVerbose(
            language=result.language or language or "unknown",
            duration=duration,
            text=result.text,
            segments=segments,
            words=words,
        )
    if response_format == "vtt":
        return "".join(
            format_as_vtt(seg.text, seg.t0_ms / 1000, seg.t1_ms / 1000, i) for i, seg in enumerate(result.segments)
        ), "text/vtt"
    if response_format == "srt":
        return "".join(
            format_as_srt(seg.text, seg.t0_ms / 1000, seg.t1_ms / 1000, i) for i, seg in enumerate(result.segments)
        ), "text/plain"
    raise ValueError(f"Unsupported response format: {response_format}")


def resolve_timestamp_mode(model: Any, requested_granularities: TimestampGranularities) -> str:
    if getattr(model, "arch", "") == "qwen3_asr":
        return "none"
    if "word" in requested_granularities:
        return "word"
    if "segment" in requested_granularities:
        return "segment"
    return "none"


def resolve_language_hint(model: Any, requested: str | None) -> str | None:
    if requested is None:
        return None
    if getattr(model, "arch", "") == "qwen3_asr":
        return None

    languages = tuple(getattr(getattr(model, "capabilities", None), "languages", ()) or ())
    if requested in languages:
        return requested

    requested_lower = requested.lower()
    for language in languages:
        language_lower = language.lower()
        if language_lower == requested_lower or language_lower.startswith(requested_lower + "-"):
            return language
    return requested if not languages else None

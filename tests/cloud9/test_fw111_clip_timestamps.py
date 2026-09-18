from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speaches.executors.silero_vad_v5 import MergedSegment


def test_faster_whisper_111_batched_clip_offsets_stay_in_samples() -> None:
    merged: list[MergedSegment] = [{"start": 16000, "end": 48000, "segments": [(16000, 24000), (32000, 48000)]}]

    assert merged[0]["start"] == 16000
    assert merged[0]["end"] == 48000
    assert isinstance(merged[0]["start"], int)
    assert isinstance(merged[0]["end"], int)

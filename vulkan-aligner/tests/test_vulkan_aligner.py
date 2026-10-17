from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import choose_model, restore_original_tokens, normalized, MODELS


def test_auto_fr_en_zh_and_multilingual(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "ROOT", tmp_path)
    for folder, filename in MODELS.values():
        directory = tmp_path / folder
        directory.mkdir(exist_ok=True)
        (directory / filename).touch()
    assert choose_model(None, "fr", "Bonjour")[0] == "fastconformer-fr-q4"
    assert choose_model(None, "en", "Hello")[0] == "fastconformer-en-q4"
    assert choose_model(None, "zh", "你好")[0] == "wav2vec2-zh-q4"
    assert choose_model(None, "de", "Guten Tag")[0] == "canary-ctc-multilingual-q4"
    with pytest.raises(ValueError, match="Chinese"):
        choose_model("canary-ctc-multilingual-q4", "zh", "你好")


def test_split_hyphenated_token_uses_measured_boundaries():
    words = [{"word":"Tais-", "start":0.1,"end":0.4},
             {"word":"toi", "start":0.4,"end":0.7}]
    out=restore_original_tokens(words, "Tais-toi")
    assert out == [{"word":"Tais-toi","start":0.1,"end":0.7}]


def test_never_silently_accept_dropped_or_added_words():
    with pytest.raises(ValueError):
        restore_original_tokens([{"word":"Bonjour","start":0,"end":0.4}], "Bonjour amis")

def test_chinese_character_onsets_are_independent_and_measured():
    from server import acoustic_tokens
    from alignment_groups import groups_for_alignment
    assert acoustic_tokens("你好世界。 OpenAI") == ["你", "好", "世", "界。", "OpenAI"]
    source = [{"start": 0.0, "end": 2.0, "text": "你好世界"}]
    grouped = groups_for_alignment("你 好 世 界", 2.0, __import__("json").dumps(source))
    assert len(grouped) == 1
    measured = [{"word": c, "start": i * 0.3, "end": (i + 1) * 0.3}
                for i, c in enumerate("你好世界")]
    assert restore_original_tokens(measured, "你好世界") == measured

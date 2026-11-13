from pathlib import Path

import pytest

from phoneme_modelling.evaluation.llm_listening import LLMListenerEvaluator, parse_answer
from phoneme_modelling.schema.llm_listener_config import LLMListenerConfig
from phoneme_modelling.util.ipa import expected_ipa


@pytest.mark.parametrize("actual, expected", [
    ("ɹ", "ɹ"),
    ("tʃ", "tʃ"),
    ("distortion[ɹ(40%)|l(60%)]", "l"),
    ("distortion[ɹ(0%)|w(100%)]", "w"),
    ("distortion[ɹ(40%)|ɹ̞(60%)]", "ɹ̞"),
    ("distortion[s(40%)|ð(60%)]", "ð"),
])
def test_expected_ipa(actual, expected):
    assert expected_ipa(actual) == expected


def test_parse_answer_tolerates_code_fences():
    answer = parse_answer('```json\n{"transcription_ipa": "sː", "heard_sound": "S", "confidence": 0.9, "notes": ""}\n```')
    assert answer.heard_sound == "s"
    assert answer.transcription_ipa == "sː"


@pytest.mark.parametrize("text", ["no json here", '{"heard_sound": "x", "confidence": 0.5}', '{"heard_sound": "s", "confidence": 3}'])
def test_parse_answer_rejects_bad_answers(text):
    with pytest.raises(Exception):
        parse_answer(text)


class FakeListener(LLMListenerEvaluator):
    calls = 0

    @staticmethod
    def get_name() -> str:
        return "fake_listener"

    def _ask(self, wav_bytes: bytes) -> str:
        FakeListener.calls += 1
        return '{"transcription_ipa": "l", "heard_sound": "l", "confidence": 0.8, "notes": ""}'


def test_evaluate_uses_main_sound_of_distortion_and_caches(tmp_path: Path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF fake audio")
    config = LLMListenerConfig(model="fake", cache_dir=tmp_path / "cache", workers=2)
    listener = FakeListener(config)
    listener.prepare([clip, clip])
    result = listener.evaluate("r", "distortion[ɹ(40%)|l(60%)]", clip)
    assert result.is_correct and result.detected_phoneme_ipa == "l" and result.quality == 0.8
    assert not listener.evaluate("r", "ɹ", clip).is_correct
    assert FakeListener.calls == 1
    FakeListener(config).evaluate("r", "ɹ", clip)  # a new instance reads the cache from disk
    assert FakeListener.calls == 1

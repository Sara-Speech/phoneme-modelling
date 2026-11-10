import io
import shutil
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from phoneme_modelling.phonemes import TEXT_TO_IPA, VALID_IPA_STRINGS
from phoneme_modelling.schema.google_tts_config import GoogleTTSConfig
from phoneme_modelling.schema.phoneme_pipeline_config import PhonemePipelineConfig
from phoneme_modelling.synthetic_audio.generators.google_tts import GoogleTTSGenerator, GoogleTTSVoice
from phoneme_modelling.synthetic_audio.google_tts_sounds import AFFRICATE, GOOGLE_TTS_SOUNDS, HOLD, SINGLE
from phoneme_modelling.util.forced_alignment import ctc_forced_align, normalize_phone, tokenize_longest

RATE = 16000
TARGET_SOUNDS = [
    "r", "s", "g", "b", "z", "l", "th_voiced", "th_voiceless", "sh", "ch", "k", "v", "j", "t", "d", "f", "ng", "y",
    "w", "h", "n", "m", "p",
]


class FakeTextToSpeechClient:
    """Returns a two-second tone as the WAV bytes Cloud TTS would send, and remembers the SSML it was asked for."""

    def __init__(self) -> None:
        self.ssml: list[str] = []

    def synthesize_speech(self, input, voice, audio_config):  # noqa: A002 - mirrors the real client
        self.ssml.append(input.ssml)
        t = np.arange(2 * RATE) / RATE
        buffer = io.BytesIO()
        sf.write(buffer, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), RATE, format="WAV", subtype="PCM_16")
        return SimpleNamespace(audio_content=buffer.getvalue())


class FakeAligner:
    def __init__(self, span) -> None:
        self.span = span
        self.calls: list[tuple] = []

    def locate(self, audio, before, phones, after, targets):
        self.calls.append((before, phones, after, targets))
        return self.span


def make_voice(client, aligner, take=1, takes=1, repeats=8) -> GoogleTTSVoice:
    return GoogleTTSVoice(client, aligner, "en-US-Neural2-F", "en-US", repeats=repeats, take=take, takes=takes)


def test_every_target_sound_has_a_recipe_with_a_valid_ipa_symbol():
    assert set(GOOGLE_TTS_SOUNDS) == set(TARGET_SOUNDS)
    for name in TARGET_SOUNDS:
        assert TEXT_TO_IPA[name] in VALID_IPA_STRINGS


def test_ipa_sequences():
    assert GOOGLE_TTS_SOUNDS["r"].ipa_sequence(4) == "ɹɹɹɹʌn"
    assert GOOGLE_TTS_SOUNDS["ng"].ipa_sequence(2) == "ʌŋŋ"
    assert GOOGLE_TTS_SOUNDS["b"].ipa_sequence(8) == "bʌn"  # stops ignore the repeat count
    assert GOOGLE_TTS_SOUNDS["th_voiceless"].ipa_sequence(8) == "θʌn"
    assert GOOGLE_TTS_SOUNDS["ch"].ipa_sequence(8) == "tʃʃʃʌn"
    assert GOOGLE_TTS_SOUNDS["j"].ipa_sequence(8) == "dʒʒʒʌn"
    assert {GOOGLE_TTS_SOUNDS[s].kind for s in TARGET_SOUNDS} == {HOLD, SINGLE, AFFRICATE}


def test_voice_nickname_is_the_last_part_of_the_clip_name():
    assert make_voice(None, None).get_voice_nickname() == "neural2f"
    assert make_voice(None, None, take=2, takes=3).get_voice_nickname() == "neural2f_take2"


def test_normalize_and_tokenize():
    assert normalize_phone("aː") == "a"
    assert normalize_phone("ˈs") == "s"
    assert tokenize_longest("tʃ ʌ n", ["t", "ʃ", "tʃ", "ʌ", "n"]) == ["tʃ", "ʌ", "n"]
    assert tokenize_longest("s?s", ["s"]) == ["s", "s"]  # unknown characters are skipped


def test_forced_alignment_finds_each_token():
    blank, vocabulary = 0, 5
    log_probs = np.full((30, vocabulary), np.log(0.01))
    log_probs[:, blank] = np.log(0.96)
    for token, (first, last) in zip([1, 2, 3], [(5, 9), (12, 16), (20, 24)]):
        log_probs[first:last, token] = np.log(0.97)
        log_probs[first:last, blank] = np.log(0.005)
    spans = ctc_forced_align(log_probs, [1, 2, 3], blank=blank)
    for (start, end), (first, last) in zip(spans, [(5, 9), (12, 16), (20, 24)]):
        assert first <= start < end <= last + 1


def test_forced_alignment_rejects_audio_that_is_too_short():
    with pytest.raises(ValueError):
        ctc_forced_align(np.log(np.full((2, 4), 0.25)), [1, 2, 3], blank=0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="prep_clip needs ffmpeg")
def test_generate_audio_writes_a_prepped_clip(tmp_path):
    client, aligner = FakeTextToSpeechClient(), FakeAligner((0.5, 0.9))
    target = tmp_path / "0000_r_neural2f.wav"
    assert make_voice(client, aligner)._generate_audio(target, "r", "ɹ") is True
    with wave.open(str(target), "rb") as clip:
        assert clip.getframerate() == RATE and clip.getnchannels() == 1
        assert clip.getnframes() / RATE == pytest.approx(2.5, abs=0.01)
    assert (tmp_path / "raw" / target.name).exists()
    assert 'ph="ɹɹɹɹɹɹɹɹʌn"' in client.ssml[0]
    before, phones, after, targets = aligner.calls[0]
    assert phones == "ɹɹɹɹɹɹɹɹʌn" and "ɹ" in targets


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="prep_clip needs ffmpeg")
def test_stops_are_generated_once_not_once_per_take(tmp_path):
    client, aligner = FakeTextToSpeechClient(), FakeAligner((0.5, 0.52))
    first = make_voice(client, aligner, take=1, takes=2)._generate_audio(tmp_path / "a.wav", "b", "b")
    second = make_voice(client, aligner, take=2, takes=2)._generate_audio(tmp_path / "b.wav", "b", "b")
    assert first is True and second is False
    assert not (tmp_path / "b.wav").exists()


def test_failures_return_false_instead_of_raising(tmp_path):
    client = FakeTextToSpeechClient()
    assert make_voice(client, FakeAligner(None))._generate_audio(tmp_path / "a.wav", "r", "ɹ") is False  # not found
    assert make_voice(client, FakeAligner((0.5, 0.9)))._generate_audio(tmp_path / "b.wav", "zh", "ʒ") is False  # no recipe


def test_generator_config_name():
    assert GoogleTTSGenerator.get_config_name() == "google_tts"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="prep_clip needs ffmpeg")
def test_generator_runs_through_the_shared_pipeline_naming(tmp_path, monkeypatch):
    from phoneme_modelling.synthetic_audio.generators import google_tts

    monkeypatch.setattr(google_tts.texttospeech, "TextToSpeechClient", lambda **kwargs: FakeTextToSpeechClient())
    monkeypatch.setattr(google_tts, "PhoneAligner", lambda: FakeAligner((0.5, 0.9)))
    generator = GoogleTTSGenerator(
        GoogleTTSConfig(voices=["en-US-Neural2-F", "en-US-Neural2-D"], hold_repeats=[4, 8])
    )
    request = PhonemePipelineConfig.model_validate(dict(
        number_of_digits=4, phonemes_to_generate=["r", "b"], generator_configs={},
        distortion_application_config=dict(distortion_intensitites=[0.0]), validation_config={},
    ))
    count, metadata = generator.generate_synthetic_audios(request, 0, tmp_path)
    names = sorted(entry.file_name for entry in metadata)
    # r: two voices x two hold lengths; b (a stop) is made once per voice, in take 1 only
    assert names == [
        "0000_r_neural2f_take1.wav", "0001_r_neural2f_take2.wav", "0002_r_neural2d_take1.wav",
        "0003_r_neural2d_take2.wav", "0004_b_neural2f_take1.wav", "0005_b_neural2d_take1.wav",
    ]
    assert count == 6 and len(metadata) == 6
    assert {entry.model for entry in metadata} == {"google_tts"}
    assert {entry.target_phoneme_text for entry in metadata} == {"r", "b"}

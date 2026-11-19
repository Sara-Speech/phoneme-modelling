import pandas as pd
import pytest

from phoneme_modelling.evaluation.llm_listening import NONE, OTHER, ipa_to_phoneme, parse_answer
from phoneme_modelling.util.loaders import load_audio_metadata


@pytest.mark.parametrize("ipa, phoneme", [
    ("s", "s"),
    ("sː", "s"),
    ("sss", "s"),
    ("/s/", "s"),
    ("sə", "s"),
    ("əɹ", "r"),
    ("ɹ̞", "r"),
    ("r", "r"),
    ("ɚ", "r"),
    ("ɡ", "g"),
    ("gə", "g"),
    ("kʰ", "k"),
    ("pʰə", "p"),
    ("t͡ʃ", "ch"),
    ("ʧ", "ch"),
    ("dʒɑ", "j"),
    ("ð", "th_voiced"),
    ("θ", "th_voiceless"),
    ("ʃ", "sh"),
    ("ŋ", "ng"),
    ("j", "y"),
    ("ˈwʌ", "w"),
    ("", NONE),
    ("ə", NONE),
    ("ts", OTHER),
    ("ʒ", OTHER),
])
def test_ipa_to_phoneme(ipa, phoneme):
    assert ipa_to_phoneme(ipa) == phoneme


def test_parse_answer_tolerates_code_fences():
    assert parse_answer('```json\n{"ipa": "sː"}\n```').ipa == "sː"


@pytest.mark.parametrize("text", ['{"ipa": "m}', '{"ipa": "m', '{"ipa": "m\n}'])
def test_parse_answer_tolerates_unterminated_string(text):
    assert parse_answer(text).ipa == "m"


def test_parse_answer_rejects_a_refusal():
    with pytest.raises(ValueError):
        parse_answer("Sure, please provide the audio clip.")


def test_load_audio_metadata_drops_removed_rows_and_resolves_paths(tmp_path):
    (tmp_path / "phonemes").mkdir()
    for name in ["a.wav", "b.wav", "c.wav"]:
        (tmp_path / "phonemes" / name).touch()
    pd.DataFrame({
        "file_name": ["a.wav", "b.wav", "c.wav"],
        "file_path": ["phonemes/a.wav", "phonemes/b.wav", "phonemes/c.wav"],
        "Remove?": [None, "Y", " y "],
    }).to_excel(tmp_path / "metadata.xlsx", index=False)
    rows = load_audio_metadata(tmp_path / "metadata.xlsx", tmp_path)
    assert [row["file_name"] for row in rows] == ["a.wav"]
    assert rows[0]["file_path"] == str(tmp_path / "phonemes" / "a.wav")
    assert "Remove?" not in rows[0]

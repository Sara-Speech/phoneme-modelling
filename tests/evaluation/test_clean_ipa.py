import pytest

from phoneme_modelling.util.ipa import clean_ipa


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("s", "s"),
        ("b ə", "b"),  # vowels are ignored
        ("▁ ə n", "n"),  # ZIPA's word-boundary marker is ignored
        ("ɡ a", "g"),  # script ɡ (U+0261) inside a longer output, from wav2vec2 eSpeak
        ("ɝɝɝɝ", "ɹ"),  # an r-coloured vowel for a held /r/, from wav2vec2 (TIMIT)
        ("ɚ", "ɹ"),
        ("t̠ʃ a", "tʃ"),  # longest variant first, so this is not read as "tʃ" + "ʃ"
        ("d̠ʒ", "dʒ"),
        ("ʃ ʃ s", "ʃ"),  # the most frequent phoneme wins
        ("ə a", ""),  # no consonant
        ("", ""),
    ],
)
def test_clean_ipa(raw: str, expected: str) -> None:
    assert clean_ipa(raw) == expected

from collections import Counter
import re

from phoneme_modelling.phonemes import ALTERNATIVE_IPA, IpaRepresentations


STRINGS_TO_REMOVE_FROM_IPA = [" ", "_", "▁", ".", ",", ";"]

ORDERED_IPA_PHONEMES: list[str] = sorted(
    [member.value for member in IpaRepresentations],
    key=len,
    reverse=True,
)

VALID_IPA_PATTERN = re.compile(
    "|".join(re.escape(p) for p in ORDERED_IPA_PHONEMES)
)

# Longest first, so "t̠ʃ" becomes "tʃ" before "t̠" alone is considered.
ORDERED_ALTERNATIVE_IPA: list[tuple[str, str]] = sorted(ALTERNATIVE_IPA.items(), key=lambda item: len(item[0]), reverse=True)


def clean_ipa(ipa: str) -> str:
    """The most frequent of the IpaRepresentations phonemes in a recognizer's output, or "" if there is none.
    Variant symbols (script ɡ, ɝ/ɚ for /r/, t̠ʃ, ...) are replaced wherever they appear, not only when they are the
    whole output, so "ɡ a" counts as /g/ and "ɝɝɝ" as /ɹ/."""
    for text in STRINGS_TO_REMOVE_FROM_IPA:
        ipa = ipa.replace(text, "")
    for alternative, replacement in ORDERED_ALTERNATIVE_IPA:
        ipa = ipa.replace(alternative, replacement)
    matched_phonemes: list[str] = VALID_IPA_PATTERN.findall(ipa)
    if not matched_phonemes:
        return ""
    return Counter(matched_phonemes).most_common(1)[0][0]
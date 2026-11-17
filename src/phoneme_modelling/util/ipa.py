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


def clean_ipa(ipa: str) -> str:
    for text in STRINGS_TO_REMOVE_FROM_IPA:
        ipa = ipa.replace(text, "")
    if ipa in ALTERNATIVE_IPA:
        ipa = ALTERNATIVE_IPA.get(ipa, ipa)
    matched_phonemes: list[str] = VALID_IPA_PATTERN.findall(ipa)
    if not matched_phonemes:
        return ""
    return Counter(matched_phonemes).most_common(1)[0][0]
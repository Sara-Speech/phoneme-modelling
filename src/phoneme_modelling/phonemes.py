from enum import Enum

class IpaRepresentations(Enum):
    R = "ɹ"
    S = "s"
    G = "g"
    B = "b"
    Z = "z"
    L = "l"
    TH_VOICED = "ð"
    TH_VOICELESS = "θ"
    SH = "ʃ"
    CH = "tʃ"
    K = "k"
    V = "v"
    J = "dʒ"
    T = "t"
    D = "d"
    F = "f"
    NG = "ŋ"
    Y = "j"
    W = "w"
    H = "h"
    N = "n"
    M = "m"
    P = "p"

ALTERNATIVE_IPA = {
    "ʤ": "dʒ",
    "ʧ": "tʃ",
    "r": "ɹ",
    "ɹ̩": "ɹ",
    "ɻ": "ɹ",
    "ɾ": "ɹ",
    "ɚ": "ɹ",
    "ɝ": "ɹ",
    "ɡ": "g",
    "ɡʲ": "g",
    "ɣ": "g",
    "pʰ": "p",
    "p̚": "p",
    "tʰ": "t",
    "t̚": "t",
    "t̪": "t",
    "kʰ": "k",
    "k̚": "k",
    "kʲ": "k",
    "b̚": "b",
    "d̚": "d",
    "d̪": "d",
    "t̠ʃ": "tʃ",
    "tʃʰ": "tʃ",
    "t̠": "tʃ",
    "d̠ʒ": "dʒ",
    "d̠": "dʒ",
    "s̪": "s",
    "sː": "s",
    "z̪": "z",
    "zː": "z",
    "ʃʲ": "ʃ",
    "l̪": "l",
    "lː": "l",
    "ɫ": "l",
    "l̩": "l",
    "m̩": "m",
    "n̩": "n",
    "n̪": "n",
    "ŋ̩": "ŋ",
    "ɦ": "h",
    "ʍ": "w",
}

TEXT_TO_IPA = {}
for member in IpaRepresentations:
    TEXT_TO_IPA[member.name.upper()] = member.value
    TEXT_TO_IPA[member.name.lower()] = member.value

IPA_TO_TEXT = {member.value: member.name.lower() for member in IpaRepresentations}

VALID_SOUND_TEXTS = set()
for member in IpaRepresentations:
    VALID_SOUND_TEXTS.add(member.name.upper())
    VALID_SOUND_TEXTS.add(member.name.lower())

VALID_IPA_STRINGS = set([member.value for member in IpaRepresentations])

all_ipas = [member.value for member in IpaRepresentations]
all_ipas.extend(list(ALTERNATIVE_IPA.keys()))
ALL_ACCEPTED_IPA_STRINGS = set(all_ipas)

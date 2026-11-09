"""How each of the 23 target sounds is spoken by Google Cloud TTS and found again in the audio.

A sound is sent as an SSML IPA phoneme tag inside "Please say ___ again." (a bare sound with no context is unstable).
Continuants can be held by repeating their symbol inside the tag, which gives 0.3 to 0.7 s of sound. Stops and the two
th sounds cannot be held (a repeated th turns into a "t"), so they stay single, and ch and j get a longer fricative half.
"""
from dataclasses import dataclass

HOLD = "hold"
SINGLE = "single"
AFFRICATE = "affricate"

AFFRICATE_FRICATIVE_REPEATS = 3
STOP_MARGIN_SECONDS = 0.03  # kept around a crop when the sound is a burst; held sounds are cut at the aligned edges


@dataclass(frozen=True)
class GoogleTTSSound:
    ipa: str  # the sound as Google's IPA tag spells it
    word: str  # the word inside the tag (Google speaks the ``ph`` text instead)
    kind: str  # HOLD, SINGLE or AFFRICATE
    aligned_as: frozenset[str]  # the phones the aligner may report for it
    head: str = ""  # IPA spoken before the sound
    tail: str = "ʌn"  # IPA spoken after the sound
    fricative: str = ""  # AFFRICATE only: the fricative half, repeated after ``ipa``

    def ipa_sequence(self, repeats: int) -> str:
        """The IPA inside the tag. ``repeats`` only matters for held sounds."""
        if self.kind == HOLD:
            return self.head + self.ipa * repeats + self.tail
        if self.kind == AFFRICATE:
            return self.ipa + self.fricative * AFFRICATE_FRICATIVE_REPEATS + self.tail
        return self.head + self.ipa + self.tail

    def margin_seconds(self) -> float:
        return 0.0 if self.kind == HOLD else STOP_MARGIN_SECONDS


_G = "ɡ"  # IPA g, the character the aligner's vocabulary uses


def _hold(ipa: str, word: str, head: str = "", tail: str = "ʌn") -> GoogleTTSSound:
    return GoogleTTSSound(ipa=ipa, word=word, kind=HOLD, aligned_as=frozenset({ipa}), head=head, tail=tail)


def _single(ipa: str, word: str, aligned_as: set[str] | None = None) -> GoogleTTSSound:
    return GoogleTTSSound(ipa=ipa, word=word, kind=SINGLE, aligned_as=frozenset(aligned_as or {ipa}))


# keys are the sound names used by the pipeline (phoneme_modelling.phonemes)
GOOGLE_TTS_SOUNDS: dict[str, GoogleTTSSound] = {
    "r": _hold("ɹ", "run"),
    "l": _hold("l", "lun"),
    "w": _hold("w", "wun"),
    "y": _hold("j", "yun"),
    "s": _hold("s", "sun"),
    "z": _hold("z", "zun"),
    "sh": _hold("ʃ", "shun"),
    "f": _hold("f", "fun"),
    "v": _hold("v", "vun"),
    "h": _hold("h", "hun"),
    "m": _hold("m", "mun"),
    "n": _hold("n", "nun"),
    "ng": _hold("ŋ", "ung", head="ʌ", tail=""),
    "th_voiceless": _single("θ", "thun"),
    "th_voiced": _single("ð", "thun"),
    "b": _single("b", "bun"),
    "p": _single("p", "pun"),
    "d": _single("d", "dun"),
    "t": _single("t", "tun"),
    "g": _single(_G, "gun", {_G, "g"}),
    "k": _single("k", "kun"),
    "ch": GoogleTTSSound(ipa="t", word="chun", kind=AFFRICATE, aligned_as=frozenset({"tʃ", "t", "ʃ"}), fricative="ʃ"),
    "j": GoogleTTSSound(ipa="d", word="jun", kind=AFFRICATE, aligned_as=frozenset({"dʒ", "d", "ʒ"}), fricative="ʒ"),
}

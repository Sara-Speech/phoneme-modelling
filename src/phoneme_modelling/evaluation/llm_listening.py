"""Shared logic for evaluators that ask a multimodal LLM to listen to a clip and transcribe it in IPA.

The model is not told which sound the clip should be. It is given the list of the phonemes in IpaRepresentations
(each with an example word) and writes the IPA it hears. That IPA is then mapped to one of the phonemes in IpaRepresentations by ipa_to_phoneme, in
code, so the mapping is deterministic and the same for every model.

Answers are cached per clip (keyed by model, prompt version and the audio's hash), so a rerun or an interrupted run
doesn't pay for clips it already has.
"""
import hashlib
import json
import random
import re
import time
import unicodedata
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from phoneme_modelling.evaluation.evaluator import AudioEvaluator
from phoneme_modelling.phonemes import IPA_TO_TEXT, IpaRepresentations
from phoneme_modelling.schema.llm_listener_config import LLMListenerConfig
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluation
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata


PROMPT_VERSION = 2  # bump when LISTENER_PROMPT changes, so answers cached under the old prompt are not reused

# An English word for each phoneme, so the model reads each IPA symbol the intended way (j is the "y" in "yes").
EXAMPLE_WORDS = {
    IpaRepresentations.R: "red",
    IpaRepresentations.S: "sun",
    IpaRepresentations.G: "go",
    IpaRepresentations.B: "bat",
    IpaRepresentations.Z: "zoo",
    IpaRepresentations.L: "leg",
    IpaRepresentations.TH_VOICED: "this",
    IpaRepresentations.TH_VOICELESS: "think",
    IpaRepresentations.SH: "she",
    IpaRepresentations.CH: "chair",
    IpaRepresentations.K: "cat",
    IpaRepresentations.V: "van",
    IpaRepresentations.J: "jam",
    IpaRepresentations.T: "top",
    IpaRepresentations.D: "dog",
    IpaRepresentations.F: "fan",
    IpaRepresentations.NG: "sing",
    IpaRepresentations.Y: "yes",
    IpaRepresentations.W: "wet",
    IpaRepresentations.H: "hat",
    IpaRepresentations.N: "net",
    IpaRepresentations.M: "man",
    IpaRepresentations.P: "pen",
}
PHONEME_LIST = "\n".join(f"- {phoneme.value} (as in \"{EXAMPLE_WORDS[phoneme]}\")" for phoneme in IpaRepresentations)

LISTENER_PROMPT = f"""You are an expert phonetician. Listen to the audio clip and transcribe the consonant you hear in \
the International Phonetic Alphabet.

The clip is a short recording of an isolated English consonant, not a word, possibly with a short vowel before or after \
it. It should be one of these {len(IpaRepresentations)} consonants:
{PHONEME_LIST}

Transcribe what is actually said, not what the speaker probably meant, and don't guess a word. Answer with the symbol \
from the list that matches the consonant you hear. If it is clearly none of them, write the IPA of what you hear \
instead. Use standard IPA symbols only (no letters spelled out, no slashes or brackets). If there is no speech, use an \
empty string.

Reply with only a JSON object of the form {{"ipa": "<your transcription>"}}."""

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {"ipa": {"type": "string"}},
    "required": ["ipa"],
    "additionalProperties": False,
}

OTHER = "other"  # speech that doesn't reduce to one of the phonemes (two consonants, an unlisted sound, ...)
NONE = "none"  # no consonant at all: silence, or only a vowel

# Symbols that mean the same phoneme as one in IpaRepresentations. ɚ/ɝ are how a held English /r/ is often written.
IPA_EQUIVALENTS = {
    "ɡ": "g",  # U+0261 script g
    "r": "ɹ",
    "ɻ": "ɹ",
    "ɚ": "ɹ",
    "ɝ": "ɹ",
    "ʤ": "dʒ",
    "ʧ": "tʃ",
}
VOWELS = set("aeiouyæɐɑɒɔəɘɛɜɞɤɨɪʉʊʌʏøœɶɯɵ")
# Marks that don't change which phoneme it is: stress, length, syllable breaks, transcription brackets.
IGNORED_CHARACTERS = set("ˈˌːˑ.‿|/[]()-, ")


class ListenerAnswer(BaseModel):
    ipa: str = ""


def parse_answer(text: str) -> ListenerAnswer:
    """The model's JSON answer, tolerating code fences or text around the object. gpt-audio models can't be forced to
    reply in JSON and occasionally drop the closing quote ({"ipa": "m}), so if the JSON doesn't parse, the "ipa" value
    is read up to the next quote, brace or line break."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is not None:
        try:
            return ListenerAnswer.model_validate(json.loads(match.group(0)))
        except json.JSONDecodeError:
            pass
    value = re.search(r'"ipa"\s*:\s*"([^"}\n]*)', text)
    if value is None:
        raise ValueError(f"No ipa value in the answer: {text!r}")
    return ListenerAnswer(ipa=value.group(1).strip())


def normalize_ipa(ipa: str) -> str:
    """The consonants of an IPA transcription: diacritics, tie bars, superscripts (ʰ ʷ ...), stress, length and vowels
    removed, equivalent symbols replaced, and repeats of the same symbol collapsed ("sss" -> "s")."""
    for symbol, replacement in IPA_EQUIVALENTS.items():
        ipa = ipa.replace(symbol, replacement)
    kept = []
    for char in unicodedata.normalize("NFD", ipa):
        if unicodedata.category(char) == "Mn" or char in IGNORED_CHARACTERS or char in VOWELS:
            continue
        if unicodedata.category(char) == "Lm":  # modifier letters: ʰ ʷ ʲ ˠ ˤ ⁿ ˡ ...
            continue
        kept.append(char)
    consonants = unicodedata.normalize("NFC", "".join(kept))
    return re.sub(r"(.)\1+", r"\1", consonants)


def ipa_to_phoneme(ipa: str) -> str:
    """The phoneme (an IpaRepresentations name, lowercase) an IPA transcription reduces to, NONE if it has no
    consonant, or OTHER if it doesn't reduce to exactly one phoneme."""
    consonants = normalize_ipa(ipa)
    if consonants == "":
        return NONE
    return IPA_TO_TEXT.get(consonants, OTHER)


class LLMListenerEvaluator(AudioEvaluator):
    """Base class: subclasses only implement _transcribe (one API call)."""

    REQUEST_VERSION = ""  # set in a subclass when its request format changes, so its old cached answers aren't reused

    def __init__(self, config: LLMListenerConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir) / self.get_name()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _transcribe(self, audio: bytes) -> str:
        """The raw text of the model's answer for one WAV file's bytes."""
        raise NotImplementedError()

    def evaluate(self, audio_info: SyntheticPhonemeMetadata) -> PhonemeEvaluation:
        ipa = self.__listen(Path(audio_info.file_path))
        if ipa is None:  # every attempt failed; counted as incorrect, with no detection
            return PhonemeEvaluation(is_correct=False)
        phoneme = ipa_to_phoneme(ipa)
        return PhonemeEvaluation(
            is_correct=phoneme == audio_info.actual_phoneme_text,
            detected_phoneme_text=phoneme,
            detected_phoneme_ipa=ipa,
        )

    def evaluate_all(self, audio_infos: list[SyntheticPhonemeMetadata]) -> list[PhonemeEvaluation]:
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as pool:
            return list(pool.map(self.evaluate, audio_infos))

    def __listen(self, path: Path) -> Optional[str]:
        """The IPA the model hears in the clip, from the cache when possible. None if every attempt failed."""
        audio = path.read_bytes()
        prefix = f"{self.config.model}|{PROMPT_VERSION}|"
        if self.REQUEST_VERSION:
            prefix += f"{self.REQUEST_VERSION}|"
        key = hashlib.sha256(prefix.encode() + audio).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return ListenerAnswer.model_validate_json(cache_file.read_text()).ipa
        errors = rate_limits = 0
        while True:
            try:
                answer = parse_answer(self._transcribe(audio))
                cache_file.write_text(answer.model_dump_json())
                print(f"{self.get_name()} {path.name}: {answer.ipa!r}")
                return answer.ipa
            except Exception as e:  # rate limits, timeouts, network errors, malformed answers
                if is_rate_limit(e):
                    rate_limits += 1
                    if rate_limits > self.config.max_rate_limit_retries:
                        print(f"{self.get_name()} {path.name}: gave up after {rate_limits} rate limits")
                        return None
                    # Jitter so the parallel requests don't all retry at the same moment and hit the limit again.
                    wait = min(self.config.rate_limit_wait_seconds * 2 ** (rate_limits - 1), 300) * random.uniform(1, 1.5)
                    print(f"{self.get_name()} {path.name}: rate limited (429), retrying in {wait:.0f} s")
                else:
                    errors += 1
                    if errors > self.config.max_retries:
                        print(f"{self.get_name()} {path.name}: gave up after {errors} errors, last: {e}")
                        return None
                    wait = 2 ** (errors - 1)
                    print(f"{self.get_name()} {path.name}: attempt failed ({e}), retrying in {wait} s")
                time.sleep(wait)


def is_rate_limit(error: Exception) -> bool:
    """A 429 / quota error: google-genai sets .code, openai sets .status_code."""
    return getattr(error, "code", None) == 429 or getattr(error, "status_code", None) == 429

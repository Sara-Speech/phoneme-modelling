"""Shared logic for evaluators that ask a multimodal LLM to listen to a clip and say which sound it hears.

The model is not told which sound the clip should be: it picks from the closed set of SOUND_LABELS (plus "other" and
"none") and also writes a free IPA transcription. Answers are cached per clip (keyed by the audio's hash), so an
interrupted run resumes without paying for clips it already has.
"""
import hashlib
import json
import re
import threading
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from phoneme_modelling.evaluation.evaluator import AudioEvaluator
from phoneme_modelling.phonemes import IPA_TO_TEXT, IpaRepresentations
from phoneme_modelling.schema.llm_listener_config import LLMListenerConfig
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluation
from phoneme_modelling.util.ipa import expected_ipa


PROMPT_VERSION = 1  # bump when the prompt changes, so cached answers from the old prompt are not reused

EXAMPLE_WORDS = {
    "r": "red", "s": "sun", "g": "go", "b": "bat", "z": "zoo", "l": "lip", "th_voiced": "this",
    "th_voiceless": "thin", "sh": "ship", "ch": "chip", "k": "kite", "v": "van", "j": "jam", "t": "top",
    "d": "dog", "f": "fan", "ng": "the end of sing", "y": "yes", "w": "wet", "h": "hat", "n": "net",
    "m": "man", "p": "pen",
}
SOUND_LABELS = [member.name.lower() for member in IpaRepresentations]
OTHER_LABEL = "other"  # a speech sound outside the list (a vowel alone, a sound between two categories, ...)
NONE_LABEL = "none"  # silence, noise, or nothing speech-like
ALL_LABELS = SOUND_LABELS + [OTHER_LABEL, NONE_LABEL]

_SOUND_LINES = "\n".join(
    f"- {label}: IPA /{IpaRepresentations[label.upper()].value}/, as in '{EXAMPLE_WORDS[label]}'"
    for label in SOUND_LABELS
)

LISTENER_PROMPT = f"""You are an expert phonetician grading a dataset of short, isolated English speech sounds.

Listen to the audio clip. It should contain one consonant sound, possibly held or followed by a short vowel. Some
clips were edited on purpose so that the sound is mispronounced or changed into another sound. Say what you actually
hear, not what the speaker probably meant.

1. transcription_ipa: a narrow IPA transcription of everything you hear (e.g. "sː", "wə", "ɹ̞"). Use "" for silence.
2. heard_sound: the one consonant from this list that best matches the main sound:
{_SOUND_LINES}
   Use "{OTHER_LABEL}" if the main sound is a speech sound not on the list, and "{NONE_LABEL}" if there is no speech.
3. confidence: from 0 to 1, how sure you are of heard_sound.
4. notes: one short sentence on anything unusual (distortion, extra vowel, noise, clipping). "" if nothing.

Reply with only a JSON object with exactly these keys: transcription_ipa, heard_sound, confidence, notes."""

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "transcription_ipa": {"type": "string"},
        "heard_sound": {"type": "string", "enum": ALL_LABELS},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["transcription_ipa", "heard_sound", "confidence", "notes"],
    "additionalProperties": False,
}


class ListenerAnswer(BaseModel):
    transcription_ipa: str = ""
    heard_sound: str
    confidence: float = Field(ge=0, le=1)
    notes: str = ""


def parse_answer(text: str) -> ListenerAnswer:
    """The model's JSON answer, tolerating code fences or text around the object."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError(f"no JSON object in answer: {text!r}")
    answer = ListenerAnswer.model_validate(json.loads(match.group(0)))
    answer.heard_sound = answer.heard_sound.strip().lower().replace(" ", "_")
    if answer.heard_sound not in ALL_LABELS:
        raise ValueError(f"heard_sound {answer.heard_sound!r} is not one of the labels")
    return answer


class LLMListenerEvaluator(AudioEvaluator):
    """Base for the LLM listeners. Subclasses only implement _ask: send the wav bytes with LISTENER_PROMPT, return the
    model's raw text answer."""

    def __init__(self, config: LLMListenerConfig):
        self.config = config
        cache_dir = Path(config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", config.model)
        self.__cache_path = cache_dir / f"{self.get_name()}_{safe_model}.jsonl"
        self.__cache: dict[str, dict] = {}
        self.__lock = threading.Lock()
        if self.__cache_path.exists():
            for line in self.__cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.__cache[entry["key"]] = entry["answer"]

    @abstractmethod
    def _ask(self, wav_bytes: bytes) -> str:
        raise NotImplementedError()

    def prepare(self, audio_paths: list[Path]) -> None:
        """Answer every clip not yet cached, config.workers at a time."""
        missing = [path for path in dict.fromkeys(audio_paths) if self.__key(path.read_bytes()) not in self.__cache]
        if not missing:
            return
        print(f"{self.get_name()} ({self.config.model}): listening to {len(missing)} clips")
        done = 0
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            for _ in pool.map(self.__answer_safely, missing):
                done += 1
                if done % 50 == 0 or done == len(missing):
                    print(f"{self.get_name()}: {done}/{len(missing)}")

    def evaluate(self, target_phoneme: str, target_phoneme_ipa: str, audio_path: Path) -> PhonemeEvaluation:
        answer = self.__answer(audio_path)
        if answer is None:
            return PhonemeEvaluation(is_correct=False, notes="no answer from the model (see the log)")
        expected_text = IPA_TO_TEXT.get(expected_ipa(target_phoneme_ipa))
        detected_ipa = IpaRepresentations[answer.heard_sound.upper()].value if answer.heard_sound in SOUND_LABELS else None
        return PhonemeEvaluation(
            is_correct=expected_text is not None and answer.heard_sound == expected_text,
            detected_phoneme_text=answer.heard_sound,
            detected_phoneme_ipa=detected_ipa,
            transcription_ipa=answer.transcription_ipa,
            quality=answer.confidence,
            notes=answer.notes or None,
        )

    def __key(self, wav_bytes: bytes) -> str:
        return f"v{PROMPT_VERSION}:{hashlib.sha1(wav_bytes).hexdigest()}"

    def __answer_safely(self, audio_path: Path) -> None:
        try:
            self.__answer(audio_path)
        except Exception as e:  # one bad clip shouldn't stop the run; evaluate() reports it as unanswered
            print(f"FAILED {self.get_name()} {audio_path.name}: {e}")

    def __answer(self, audio_path: Path) -> Optional[ListenerAnswer]:
        wav_bytes = audio_path.read_bytes()
        key = self.__key(wav_bytes)
        with self.__lock:
            if key in self.__cache:
                return ListenerAnswer.model_validate(self.__cache[key])
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                answer = parse_answer(self._ask(wav_bytes))
                break
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                last_error = e
            except Exception as e:  # API errors: back off, then retry
                last_error = e
                time.sleep(2 ** attempt)
        else:
            print(f"FAILED {self.get_name()} {audio_path.name}: {last_error}")
            return None
        with self.__lock:
            self.__cache[key] = answer.model_dump()
            with open(self.__cache_path, "a", encoding="utf-8") as cache_file:
                record = {"key": key, "file_name": audio_path.name, "answer": answer.model_dump()}
                cache_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return answer

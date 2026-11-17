import re
import os
import tempfile
import numpy as np
from scipy.io import wavfile
from typing import Optional

from allosaurus.app import read_recognizer
from pydantic import BaseModel

from phoneme_modelling.evaluation.evaluator import AudioEvaluator
from phoneme_modelling.phonemes import (
    ALL_ACCEPTED_IPA_STRINGS,
    ALTERNATIVE_IPA,
    IPA_TO_TEXT,
    VALID_IPA_STRINGS,
)
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluation
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.util.ipa import clean_ipa
from phoneme_modelling.util.loaders import load_as_numpy_array



class AllosaurusEvaluatorConfig(BaseModel):
    model_name: str = "latest"  # Use "latest" (uni2005) or "eng2102"
    lang_id: str = "eng"        # Restricts decoding to English phone inventory
    emit: float = 1.2           # >1.0 encourages phone emission on very short clips
    topk: int = 5               # Inspect top-k phone candidates per emitted frame


class AllosaurusOutput():

    def __init__(self, raw_output: str):
        self.__phonemes: list[list[tuple[str, float]]] = []
        frames = raw_output.split("|")
        for frame in frames:
            phoneme_candidates = self._process_frame(frame)
            self.__phonemes.append(phoneme_candidates)

    def _process_frame(self, frame: str) -> list[tuple[str, float]]:
        matches = re.findall(r"(\S+)\s+\(([0-9.]+)\)", frame)
        if not matches:
            return []
        phoneme_candidates = []
        for match, probability_str in matches:
            phoneme_candidates.append((match, float(probability_str)))
        return phoneme_candidates

    def get_most_likely_limited_to(self, acceptable_phonemes: set[str]) -> str:
        used_phonemes = []
        for phoneme_candidates in self.__phonemes:
            most_likely, _ = self._pick_most_likely_acceptable(phoneme_candidates, acceptable_phonemes)
            if most_likely is not None:
                used_phonemes.append(most_likely)
        return "".join(used_phonemes)

    def _pick_most_likely_acceptable(
        self,
        candidates: list[tuple[str, float]],
        acceptable_phonemes: set[str]
    ) -> tuple[str, float] | tuple[None, None]:
        for candidate, prob in candidates:
            if candidate in acceptable_phonemes:
                return candidate, prob
        return None, None

    def get_most_likely_single_phoneme(self, acceptable_phonemes: set[str]) -> str:
        max_likely = ""
        max_prob = 0
        for phoneme_candidates in self.__phonemes:
            most_likely, prob = self._pick_most_likely_acceptable(phoneme_candidates, acceptable_phonemes)
            if most_likely is None or prob is None:
                continue
            if prob > max_prob:
                max_prob = prob
                max_likely = most_likely
        return max_likely



class AllosaurusEvaluator(AudioEvaluator):

    def __init__(self, config: Optional[AllosaurusEvaluatorConfig] = None):
        self.__config = config or AllosaurusEvaluatorConfig()
        self.__model = read_recognizer(self.__config.model_name)

    @staticmethod
    def get_name() -> str:
        return "allosaurus"

    def prepare_clean_16k_pcm_wav(self, file_path: str) -> str:
        audio = load_as_numpy_array(file_path, 16000)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = (audio / peak) * 0.35
        pad = np.zeros(4000, dtype=np.float32)
        padded_audio = np.concatenate([pad, audio, pad])
        pcm_int16 = np.clip(padded_audio * 32767, -32768, 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(tmp.name, 16000, pcm_int16)
        return tmp.name

    def _normalize_phoneme(self, phoneme: str) -> str:
        phoneme = phoneme.strip()
        if phoneme in ALTERNATIVE_IPA:
            return ALTERNATIVE_IPA[phoneme]
        cleaned = clean_ipa(phoneme)
        if cleaned in ALTERNATIVE_IPA:
            return ALTERNATIVE_IPA[cleaned]
        stripped = re.sub(r"[ʰː̩̪̠ʲ̚]", "", cleaned).replace("ɡ", "g")
        return ALTERNATIVE_IPA.get(stripped, stripped)

    def evaluate(self, audio_info: SyntheticPhonemeMetadata) -> PhonemeEvaluation:
        temp_name = self.prepare_clean_16k_pcm_wav(audio_info.file_path)
        try:
            raw_output = self.__model.recognize(
                temp_name,
                lang_id=self.__config.lang_id,
                topk=self.__config.topk,
                emit=self.__config.emit,
            )
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)
        output_obj = AllosaurusOutput(raw_output)
        detected_ipa = output_obj.get_most_likely_single_phoneme(ALL_ACCEPTED_IPA_STRINGS)
        detected_text = IPA_TO_TEXT.get(detected_ipa) if detected_ipa else None
        is_correct = detected_ipa == audio_info.actual_phoneme_ipa
        return PhonemeEvaluation(
            is_correct=is_correct,
            detected_phoneme_text=detected_text,
            detected_phoneme_ipa=detected_ipa,
            detected_raw=str(raw_output)
        )
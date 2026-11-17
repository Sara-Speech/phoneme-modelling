from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.util.ipa import clean_ipa
from phoneme_modelling.util.loaders import load_as_numpy_array
from phoneme_modelling.evaluation.evaluator import AudioEvaluator
from phoneme_modelling.phonemes import IPA_TO_TEXT
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluation


W2V2_REPO = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
W2V2_SAMPLE_RATE = 16000



class Wav2Vec2Evaluator(AudioEvaluator):

    def __init__(self, config: Optional[BaseModel] = None):
        self.__processor = Wav2Vec2Processor.from_pretrained(W2V2_REPO)
        self.__model = Wav2Vec2ForCTC.from_pretrained(W2V2_REPO).eval()
        pass

    @staticmethod
    def get_name() -> str:
        return "wav2vec2"

    def evaluate(self, audio_info: SyntheticPhonemeMetadata) -> PhonemeEvaluation:
        audio = load_as_numpy_array(audio_info.file_path, W2V2_SAMPLE_RATE)
        with torch.no_grad():
            encodings = self.__processor.feature_extractor(audio, sampling_rate=W2V2_SAMPLE_RATE, return_tensors="pt")
            logits = self.__model(encodings.input_values).logits
            detected_raw = self.__processor.batch_decode(logits.argmax(-1))[0].strip()
        detected_ipa = clean_ipa(detected_raw)
        is_correct = detected_ipa == audio_info.actual_phoneme_ipa
        detected_text = None
        if detected_ipa in IPA_TO_TEXT:
            detected_text = IPA_TO_TEXT.get(detected_ipa)
        return PhonemeEvaluation(
            is_correct=is_correct,
            detected_phoneme_text=detected_text,
            detected_phoneme_ipa=detected_ipa,
            detected_raw=detected_raw
        )

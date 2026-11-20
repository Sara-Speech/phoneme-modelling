import os
import io
from pathlib import Path

import librosa
import numpy as np
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from phoneme_modelling.schema.sara_model_evaluator_config import SaraModelEvaluatorConfig
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.util.aws import download_s3_path, list_s3_path
from phoneme_modelling.util.ipa import clean_ipa
from phoneme_modelling.util.loaders import load_as_numpy_array
from phoneme_modelling.evaluation.evaluator import AudioEvaluator
from phoneme_modelling.phonemes import IPA_TO_TEXT
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluation


SAMPLE_RATE = 16000
MAX_SECONDS = 5
MAX_LEN = SAMPLE_RATE * MAX_SECONDS



class SaraModelEvaluator(AudioEvaluator):

    def __init__(self, config: SaraModelEvaluatorConfig):
        if not os.path.exists(config.local_path):
            download_s3_path(config.remote_path, config.local_path)
        try:
            self.load_model(config.local_path)
        except:
            download_s3_path(config.remote_path, config.local_path)
            self.load_model(config.local_path)

    def load_model(self, model_path: Path):
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.__model: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(model_path)
        torch.nn.Module.to(self.__model, self.__device)
        self.__model.eval()

    @staticmethod
    def get_name() -> str:
        return "sara"

    def _load_audio(self, file_path: str) -> np.ndarray:
        audio_data, sample_rate = sf.read(str(file_path))
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sample_rate != SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
        return audio_data

    def _pad_or_trim(self, input_values: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_values.shape
        if seq_len == MAX_LEN:
            return input_values
        if seq_len > MAX_LEN:
            return input_values[:, :MAX_LEN]
        return torch.nn.functional.pad(
            input_values, (0, MAX_LEN - seq_len), mode="constant", value=0.0
        )

    def evaluate(self, audio_info: SyntheticPhonemeMetadata) -> PhonemeEvaluation:
        audio = self._load_audio(audio_info.file_path)
        inputs = self.__processor(
            audio,
            sampling_rate=SAMPLE_RATE, # type: ignore
            return_tensors="pt", # type: ignore
            padding=True, # type: ignore
        )
        input_values = self._pad_or_trim(inputs.input_values).to(self.__device)
        with torch.no_grad():
            logits = self.__model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        detected_raw = self.__processor.batch_decode(predicted_ids)[0]
        detected_ipa = clean_ipa(detected_raw)
        is_correct = detected_ipa == audio_info.actual_phoneme_ipa
        detected_text = IPA_TO_TEXT.get(detected_ipa)
        return PhonemeEvaluation(
            is_correct=is_correct,
            detected_phoneme_text=detected_text,
            detected_phoneme_ipa=detected_ipa,
            detected_raw=detected_raw,
        )

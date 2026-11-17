import numpy as np
from pathlib import Path
from typing import Optional, cast

import torch
from onnxruntime import InferenceSession
from huggingface_hub import hf_hub_download
from lhotse.features.kaldi.extractors import Fbank, FbankConfig
from pydantic import BaseModel

from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.util.ipa import clean_ipa
from phoneme_modelling.util.loaders import load_as_numpy_array
from phoneme_modelling.evaluation.evaluator import AudioEvaluator
from phoneme_modelling.phonemes import IPA_TO_TEXT
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluation


ZIPA_REPO = "anyspeech/zipa-large-crctc-500k"
ZIPA_SAMPLE_RATE = 16000



class ZipaEvaluator(AudioEvaluator):

    def __init__(self, config: Optional[BaseModel] = None):
        self.__session = InferenceSession(hf_hub_download(ZIPA_REPO, "model.onnx"))
        self.__vocab = {}
        for line in open(hf_hub_download(ZIPA_REPO, "tokens.txt"), encoding="utf-8"):
            parts = line.split()
            if parts:
                self.__vocab[int(parts[1]) if len(parts) > 1 else len(self.__vocab)] = parts[0]
        self.__fbank = Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))
        pass

    @staticmethod
    def get_name() -> str:
        return "zipa"

    def get_audio_ipa(self, audio: np.ndarray) -> str:
        audio_tensor_batch = [torch.from_numpy(audio).float().unsqueeze(0)]
        extracted = self.__fbank.extract_batch(audio_tensor_batch, sampling_rate=ZIPA_SAMPLE_RATE)[0]
        if hasattr(extracted, "to_dense"):
            feat: torch.Tensor = cast(torch.Tensor, extracted.to_dense()) # type: ignore[attr-defined]
        else:
            feat: torch.Tensor = torch.as_tensor(extracted)
        feat = feat.unsqueeze(0)
        model_inputs = {"x": feat.numpy(), "x_lens": np.array([feat.shape[1]], dtype=np.int64)}
        model_outputs = cast(list[np.ndarray], self.__session.run(None, model_inputs))
        logp = model_outputs[0][0]
        out, prev = [], -1
        for i in logp.argmax(-1):  # greedy CTC, blank = 0
            if i != 0 and i != prev:
                out.append(self.__vocab.get(int(i), ""))
            prev = i
        return " ".join(out)

    def evaluate(self, audio_info: SyntheticPhonemeMetadata) -> PhonemeEvaluation:
        audio = load_as_numpy_array(audio_info.file_path, ZIPA_SAMPLE_RATE)
        detected_raw = self.get_audio_ipa(audio)
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

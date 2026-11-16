from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from phoneme_modelling.evaluation.evaluator import AudioEvaluator
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluation
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata



class DebugEvaluator(AudioEvaluator):

    def __init__(self, config: Optional[BaseModel] = None):
        # You can create a schema for the evaluator that extends pydantic's base model, then it will expect the config
        # json to have those fields for the evaluator and automatically send you the config data. You just have to add
        # the schema together with the class when registering the evaluator with the factory.
        pass

    @staticmethod
    def get_name() -> str:
        return "debug"

    def evaluate(self, audio_info: SyntheticPhonemeMetadata) -> PhonemeEvaluation:
        debug_text = (
            f"Running debug evaluator for target_phoneme: {audio_info.actual_phoneme_text}, target_phoneme_ipa: "
            f"{audio_info.actual_phoneme_ipa}, audio_path: {audio_info.file_path}"
        )
        print(debug_text)
        return PhonemeEvaluation(
            is_correct=True,
            detected_phoneme_text=audio_info.actual_phoneme_text, # Optional
            detected_phoneme_ipa=audio_info.actual_phoneme_ipa, # Optional
            quality=0.9 # Optional
        )
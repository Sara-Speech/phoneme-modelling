from typing import Optional

from pydantic import BaseModel



class PhonemeEvaluation(BaseModel):
    is_correct: bool
    detected_phoneme_text: Optional[str] = None
    detected_phoneme_ipa: Optional[str] = None
    quality: Optional[float] = None


class PhonemeEvaluationOutput(BaseModel):
    file_name: str
    evaluation: str
    file_path: str
    target_phoneme_text: str
    actual_phoneme_text: str
    actual_phoneme_ipa: str
    is_correct: bool
    detected_phoneme_text: Optional[str] = None
    detected_phoneme_ipa: Optional[str] = None
    quality: Optional[float] = None
    model: str
    voice: str
    is_distorted: bool = False
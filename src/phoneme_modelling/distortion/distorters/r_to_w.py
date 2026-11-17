import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import shift_formants
from phoneme_modelling.util.acoustics.clip import Clip


class RToWDistorter(PairDistorter):
    """r said as w: lowers F2, raises F3 and raises F1 slightly, by amounts fitted on 2026-11-15 so strength 1 of
    Fenrir's r (10-25 take 1) lands on Fenrir's own w (F1/F2/F3 within 3%)."""

    FROM_PHONEME = "r"
    TO_PHONEME = "w"

    @staticmethod
    def get_name() -> str:
        return "r_to_w"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return shift_formants(clip, measures, alpha, self.parameters, formants=(1, 2, 3))


class RToWMatchedDistorter(RToWDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "r_to_w_matched"

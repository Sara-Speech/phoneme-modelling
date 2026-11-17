import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import change_stop
from phoneme_modelling.util.acoustics.clip import Clip


class PToBDistorter(PairDistorter):
    """p said as b: adds prevoicing and cuts the release toward b, with targets fitted on 2026-11-15 so strength 1 of
    Fenrir's p (10-25 take 1) measures like Fenrir's own b (85 ms prevoicing, 19 ms release). Burst place stays."""

    FROM_PHONEME = "p"
    TO_PHONEME = "b"

    @staticmethod
    def get_name() -> str:
        return "p_to_b"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return change_stop(clip, measures, alpha, self.parameters, voice_f0, rng, voicing="on", place=False)


class PToBMatchedDistorter(PToBDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "p_to_b_matched"

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from phoneme_modelling.distortion.distorters.ch_to_sh import ChToShDistorter
from phoneme_modelling.distortion.distorters.r_distortion import RDistortionDistorter
from phoneme_modelling.distortion.distorters.s_to_z import SToZDistorter
from phoneme_modelling.distortion.pair_registry import PAIR_DISTORTERS
from phoneme_modelling.distortion.pipeline import AudioDistortionPipeline
from phoneme_modelling.phonemes import TEXT_TO_IPA
from phoneme_modelling.schema.phoneme_pipeline_config import (
    DistortionApplicationConfig,
    PhonemePipelineConfig,
    PhonemeValidationConfig,
)
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.tuning import report
from phoneme_modelling.tuning.takes import check_takes, consonants_only, take_number

# The synthetic sounds the distortion tests use, loaded under another name so the two conftest files don't clash.
_spec = importlib.util.spec_from_file_location(
    "distortion_sounds", Path(__file__).parents[1] / "distortion" / "conftest.py"
)
_sounds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sounds)
rng = _sounds.rng
sound_files = _sounds.sound_files

NAMES = [cls.get_name() for cls in PAIR_DISTORTERS]


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("0005_s_fenrir_take1_s_to_t_distortion_1.wav", "s_to_t"),
        ("0010_th_voiceless_fenrir_th_voiceless_to_t_distortion_2.wav", "th_voiceless_to_t"),
        ("0123_th_voiced_fenrir_take2_th_voiced_to_d_matched_distortion_3.wav", "th_voiced_to_d_matched"),
        ("0042_r_fenrir_take10_r_distortion_matched_distortion_1.wav", "r_distortion_matched"),
        ("0012_s_fenrir_take3.wav", None),
    ],
)
def test_distorter_is_read_from_the_file_name(file_name: str, expected: str | None) -> None:
    assert report.distorter_of(file_name, NAMES) == expected


def test_pairs_are_named_in_plain_words() -> None:
    assert report.folder_name(3, RDistortionDistorter) == "03_distorted-r"
    assert report.mistake(SToZDistorter) == "s (as in sip) said as z (as in zip)"
    assert report.describe(SToZDistorter) == ("reshapes the hiss, then adds voicing.", "")
    assert "needs work" in report.describe(ChToShDistorter)[1]


def test_every_pair_appears_once_with_r_first() -> None:
    order = report.pair_order()
    assert len(order) == len({cls.get_name() for cls in order}) == 57
    assert all(cls.FROM_PHONEME == "r" for cls in order[:4])


@pytest.mark.parametrize(("heard", "target", "same"), [("sə", "s", True), ("ʧ", "tʃ", True), ("zə", "s", False)])
def test_right_consonant_ignores_vowels_and_spelling(heard: str, target: str, same: bool) -> None:
    assert (consonants_only(heard) == consonants_only(target)) is same


def test_takes_are_kept_when_the_recognizer_hears_the_right_sound() -> None:
    clean = pd.DataFrame(
        {
            "file_name": ["0000_s_fenrir_take1.wav", "0001_s_fenrir_take10.wav"],
            "target_phoneme_text": ["s", "s"],
            "voice": ["fenrir_take1", "fenrir_take10"],
        }
    )
    evaluation = pd.DataFrame(
        {
            "file_name": clean["file_name"],
            "evaluation": "wav2vec2_espeak",
            "detected_phoneme_ipa": ["sə", "ʃ"],
        }
    )
    sheet = check_takes(clean, evaluation)
    assert list(sheet["take"]) == [1, 10]
    assert list(sheet["keep"]) == [True, False]
    assert take_number("fenrir") == 1


def test_pack_has_every_file_for_a_pair_and_marks_what_is_missing(sound_files: dict[str, Path], tmp_path: Path) -> None:
    rows = [
        SyntheticPhonemeMetadata(
            file_name=sound_files[key].name,
            file_path=str(sound_files[key]),
            model="gemini",
            voice="kore",
            target_phoneme_text=key,
            actual_phoneme_text=key,
            actual_phoneme_ipa=TEXT_TO_IPA[key],
            distortion_intensity=0.0,
            duration=1.0,
        ).model_dump()
        for key in ("r", "w")
    ]
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    config = PhonemePipelineConfig(
        number_of_digits=3,
        phonemes_to_generate=["r"],
        distortion_application_config=DistortionApplicationConfig(distortion_intensitites=[0.3, 0.6, 1.0]),
        validation_config=PhonemeValidationConfig(),
        generator_configs={},
        distorter_configs={"r_to_w": None, "r_to_w_matched": None},
    )
    AudioDistortionPipeline.run(config, tmp_path / "phonemes", metadata_path, current_audio_count=2)
    takes = pd.DataFrame({"file_name": [r["file_name"] for r in rows], "sound": ["r", "w"], "take": 1, "keep": True})

    out = tmp_path / "pack"
    sheet = report.build_pack(pd.read_csv(metadata_path), takes, out).set_index("pair")

    folder = out / sheet.loc["r_to_w", "folder"]
    assert sheet.loc["r_to_w", "status"] == "ok"
    assert {path.name for path in folder.glob("*.wav")} == {
        "r-said-as-w_A_correct-r.wav",
        "r-said-as-w_B_strength-0.3.wav",
        "r-said-as-w_C_strength-0.6.wav",
        "r-said-as-w_D_strength-1.wav",
        "r-said-as-w_E_real-w-same-voice.wav",
    }
    assert {path.name for path in (folder / "matched").glob("*.wav")} == {
        "r-said-as-w_matched_B_strength-0.3.wav",
        "r-said-as-w_matched_C_strength-0.6.wav",
        "r-said-as-w_matched_D_strength-1.wav",
    }
    assert "r_to_l at 0.3" in sheet.loc["r_to_l", "status"]  # never distorted in this test
    assert sheet.loc["s_to_z", "status"] == "blocked: no take of s"
    assert (out / sheet.loc["s_to_z", "folder"] / "BLOCKED.txt").is_file()

    covered = report.coverage(sheet.reset_index(), takes)
    assert len(covered) == 57 + len(report.SWAP_ONLY) + len(report.NOT_PAIR_EDITS)

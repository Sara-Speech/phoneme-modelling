"""Which clean takes the distortions start from.

A take is kept when the recognizer hears the right consonant: the detected sounds, with vowels and marks dropped and
repeats collapsed, are exactly the target sound (the 11-07 "right consonant" rule, so "sə" counts as "s"). The result
is written to a takes sheet with a ``keep`` column that a listener can overrule by hand.

    uv run python -m phoneme_modelling.tuning.takes clean-metadata --metadata data/fenrir/metadata.csv \
        --out data/fenrir/clean_metadata.csv
    uv run python src/phoneme_modelling/generate_synthetic_audio.py --config_path config/fenrir.json --mode eval \
        --output_metadata_path data/fenrir/clean_metadata.csv --evaluation_output_path data/fenrir/evaluation_clean.csv
    uv run python -m phoneme_modelling.tuning.takes check --metadata data/fenrir/clean_metadata.csv \
        --evaluation data/fenrir/evaluation_clean.csv --out data/fenrir/takes.csv
"""

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from phoneme_modelling.phonemes import TEXT_TO_IPA
from phoneme_modelling.util.acoustics import dsp
from phoneme_modelling.util.acoustics.delivery import SAMPLE_RATE, as_clip
from phoneme_modelling.util.loaders import load_as_numpy_array

VOWELS = set("aeiouyæɑɒɔəɛɜɪʊʌɐɤɯøœɨʉɘɵɞɶʏ")
VARIANTS = {"ʧ": "tʃ", "ʤ": "dʒ", "ɡ": "g", "ɝ": "ɹ", "ɚ": "ɹ", "r": "ɹ"}
CHECK_EVALUATOR = "wav2vec2_espeak"
TAKE = re.compile(r"_take(\d+)$")
SOUND_TEXTS = sorted(text for text in TEXT_TO_IPA if text.islower())
MAX_BACKGROUND_DB = -20.0  # a take's background must be at least this far below the sound


def consonants_only(ipa: object) -> str:
    """Drop vowels and diacritics, map variant symbols, collapse repeats: 'bə' -> 'b', 'ʃʃ' -> 'ʃ', 'pʰa' -> 'p'."""
    text = ipa if isinstance(ipa, str) else ""
    for variant, standard in VARIANTS.items():
        text = text.replace(variant, standard)
    kept = [c for c in text if c not in VOWELS and unicodedata.category(c) not in ("Mn", "Lm") and c not in "ː "]
    collapsed: list[str] = []
    for c in kept:
        if not collapsed or collapsed[-1] != c:
            collapsed.append(c)
    return "".join(collapsed)


def take_number(voice: str) -> int:
    """The take in a pipeline voice name: ``fenrir_take3`` is take 3, a plain ``fenrir`` is take 1."""
    match = TAKE.search(voice)
    return int(match.group(1)) if match else 1


def clean_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    return metadata[metadata["distortion_intensity"] == 0].copy()


def background_db(path: Path) -> float:
    """Level of everything around the sound, relative to the sound (dB). Every edit and listening clip is placed in
    noise copied from this background, so a take with a loud one (Gemini sometimes adds other sounds) fills the
    whole 2.5 s clip with that noise."""
    samples = load_as_numpy_array(path, SAMPLE_RATE).astype(np.float64)
    first, last = as_clip(samples).speech_bounds()
    guard = dsp.to_samples(SAMPLE_RATE, 20)
    around = np.concatenate([samples[: max(0, first - guard)], samples[last + guard :]])
    if len(around) < dsp.to_samples(SAMPLE_RATE, 50):
        return float("-inf")
    return float(20 * np.log10(dsp.rms(around) / (dsp.rms(samples[first:last]) + 1e-12) + 1e-12))


def check_takes(clean: pd.DataFrame, evaluation: pd.DataFrame, evaluator: str = CHECK_EVALUATOR) -> pd.DataFrame:
    """One row per clean take: what the recognizer heard, how loud its background is, and whether it is kept (the
    right consonant and a quiet background)."""
    heard = evaluation[evaluation["evaluation"] == evaluator].set_index("file_name")
    rows = []
    for row in clean.itertuples():
        detected = heard["detected_phoneme_ipa"].get(row.file_name) if row.file_name in heard.index else None
        target = TEXT_TO_IPA[row.target_phoneme_text]
        right = consonants_only(detected) == consonants_only(target)
        background = background_db(Path(row.file_path)) if "file_path" in clean.columns else float("-inf")
        rows.append(
            {
                "file_name": row.file_name,
                "sound": row.target_phoneme_text,
                "take": take_number(row.voice),
                "voice": row.voice,
                "heard": detected if isinstance(detected, str) else "",
                "right_consonant": right,
                "background_db": round(background, 1),
                "keep": right and background <= MAX_BACKGROUND_DB,
                "listener_note": "",
            }
        )
    return pd.DataFrame(rows).sort_values(["sound", "take"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    clean = commands.add_parser("clean-metadata", help="copy only the clean rows, for an evaluation of the takes")
    clean.add_argument("--metadata", type=Path, required=True)
    clean.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("check", help="write the takes sheet from the recognizer's results")
    check.add_argument("--metadata", type=Path, required=True)
    check.add_argument("--evaluation", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--force", action="store_true", help="overwrite a sheet that may hold hand edits")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    if args.command == "clean-metadata":
        clean_rows(metadata).to_csv(args.out, index=False)
        return
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists and may hold hand edits; pass --force to overwrite it")
    sheet = check_takes(clean_rows(metadata), pd.read_csv(args.evaluation))
    sheet.to_csv(args.out, index=False)
    kept = sheet[sheet["keep"]].groupby("sound").size().reindex(SOUND_TEXTS, fill_value=0)
    print(f"kept {int(sheet['keep'].sum())} of {len(sheet)} takes")
    print(kept.to_string())


if __name__ == "__main__":
    main()

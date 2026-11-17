"""Phase 1 hand-over: every pair on one voice, a sheet to mark by ear, and which error types are covered.

The pack has one folder per pair, named in plain words (``01_r-said-as-w``):

    r-said-as-w_A_correct-r.wav             the take every edit starts from
    r-said-as-w_B_strength-0.3.wav          about 30% of the way to the mistake
    r-said-as-w_C_strength-0.6.wav          about 60% of the way
    r-said-as-w_D_strength-1.wav            all the way
    r-said-as-w_E_real-w-same-voice.wav     the same voice really saying the other sound
    matched/                    B to D again, with the length, pitch and level of a correct clip

A and E are put in the same delivery format as the edits (16 kHz, at least 2.5 s, in noise), so the only difference
you hear is the edit. Each pair uses the first kept take of its sound (see ``takes``), or its first take, marked
unconfirmed in the sheet, when the recognizer kept none.

    uv run python -m phoneme_modelling.tuning.report --metadata data/fenrir/metadata.csv \
        --takes data/fenrir/takes.csv --out-dir ~/Downloads/11-15-2026_Fenrir_All_57_Pairs \
        --sheet data/fenrir/listening_sheet.csv --coverage data/fenrir/coverage.csv
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from phoneme_modelling.distortion.pair_distorter import INTENSITIES, PARAMETERS_FOLDER, PairDistorter
from phoneme_modelling.distortion.pair_registry import PAIR_DISTORTERS
from phoneme_modelling.util.acoustics.delivery import OUTPUT_PEAK, SAMPLE_RATE, as_clip, deliver, save_wav, stable_rng
from phoneme_modelling.util.loaders import load_as_numpy_array

SOUND_WORDS = {
    "r": ("r", "red"),
    "w": ("w", "wet"),
    "l": ("l", "leg"),
    "y": ("y", "yes"),
    "s": ("s", "sip"),
    "z": ("z", "zip"),
    "th_voiced": ("voiced th", "this"),
    "th_voiceless": ("voiceless th", "thin"),
    "sh": ("sh", "shoe"),
    "ch": ("ch", "chip"),
    "j": ("j", "jam"),
    "f": ("f", "fan"),
    "v": ("v", "van"),
    "p": ("p", "pat"),
    "b": ("b", "bat"),
    "t": ("t", "top"),
    "d": ("d", "dog"),
    "k": ("k", "cat"),
    "g": ("g", "go"),
    "m": ("m", "map"),
    "n": ("n", "nap"),
    "ng": ("ng", "sing"),
    "h": ("h", "hat"),
}
FAMILY_ORDER = [
    "formant",
    "fricative",
    "fric_to_stop",
    "fric_to_affricate",
    "affricate_to_fric",
    "affricate_to_stop",
    "stop",
    "nasal_place",
    "nasal_to_stop",
]
COMPARISON_SOUND = {"r_distortion": "w"}  # distorted r is part way to w, so w is the sound to compare with
STRENGTH_FILES = {0.3: "B_strength-0.3", 0.6: "C_strength-0.6", 1.0: "D_strength-1"}
SWAP_ONLY = [("r", "h"), ("r", "d"), ("r", "n"), ("r", "z"), ("g", "w"), ("g", "y"), ("t", "th_voiceless"), ("m", "w")]
NOT_PAIR_EDITS = [
    ("j said as ts", "not possible: there is no ts recording, and the endpoint cannot output it"),
    ("omissions (a sound left out)", "not a pair edit: for a sound on its own, leaving it out is silence"),
    ("other_substitution (any unlisted swap, e.g. s said as sh)", "not a pair edit: the endpoint's catch-all label"),
    ("no_sound_detected, wrong_target, multiple_targets", "not a pair edit: outcomes of the whole recording"),
]
PLAIN = [cls for cls in PAIR_DISTORTERS if not cls.MATCH_NUISANCE]
MATCHED = {cls.get_name().removesuffix("_matched"): cls for cls in PAIR_DISTORTERS if cls.MATCH_NUISANCE}


def name_of(sound: str) -> str:
    return SOUND_WORDS[sound][0]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def mistake(cls: type[PairDistorter]) -> str:
    """The pair in plain words, with example words: 'r (as in red) said as w (as in wet)'."""
    source, _ = SOUND_WORDS[cls.FROM_PHONEME]
    if cls.get_name() in COMPARISON_SOUND:
        return f"distorted {source} (as in {SOUND_WORDS[cls.FROM_PHONEME][1]}), part way to {COMPARISON_SOUND[cls.get_name()]}"
    target, target_word = SOUND_WORDS[cls.TO_PHONEME]
    return f"{source} (as in {SOUND_WORDS[cls.FROM_PHONEME][1]}) said as {target} (as in {target_word})"


def folder_name(number: int, cls: type[PairDistorter]) -> str:
    if cls.get_name() in COMPARISON_SOUND:
        return f"{number:02d}_distorted-{slug(name_of(cls.FROM_PHONEME))}"
    return f"{number:02d}_{slug(name_of(cls.FROM_PHONEME))}-said-as-{slug(name_of(cls.TO_PHONEME))}"


def describe(cls: type[PairDistorter]) -> tuple[str, str]:
    """(what the edit does, the 10-25 flag) from the distorter's docstring."""
    text = " ".join((cls.__doc__ or "").split())
    _, _, after_label = text.partition(": ")
    edit, _, flag = after_label.partition(". ")
    return edit.rstrip(".") + ".", flag


def family(cls: type[PairDistorter]) -> str:
    return json.loads((PARAMETERS_FOLDER / f"{cls.get_name()}.json").read_text(encoding="utf-8"))["family"]


def pair_order() -> list[type[PairDistorter]]:
    """Grouped by the kind of edit, as in the README, with the r pairs first."""
    return sorted(PLAIN, key=lambda cls: (FAMILY_ORDER.index(family(cls)), cls.FROM_PHONEME != "r", cls.get_name()))


def distorter_of(file_name: str, names: list[str]) -> str | None:
    """The distorter that made a clip, from its name ``{n}_{sound}_{voice}_{distorter}_distortion_{k}.wav``.
    None for a clean clip."""
    stem = file_name.removesuffix(".wav")
    for name in sorted(names, key=len, reverse=True):
        if re.search(rf"_{re.escape(name)}_distortion_\d+$", stem):
            return name
    return None


def seed_of(file_name: str) -> str:
    """A clean clip's name without its counter: ``0012_s_fenrir_take3.wav`` -> ``s_fenrir_take3``."""
    return file_name.removesuffix(".wav").partition("_")[2]


def pick_take(takes: pd.DataFrame, sound: str) -> tuple[str | None, bool]:
    """The first kept take of ``sound``, or, when none was kept, the take with the quietest background, marked as
    unconfirmed. (None, False) when the voice has no take of the sound at all."""
    of_sound = takes[takes["sound"] == sound].sort_values("take")
    kept = of_sound[of_sound["keep"].astype(bool)]
    if not kept.empty:
        return str(kept.iloc[0]["file_name"]), True
    if "background_db" in of_sound.columns:
        of_sound = of_sound.sort_values(["background_db", "take"])
    return (str(of_sound.iloc[0]["file_name"]), False) if not of_sound.empty else (None, False)


def complete_take(metadata: pd.DataFrame, takes: pd.DataFrame, cls: type[PairDistorter]) -> tuple[str | None, bool]:
    """The take of the pair's source sound to show: the first kept take with every edit made (plain and matched, at
    every strength), else the kept take with the most, else ``pick_take``'s choice."""
    of_sound = takes[(takes["sound"] == cls.FROM_PHONEME) & takes["keep"].astype(bool)].sort_values("take")
    best, most = None, -1
    for name in of_sound["file_name"]:
        seed = seed_of(name)
        made = len(edits_of(metadata, seed, cls.get_name())) + len(
            edits_of(metadata, seed, MATCHED[cls.get_name()].get_name())
        )
        if made > most:
            best, most = name, made
        if made == 2 * len(INTENSITIES):
            break
    return (best, True) if best is not None else pick_take(takes, cls.FROM_PHONEME)


def edits_of(metadata: pd.DataFrame, seed: str, name: str) -> dict[float, Path]:
    """The clips ``name`` made from the clean take ``seed``, by strength."""
    pattern = re.compile(rf"^\d+_{re.escape(seed)}_{re.escape(name)}_distortion_\d+\.wav$")
    found = {}
    for row in metadata.itertuples():
        if pattern.match(row.file_name):
            found[round(float(row.distortion_intensity), 2)] = Path(row.file_path)
    return found


def deliver_clean(source: Path, target: Path) -> None:
    """A clean take in the delivery format, the way PairDistorter delivers an edit (same level, same kind of noise)."""
    samples = load_as_numpy_array(source, SAMPLE_RATE).astype(np.float64)
    gain = OUTPUT_PEAK / (float(np.max(np.abs(samples))) + 1e-12)
    save_wav(target, deliver(samples, as_clip(samples), stable_rng(source.name, "listening")) * gain)


def build_pack(
    metadata: pd.DataFrame, takes: pd.DataFrame, out_dir: Path, pairs: list[str] | None = None
) -> pd.DataFrame:
    """Writes one folder per pair (only ``pairs``, if given, keeping their usual numbers) and returns the sheet."""
    paths = dict(zip(metadata["file_name"], metadata["file_path"], strict=True))
    rows = []
    for number, cls in enumerate(pair_order(), start=1):
        if pairs is not None and cls.get_name() not in pairs:
            continue
        name = cls.get_name()
        folder = out_dir / folder_name(number, cls)
        # Every file name carries its pair: audio players and previews that cache by file name would otherwise play
        # one folder's B_strength-0.3.wav for every folder.
        tag = folder.name.partition("_")[2]
        folder.mkdir(parents=True, exist_ok=True)
        edit, flag = describe(cls)
        compare = COMPARISON_SOUND.get(name, cls.TO_PHONEME)
        missing = []
        source, source_confirmed = complete_take(metadata, takes, cls)
        real, real_confirmed = pick_take(takes, compare)
        real_path = Path(paths[real]) if real is not None else None
        if source is None:
            reason = f"blocked: no take of {name_of(cls.FROM_PHONEME)}"
            (folder / "BLOCKED.txt").write_text(f"{reason}. Nothing could be made for this pair.\n", encoding="utf-8")
            missing.append(reason)
        else:
            deliver_clean(Path(paths[source]), folder / f"{tag}_A_correct-{slug(name_of(cls.FROM_PHONEME))}.wav")
            for variant, subfolder in ((name, folder), (MATCHED[name].get_name(), folder / "matched")):
                made = edits_of(metadata, seed_of(source), variant)
                for strength in INTENSITIES:
                    clip = made.get(round(strength, 2))
                    if clip is None or not clip.is_file():
                        missing.append(f"{variant} at {strength:g}")
                        continue
                    subfolder.mkdir(exist_ok=True)
                    matched = "_matched" if subfolder != folder else ""
                    shutil.copyfile(clip, subfolder / f"{tag}{matched}_{STRENGTH_FILES[strength]}.wav")
            if real is None:
                missing.append(f"no take of {name_of(compare)} to compare with")
            else:
                deliver_clean(real_path, folder / f"{tag}_E_real-{slug(name_of(compare))}-same-voice.wav")
        rows.append(
            {
                "number": number,
                "folder": folder.name,
                "pair": name,
                "mistake": mistake(cls),
                "what_the_edit_does": edit,
                "flag_from_10_25": flag,
                "source_take": source or "",
                "source_take_confirmed": source_confirmed,
                "real_take": real or "",
                "real_take_confirmed": real_confirmed,
                "status": "ok" if not missing else "; ".join(missing),
                "by_ear_0.3": "",
                "by_ear_0.6": "",
                "by_ear_1": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def coverage(sheet: pd.DataFrame, takes: pd.DataFrame) -> pd.DataFrame:
    """Every endpoint error type and what covers it on this voice."""
    rows = [
        {
            "group": "graded pair (edited clips)",
            "error": row.mistake,
            "covered_by": f"pack folder {row.folder}" if not row.status.startswith("blocked") else "",
            "note": "" if row.status == "ok" else row.status,
        }
        for row in sheet.itertuples()
    ]
    for source, target in SWAP_ONLY:
        real, _ = pick_take(takes, target)
        rows.append(
            {
                "group": "swap-only pair (no edit family)",
                "error": f"{name_of(source)} said as {name_of(target)}",
                "covered_by": f"real clip {real}" if real else "",
                "note": "" if real else f"no take of {name_of(target)}",
            }
        )
    rows += [
        {"group": "not a pair edit", "error": error, "covered_by": "", "note": note} for error, note in NOT_PAIR_EDITS
    ]
    return pd.DataFrame(rows)


PACK_README = """# Every mistake on one voice ({voice}), three strengths each

Each folder is one mistake. Play its files in order:

| File | What you hear |
|---|---|
| **A** correct | The sound said correctly. Every edit below starts from this recording |
| **B** strength 0.3 | Edited about 30% of the way toward the mistake: a slight version |
| **C** strength 0.6 | About 60% of the way |
| **D** strength 1 | All the way |
| **E** real | The same voice really saying the other sound, for comparison |

D should sound like E. B and C should sound like A turning into E.
`matched/` holds B to D again with the length, pitch and loudness of a correct clip, so a model cannot tell them apart
from correct clips by those alone.

Mark each folder in `listening_sheet.csv`:

- `by_ear_0.3`: can you hear a slight change, while it is still mostly the original sound? (yes / no)
- `by_ear_0.6` and `by_ear_1`: does it sound like the mistake? (yes / partly / no)
- `notes`: anything else, e.g. "sounds like a lisp, not a th"

{blocked}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--takes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--voice", default="Gemini Fenrir")
    parser.add_argument("--pairs", nargs="+", help="only these pairs (plain names)")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    takes = pd.read_csv(args.takes)
    sheet = build_pack(metadata, takes, args.out_dir, args.pairs)
    covered = coverage(sheet, takes)
    for frame, path in ((sheet, args.sheet), (covered, args.coverage)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    shutil.copyfile(args.sheet, args.out_dir / "listening_sheet.csv")
    problems = sheet[sheet["status"] != "ok"]
    blocked = "".join(f"- **{row.folder}**: {row.status}\n" for row in problems.itertuples())
    note = f"## Folders with something missing\n\n{blocked}" if blocked else ""
    (args.out_dir / "README.md").write_text(PACK_README.format(voice=args.voice, blocked=note), encoding="utf-8")
    print(f"{len(sheet)} pairs, {len(problems)} with something missing; pack in {args.out_dir}")


if __name__ == "__main__":
    main()

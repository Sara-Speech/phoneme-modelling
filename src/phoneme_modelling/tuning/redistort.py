"""Re-make the clips of some pairs (plain and matched) after their edit changed, and leave every other clip alone.

The pairs' old clips move to ``<folder>_before/`` so they can still be compared, their rows leave the metadata, and
the pairs run again on the clean takes with the next free file numbers.

    uv run python -m phoneme_modelling.tuning.redistort --config config/fenrir.json \
        --metadata data/fenrir/metadata.csv --folder phonemes/fenrir --pairs r_to_w r_to_l
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from phoneme_modelling.distortion.pair_registry import PAIR_DISTORTERS
from phoneme_modelling.distortion.pipeline import AudioDistortionPipeline
from phoneme_modelling.schema.phoneme_pipeline_config import PhonemePipelineConfig
from phoneme_modelling.tuning.report import distorter_of

ALL_NAMES = [cls.get_name() for cls in PAIR_DISTORTERS]


def redistort(config: PhonemePipelineConfig, metadata_path: Path, folder: Path, pairs: list[str]) -> pd.DataFrame:
    """Re-makes ``pairs`` and their matched copies; returns the new rows."""
    unknown = set(pairs) - set(ALL_NAMES)
    if unknown:
        raise ValueError(f"unknown pairs: {sorted(unknown)}")
    names = set(pairs) | {f"{pair}_matched" for pair in pairs}
    metadata = pd.read_csv(metadata_path)
    made_by = metadata["file_name"].map(lambda name: distorter_of(name, ALL_NAMES))
    old = metadata[made_by.isin(names)]
    before = folder.parent / f"{folder.name}_before"
    before.mkdir(parents=True, exist_ok=True)
    for path in old["file_path"]:
        if Path(path).is_file():
            shutil.move(path, before / Path(path).name)
    kept = metadata[~made_by.isin(names)]
    next_number = 1 + max(int(name.partition("_")[0]) for name in metadata["file_name"])

    run_config = config.model_copy(update={"distorter_configs": dict.fromkeys(sorted(names))})
    with tempfile.TemporaryDirectory() as scratch:
        clean_path = Path(scratch) / "clean.csv"
        kept[kept["distortion_intensity"] == 0].to_csv(clean_path, index=False)
        AudioDistortionPipeline.run(run_config, folder, clean_path, next_number)
        made = pd.read_csv(clean_path)
    new_rows = made[made["distortion_intensity"] > 0]
    pd.concat([kept, new_rows], ignore_index=True).to_csv(metadata_path, index=False)
    return new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="plain pair names; matched copies are included")
    args = parser.parse_args()

    config = PhonemePipelineConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    new_rows = redistort(config, args.metadata, args.folder, args.pairs)
    print(f"made {len(new_rows)} clips for {len(args.pairs)} pairs (old clips in {args.folder.name}_before/)")


if __name__ == "__main__":
    main()

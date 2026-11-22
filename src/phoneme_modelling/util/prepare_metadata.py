"""Turns the hand-reviewed metadata spreadsheet into the CSV that `make eval` reads by default.

The spreadsheet (downloaded from Drive with the audio) is the source of truth: edit the review there, then rerun this.
The Remove? column is kept, and load_audio_metadata skips the rows marked Y, so the CSV is a full copy of the review.

    uv run python -m phoneme_modelling.util.prepare_metadata
"""
import argparse
from pathlib import Path

import pandas as pd

from phoneme_modelling.util.loaders import REMOVE_COLUMN, load_audio_metadata


def prepare_metadata(xlsx_path: Path, csv_path: Path, audio_base_dir: Path) -> dict[str, int]:
    """Writes the spreadsheet to csv_path and returns counts for a sanity check: rows, rows marked for removal, rows
    kept, clean clips kept, and kept rows whose audio file is missing under audio_base_dir."""
    dataframe = pd.read_excel(xlsx_path)
    temporary_path = csv_path.with_name(csv_path.name + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(csv_path)  # never leaves a half-written CSV behind

    removed = 0
    if REMOVE_COLUMN in dataframe.columns:
        removed = int((dataframe[REMOVE_COLUMN].astype(str).str.strip().str.lower() == "y").sum())
    kept = load_audio_metadata(csv_path, audio_base_dir)
    return {
        "rows": len(dataframe),
        "marked_for_removal": removed,
        "kept": len(kept),
        "kept_clean": sum(1 for row in kept if not row["distortion_intensity"]),
        "missing_audio": sum(1 for row in kept if not Path(str(row["file_path"])).is_file()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", type=Path, default=Path("data/phonemes_metadata.xlsx"))
    parser.add_argument("--csv", type=Path, default=Path("data/phonemes_metadata.csv"))
    parser.add_argument("--audio_base_dir", type=Path, default=Path("data"),
                        help="folder the spreadsheet's relative file_paths are under (default: data)")
    args = parser.parse_args()
    counts = prepare_metadata(args.xlsx, args.csv, args.audio_base_dir)
    print(f"Wrote {args.csv} from {args.xlsx}: {counts['rows']} rows, {counts['marked_for_removal']} marked "
          f"{REMOVE_COLUMN}, {counts['kept']} kept ({counts['kept_clean']} clean)")
    if counts["missing_audio"]:
        print(f"WARNING: {counts['missing_audio']} kept rows have no audio file under {args.audio_base_dir}; "
              "the evaluation will skip them. Is the audio folder from the same download as the spreadsheet?")


if __name__ == "__main__":
    main()

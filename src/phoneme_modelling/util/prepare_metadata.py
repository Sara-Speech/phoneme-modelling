"""Copies the hand-reviewed metadata spreadsheet to the CSV that `make eval` reads. Rerun after editing the spreadsheet.
Its Remove? column is kept; the loader skips the rows marked Y.

    uv run python -m phoneme_modelling.util.prepare_metadata
"""
from pathlib import Path

import pandas as pd

XLSX_PATH = Path("data/phonemes_metadata.xlsx")
CSV_PATH = Path("data/phonemes_metadata.csv")


def prepare_metadata(xlsx_path: Path = XLSX_PATH, csv_path: Path = CSV_PATH) -> None:
    pd.read_excel(xlsx_path).to_csv(csv_path, index=False)


if __name__ == "__main__":
    prepare_metadata()
    print(f"Wrote {CSV_PATH} from {XLSX_PATH}")

from typing import Optional

import numpy as np
import pandas as pd
from pathlib import Path

import librosa

def load_as_numpy_array(file_path: Path|str, sample_rate: Optional[int] = None) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio

def load_as_numpy_array_with_sample_rate(file_path: Path|str) -> tuple[np.ndarray, int | float]:
    audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
    return audio, sample_rate

REMOVE_COLUMN = "Remove?"  # hand-review column in the shared metadata spreadsheet: Y/y = drop the clip


def load_csv_metadata(file_path: Path) -> list[dict]:
    dataframe = pd.read_csv(file_path).astype("object")
    dataframe = dataframe.where(pd.notna(dataframe), None)
    return dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")


def load_audio_metadata(file_path: Path, audio_base_dir: Optional[Path] = None) -> list[dict]:
    """Clip metadata from a .csv or .xlsx, without the rows marked Y/y in the Remove? column (if there is one).
    With audio_base_dir, relative file_paths are resolved against it (e.g. "data" for a downloaded folder whose
    paths are phonemes/...); otherwise they are left as they are."""
    file_path = Path(file_path)
    if file_path.suffix.lower() in (".xlsx", ".xls"):
        dataframe = pd.read_excel(file_path)
    else:
        dataframe = pd.read_csv(file_path)
    if REMOVE_COLUMN in dataframe.columns:
        removed = dataframe[REMOVE_COLUMN].astype(str).str.strip().str.lower() == "y"
        dataframe = dataframe[~removed].drop(columns=REMOVE_COLUMN)
    dataframe = dataframe.astype("object")
    rows = dataframe.where(pd.notna(dataframe), None).to_dict(orient="records")
    if audio_base_dir is not None:
        for row in rows:
            audio_path = Path(str(row["file_path"]))
            if not audio_path.is_absolute():
                row["file_path"] = str(Path(audio_base_dir) / audio_path)
    return rows
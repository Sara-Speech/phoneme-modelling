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

def load_audio_metadata(file_path: Path) -> list[dict]:
    dataframe = pd.read_csv(file_path).astype("object")
    dataframe = dataframe.where(pd.notna(dataframe), None)
    return dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")
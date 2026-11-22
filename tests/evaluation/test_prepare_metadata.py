from pathlib import Path

import pandas as pd

from phoneme_modelling.util.loaders import load_audio_metadata
from phoneme_modelling.util.prepare_metadata import prepare_metadata


def test_prepare_metadata_keeps_the_review(tmp_path: Path) -> None:
    rows = pd.DataFrame({
        "file_name": ["0000_r_a.wav", "0001_s_a.wav", "0002_g_a.wav"],
        "file_path": ["phonemes/0000_r_a.wav", "phonemes/0001_s_a.wav", "phonemes/0002_g_a.wav"],
        "Remove?": [None, "y", " Y "],
    })
    xlsx, csv = tmp_path / "metadata.xlsx", tmp_path / "metadata.csv"
    rows.to_excel(xlsx, index=False)

    prepare_metadata(xlsx, csv)

    assert "Remove?" in pd.read_csv(csv).columns
    assert [row["file_name"] for row in load_audio_metadata(csv)] == ["0000_r_a.wav"]

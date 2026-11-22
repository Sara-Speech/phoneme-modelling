from pathlib import Path

import pandas as pd

from phoneme_modelling.util.loaders import load_audio_metadata
from phoneme_modelling.util.prepare_metadata import prepare_metadata


def test_prepare_metadata_keeps_review_and_counts(tmp_path: Path) -> None:
    (tmp_path / "phonemes").mkdir()
    for name in ("0000_r_a.wav", "0001_s_a.wav", "0003_s_b_distorted.wav"):
        (tmp_path / "phonemes" / name).write_bytes(b"")
    rows = pd.DataFrame({
        "file_name": ["0000_r_a.wav", "0001_s_a.wav", "0002_g_a.wav", "0003_s_b_distorted.wav", "0004_l_a.wav"],
        "file_path": ["phonemes/0000_r_a.wav", "phonemes/0001_s_a.wav", "phonemes/0002_g_a.wav",
                      "phonemes/0003_s_b_distorted.wav", "phonemes/0004_l_a.wav"],
        "distortion_intensity": [0.0, 0.0, 0.0, 0.6, 0.0],
        "Remove?": [None, "y", None, None, " Y "],
    })
    xlsx, csv = tmp_path / "metadata.xlsx", tmp_path / "metadata.csv"
    rows.to_excel(xlsx, index=False)

    counts = prepare_metadata(xlsx, csv, tmp_path)

    assert counts == {"rows": 5, "marked_for_removal": 2, "kept": 3, "kept_clean": 2, "missing_audio": 1}
    assert "Remove?" in pd.read_csv(csv).columns  # the review is kept, not applied
    kept = [row["file_name"] for row in load_audio_metadata(csv, tmp_path)]
    assert kept == ["0000_r_a.wav", "0002_g_a.wav", "0003_s_b_distorted.wav"]
    assert not (tmp_path / "metadata.csv.tmp").exists()

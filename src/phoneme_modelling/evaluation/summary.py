"""Summary tables for an evaluation results CSV.

    uv run python src/phoneme_modelling/evaluation/summary.py --evaluation_path data/evaluation_results.csv

Writes into --output_folder (default data/evaluation_summary/):
- accuracy.csv: accuracy per evaluator, overall and by generator model, distortion intensity, and expected sound
- confusion_<evaluator>.csv: expected sound (rows) x heard sound (columns), counts
- distortion_outcomes.csv: for distorted clips, how often each evaluator heard the source sound, the target sound,
  or something else, by intensity
- listener_agreement.csv: how often each pair of evaluators gave the same sound, per clip

Correctness is recomputed here as detected IPA == expected IPA, where the expected IPA of a distorted clip is the
larger share of its mix (e.g. "l" for "distortion[ɹ(40%)|l(60%)]"). Clips whose expected sound has no label (e.g. the
derhotacized r ɹ̞) are left out of accuracy but kept in the distortion outcomes.
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from phoneme_modelling.phonemes import IPA_TO_TEXT
from phoneme_modelling.util.ipa import clean_ipa, expected_ipa


def _with_derived_columns(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["expected_ipa"] = results["actual_phoneme_ipa"].map(expected_ipa)
    results["expected_text"] = results["expected_ipa"].map(IPA_TO_TEXT)
    results["detected_ipa"] = results["detected_phoneme_ipa"].fillna("").astype(str).map(clean_ipa)
    results["heard"] = results["detected_ipa"].map(IPA_TO_TEXT)
    if "detected_phoneme_text" in results:  # LLM listeners also answer "other" / "none"
        results["heard"] = results["heard"].fillna(results["detected_phoneme_text"])
    results["heard"] = results["heard"].fillna("unrecognised")
    results["correct"] = results["detected_ipa"] == results["expected_ipa"]
    if "distortion_intensity" not in results:
        results["distortion_intensity"] = results["file_name"].str.contains("_distortion_").map({True: None, False: 0.0})
    if "model" not in results:
        results["model"] = None
    results["is_distorted"] = results["actual_phoneme_ipa"].str.startswith("distortion[")
    return results


def accuracy_table(results: pd.DataFrame) -> pd.DataFrame:
    scored = results[results["expected_text"].notna()]
    rows = []
    for breakdown in [None, "model", "distortion_intensity", "expected_text"]:
        keys = ["evaluation"] + ([breakdown] if breakdown else [])
        grouped = scored.groupby(keys, dropna=False)["correct"].agg(["mean", "size"]).reset_index()
        grouped["breakdown"] = breakdown or "overall"
        grouped["value"] = grouped[breakdown].astype(str) if breakdown else "all"
        rows.append(grouped[["evaluation", "breakdown", "value", "mean", "size"]])
    return pd.concat(rows).rename(columns={"mean": "accuracy", "size": "clips"})


def distortion_outcomes(results: pd.DataFrame) -> pd.DataFrame:
    distorted = results[results["is_distorted"]].copy()
    distorted["outcome"] = "other"
    distorted.loc[distorted["heard"] == distorted["actual_phoneme_text"], "outcome"] = "heard_source"
    distorted.loc[distorted["heard"] == distorted["expected_text"], "outcome"] = "heard_target"
    counts = distorted.groupby(["evaluation", "distortion_intensity", "outcome"]).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).round(3).reset_index()


def listener_agreement(results: pd.DataFrame) -> pd.DataFrame:
    heard = results.pivot_table(index="file_name", columns="evaluation", values="heard", aggfunc="first")
    rows = []
    for first, second in combinations(heard.columns, 2):
        both = heard[[first, second]].dropna()
        rows.append({"first": first, "second": second, "clips": len(both),
                     "agreement": (both[first] == both[second]).mean() if len(both) else None})
    return pd.DataFrame(rows)


def summarize(evaluation_path: Path, output_folder: Path) -> None:
    results = _with_derived_columns(pd.read_csv(evaluation_path))
    output_folder.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_table(results)
    accuracy.to_csv(output_folder / "accuracy.csv", index=False)
    for evaluation, rows in results.groupby("evaluation"):
        confusion = pd.crosstab(rows["expected_text"].fillna(rows["expected_ipa"]), rows["heard"])
        confusion.to_csv(output_folder / f"confusion_{evaluation}.csv")
    distortion_outcomes(results).to_csv(output_folder / "distortion_outcomes.csv", index=False)
    listener_agreement(results).to_csv(output_folder / "listener_agreement.csv", index=False)
    overall = accuracy[accuracy["breakdown"].isin(["overall", "distortion_intensity"])]
    print(overall.to_string(index=False))
    print(f"\nTables written to {output_folder}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_path", default="data/evaluation_results.csv")
    parser.add_argument("--output_folder", default="data/evaluation_summary")
    args = parser.parse_args()
    summarize(Path(args.evaluation_path), Path(args.output_folder))


if __name__ == "__main__":
    main()

import argparse
import json
import os
from pathlib import Path

from phoneme_modelling.distortion.pipeline import AudioDistortionPipeline
from phoneme_modelling.evaluation.evaluation_summarizer import AudioEvaluationSummarizer
from phoneme_modelling.evaluation.pipeline import AudioEvaluationPipeline
from phoneme_modelling.schema.phoneme_pipeline_config import PhonemePipelineConfig
from phoneme_modelling.synthetic_audio.pipeline import SyntheticAudioPipeline
from phoneme_modelling.util.env import load_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/default.json")
    parser.add_argument("--output_folder", default="phonemes")
    parser.add_argument("--output_metadata_path", default="data/phonemes_metadata.csv")
    parser.add_argument("--evaluation_output_path", default="data/evaluation_results.csv")
    parser.add_argument("--evaluation_summary_path", default="data/evaluation_summary.csv")
    parser.add_argument("--mode", choices=["all", "generate", "eval"], default="all")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    output_folder = Path(args.output_folder)
    output_metadata_path = Path(args.output_metadata_path)
    evaluation_output_path = Path(args.evaluation_output_path)
    evaluation_summary_path = Path(args.evaluation_summary_path)
    if args.mode in ["all", "generate"] and os.path.exists(output_folder):
        os.rmdir(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_metadata_path.parent, exist_ok=True)
    os.makedirs(evaluation_output_path.parent, exist_ok=True)
    os.makedirs(evaluation_summary_path.parent, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = json.load(config_file)
        config = PhonemePipelineConfig.model_validate(config_dict)

    load_env(config.env_path)

    if args.mode in ["all", "generate"]:
        if os.path.exists(output_metadata_path):
            os.remove(output_metadata_path)
        synthetic_audio_count = SyntheticAudioPipeline.run(config, output_folder, output_metadata_path)
        AudioDistortionPipeline.run(config, output_folder, output_metadata_path, synthetic_audio_count)
    if args.mode in ["all", "eval"]:
        if os.path.exists(evaluation_output_path):
            os.remove(evaluation_output_path)
        if os.path.exists(evaluation_summary_path):
            os.remove(evaluation_summary_path)
        AudioEvaluationPipeline.run(config, output_metadata_path, evaluation_output_path)
        AudioEvaluationSummarizer.run(evaluation_output_path, evaluation_summary_path)

if __name__ == "__main__":
    main()
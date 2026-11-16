from pathlib import Path

from pandas import DataFrame

from phoneme_modelling.phonemes import IpaRepresentations
from phoneme_modelling.schema.evaluator_summary_metadata import EvaluatorSummaryMetadata
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluationOutput
from phoneme_modelling.util.loaders import load_csv_metadata


ALL = "all"
TOTAL = "total"
CORRECT = "correct"
UNDISTORTED = "undistorted"
UNDISTORTED_CORRECT = "undistorted_correct"
DISTORTED = "distorted"
DISTORTED_CORRECT = "distorted_correct"
ALL_KEY = (ALL, ALL, ALL)



class AudioEvaluationSummarizer():
        
    @classmethod
    def run(
        cls,
        evaluation_output_path: Path,
        evaluation_summary_path: Path
    ) -> None:
        evaluator_results = {}
        model_voices = {}
        evaluation_metadata: list[dict] = load_csv_metadata(evaluation_output_path)
        for evaluation_dict in evaluation_metadata:
            evaluation_result = PhonemeEvaluationOutput.model_validate(evaluation_dict)
            evaluator_results = cls.handle_evaluation_result(evaluator_results, evaluation_result)
            if evaluation_result.model not in model_voices:
                model_voices[evaluation_result.model] = set()
            if evaluation_result.voice not in model_voices[evaluation_result.model]:
                model_voices[evaluation_result.model].add(evaluation_result.voice)
        summaries: list[EvaluatorSummaryMetadata] = []
        for evaluation_method in evaluator_results.keys():
            summaries.append(cls.build_metadata_for_key(evaluator_results, evaluation_method, ALL_KEY))
            for phoneme in IpaRepresentations:
                phoneme_key = (ALL, ALL, phoneme.name.lower())
                if not phoneme_key in evaluator_results[evaluation_method]:
                    continue
                summaries.append(cls.build_metadata_for_key(evaluator_results, evaluation_method, phoneme_key))
                for model in model_voices.keys():
                    model_key = (model, ALL, ALL)
                    summaries.append(cls.build_metadata_for_key(evaluator_results, evaluation_method, model_key))
                    for voice in model_voices[model]:
                        voice_key = (model, voice, ALL)
                        summaries.append(cls.build_metadata_for_key(evaluator_results, evaluation_method, voice_key))
        DataFrame([entry.model_dump() for entry in summaries]).to_csv(evaluation_summary_path, index=False)

    @classmethod
    def handle_evaluation_result(cls, evaluator_results: dict, evaluation_result: PhonemeEvaluationOutput) -> dict:
        name = evaluation_result.evaluation
        if name not in evaluator_results:
            evaluator_results[name] = {}
        specific_evaluator = evaluator_results[name]
        phoneme_key = (ALL, ALL, evaluation_result.actual_phoneme_text)
        model_key = (evaluation_result.model, ALL, ALL)
        voice_key = (evaluation_result.model, evaluation_result.voice, ALL)
        keys = [ALL_KEY, phoneme_key, model_key, voice_key]
        for key in keys:
            specific_evaluator = cls.ensure_key_exists(specific_evaluator, key)
            specific_evaluator[key][TOTAL] += 1
            if evaluation_result.is_distorted:
                specific_evaluator[key][DISTORTED] += 1
            else:
                specific_evaluator[key][UNDISTORTED] += 1
            if evaluation_result.is_correct:
                specific_evaluator[key][CORRECT] += 1
                if evaluation_result.is_distorted:
                    specific_evaluator[key][DISTORTED_CORRECT] += 1
                else:
                    specific_evaluator[key][UNDISTORTED_CORRECT] += 1
        evaluator_results[name] = specific_evaluator
        return evaluator_results

    @staticmethod
    def ensure_key_exists(evaluator_results: dict, key: tuple[str, str, str]) -> dict:
        if key not in evaluator_results:
            evaluator_results[key] = {
                TOTAL: 0,
                CORRECT: 0,
                UNDISTORTED: 0,
                UNDISTORTED_CORRECT: 0,
                DISTORTED: 0,
                DISTORTED_CORRECT: 0}
        return evaluator_results

    @classmethod
    def build_metadata_for_key(
        cls,
        evaluator_results: dict,
        evaluation_method: str,
        key: tuple[str, str, str]
    ) -> EvaluatorSummaryMetadata:
        results = evaluator_results[evaluation_method][key]
        model = key[0]
        voice = key[1]
        phoneme = key[2]
        return EvaluatorSummaryMetadata(
            evaluation=evaluation_method,
            model=model,
            voice=voice,
            phoneme=phoneme,
            accuracy=cls.get_accuracy_percentage(results[TOTAL], results[CORRECT]),
            number_of_samples=results[TOTAL],
            undistorted_accuracy=cls.get_accuracy_percentage(results[UNDISTORTED], results[UNDISTORTED_CORRECT]),
            undistorted_number_of_samples=results[UNDISTORTED],
            distorted_accuracy=cls.get_accuracy_percentage(results[DISTORTED], results[DISTORTED_CORRECT]),
            distorted_number_of_samples=results[DISTORTED]
        )

    @staticmethod
    def get_accuracy_percentage(total: int, correct: int, decimal_places: int = 3) -> float:
        if total == 0:
            return 0
        return round(correct/total, decimal_places)
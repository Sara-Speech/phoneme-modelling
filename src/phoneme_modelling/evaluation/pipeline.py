from typing import Any

from pandas import DataFrame
from pathlib import Path

from phoneme_modelling.evaluation.evaluators.allosaurus import AllosaurusEvaluator
from phoneme_modelling.evaluation.evaluators.debug import DebugEvaluator
from phoneme_modelling.evaluation.evaluator import AudioEvaluator
from phoneme_modelling.evaluation.evaluator_factory import AudioEvaluatorFactory
from phoneme_modelling.evaluation.evaluators.gemini_listener import GeminiListenerEvaluator
from phoneme_modelling.evaluation.evaluators.openai_listener import OpenAIListenerEvaluator
from phoneme_modelling.evaluation.evaluators.sara_model import SaraModelEvaluator
from phoneme_modelling.evaluation.evaluators.wav2vec2 import Wav2Vec2Evaluator
from phoneme_modelling.evaluation.evaluators.wav2vec2_espeak import Wav2Vec2EspeakEvaluator
from phoneme_modelling.evaluation.evaluators.zipa import ZipaEvaluator
from phoneme_modelling.schema.llm_listener_config import GeminiListenerConfig, LLMListenerConfig
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluationOutput
from phoneme_modelling.schema.phoneme_pipeline_config import PhonemePipelineConfig
from phoneme_modelling.schema.sara_model_evaluator_config import SaraModelEvaluatorConfig
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.util.loaders import load_audio_metadata


class AudioEvaluationPipeline():

    @staticmethod
    def build_evaluators(configs: dict[str, Any]) -> list[AudioEvaluator]:
        factory = AudioEvaluatorFactory()
        factory.register(Wav2Vec2Evaluator, None)
        factory.register(Wav2Vec2EspeakEvaluator, None)
        factory.register(ZipaEvaluator, None)
        factory.register(GeminiListenerEvaluator, GeminiListenerConfig)
        factory.register(OpenAIListenerEvaluator, LLMListenerConfig)
        factory.register(AllosaurusEvaluator, None)
        factory.register(SaraModelEvaluator, SaraModelEvaluatorConfig)
        return factory.generate(configs)

    @classmethod
    def run(
        cls,
        pipeline_config: PhonemePipelineConfig,
        audio_metadata_file_path: Path,
        evaluation_output_path: Path,
    ) -> None:
        evaluation_output: list[PhonemeEvaluationOutput] = []
        evaluator_configs = pipeline_config.evaluator_configs
        if evaluator_configs is None:
            return
        evaluators = cls.build_evaluators(evaluator_configs)
        audio_infos: list[SyntheticPhonemeMetadata] = []
        for audio_dict in load_audio_metadata(audio_metadata_file_path, pipeline_config.evaluation_audio_base_dir):
            audio_info = SyntheticPhonemeMetadata.model_validate(audio_dict)
            if not Path(audio_info.file_path).is_file():
                continue
            if pipeline_config.evaluate_only_undistorted and audio_info.distortion_intensity > 0:
                continue
            audio_infos.append(audio_info)
        print(f"Evaluating {len(audio_infos)} clips with {[evaluator.get_name() for evaluator in evaluators]}")
        for evaluator in evaluators:
            results = evaluator.evaluate_all(audio_infos)
            for audio_info, result in zip(audio_infos, results):
                output = PhonemeEvaluationOutput(
                    evaluation=evaluator.get_name(),
                    **audio_info.model_dump(),
                    **result.model_dump(),
                    is_distorted=audio_info.distortion_intensity > 0)
                evaluation_output.append(output)
        DataFrame([entry.model_dump() for entry in evaluation_output]).to_csv(evaluation_output_path, index=False)

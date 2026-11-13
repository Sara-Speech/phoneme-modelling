import os

from google import genai
from google.genai.types import GenerateContentConfig, Part

from phoneme_modelling.evaluation.llm_listening import LISTENER_PROMPT, RESPONSE_JSON_SCHEMA, LLMListenerEvaluator
from phoneme_modelling.schema.llm_listener_config import GeminiListenerConfig
from phoneme_modelling.util.env import load_env



class GeminiListenerEvaluator(LLMListenerEvaluator):
    """Gemini on Vertex AI listens to each clip and names the sound it hears (blind to the target)."""

    def __init__(self, config: GeminiListenerConfig):
        load_env(config.env_file)
        self.__client = genai.Client(vertexai=True, project=os.environ.get("GCP_PROJECT"), location=config.location)
        self.__generation_config = GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=RESPONSE_JSON_SCHEMA,
        )
        super().__init__(config)

    @staticmethod
    def get_name() -> str:
        return "gemini_listener"

    def _ask(self, wav_bytes: bytes) -> str:
        response = self.__client.models.generate_content(
            model=self.config.model,
            contents=[Part.from_bytes(data=wav_bytes, mime_type="audio/wav"), LISTENER_PROMPT],
            config=self.__generation_config,
        )
        return response.text or ""

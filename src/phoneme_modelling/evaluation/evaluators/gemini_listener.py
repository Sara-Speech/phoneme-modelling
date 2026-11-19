import os

from google import genai
from google.genai.types import GenerateContentConfig, Part

from phoneme_modelling.evaluation.llm_listening import LISTENER_PROMPT, RESPONSE_JSON_SCHEMA, LLMListenerEvaluator
from phoneme_modelling.schema.llm_listener_config import GeminiListenerConfig



class GeminiListenerEvaluator(LLMListenerEvaluator):
    """Gemini on Vertex AI (GCP_PROJECT), through generate_content with the WAV file as an inline part."""

    def __init__(self, config: GeminiListenerConfig):
        super().__init__(config)
        self.client = genai.Client(vertexai=True, project=os.environ.get("GCP_PROJECT"), location=config.location)

    @staticmethod
    def get_name() -> str:
        return "gemini_listener"

    def _transcribe(self, audio: bytes) -> str:
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=[Part.from_bytes(data=audio, mime_type="audio/wav"), LISTENER_PROMPT],
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=RESPONSE_JSON_SCHEMA,
            ),
        )
        return response.text or ""

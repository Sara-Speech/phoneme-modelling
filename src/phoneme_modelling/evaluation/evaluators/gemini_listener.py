import os

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, Part

from phoneme_modelling.evaluation.llm_listening import LISTENER_PROMPT, RESPONSE_JSON_SCHEMA, LLMListenerEvaluator
from phoneme_modelling.schema.llm_listener_config import GeminiListenerConfig



class Gemini31ProListenerEvaluator(LLMListenerEvaluator):
    """Gemini on Vertex AI (GCP_PROJECT), through generate_content with the WAV file as an inline part.

    Configured with gemini-3.1-pro-preview; the model itself comes from the config's "model". Its name stays
    "gemini_listener" so earlier results and cached answers still match."""

    def __init__(self, config: GeminiListenerConfig):
        super().__init__(config)
        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("GCP_PROJECT"),
            location=config.location,
            http_options=HttpOptions(timeout=int(config.request_timeout_seconds * 1000)),  # milliseconds
        )

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


class Gemini4NeonListenerEvaluator(Gemini31ProListenerEvaluator):
    """The same Gemini listener under its own name, so its results are kept apart from the other Gemini listeners'."""

    @staticmethod
    def get_name() -> str:
        return "gemini_4_neon_listener"


class Gemini4CarbonListenerEvaluator(Gemini31ProListenerEvaluator):
    """The same Gemini listener under its own name, so its results are kept apart from the other Gemini listeners'."""

    @staticmethod
    def get_name() -> str:
        return "gemini_4_carbon_listener"

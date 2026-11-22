import base64

from openai import OpenAI

from phoneme_modelling.evaluation.llm_listening import LISTENER_PROMPT, LLMListenerEvaluator
from phoneme_modelling.schema.llm_listener_config import LLMListenerConfig



class OpenAIListenerEvaluator(LLMListenerEvaluator):
    """An OpenAI gpt-audio* model through Chat Completions, with the WAV file as input_audio and a text-only answer.

    The prompt goes in a system message and the user message is only the audio. With the prompt and the audio together
    in the user message, gpt-audio-1.5 often ignored the audio and replied "please provide the audio clip"."""

    REQUEST_VERSION = "system-prompt"  # answers cached under the old layout (prompt + audio in one message) aren't reused

    def __init__(self, config: LLMListenerConfig):
        super().__init__(config)
        # reads OPENAI_API_KEY; retries are done by LLMListenerEvaluator, so the SDK's own are turned off
        self.client = OpenAI(timeout=config.request_timeout_seconds, max_retries=0)

    @staticmethod
    def get_name() -> str:
        return "openai_listener"

    def _transcribe(self, audio: bytes) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            modalities=["text"],
            messages=[
                {"role": "system", "content": LISTENER_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode(), "format": "wav"}},
                    ],
                },
            ],
        )
        return response.choices[0].message.content or ""

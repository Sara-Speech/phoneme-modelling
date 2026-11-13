import base64

from openai import OpenAI

from phoneme_modelling.evaluation.llm_listening import LISTENER_PROMPT, LLMListenerEvaluator
from phoneme_modelling.schema.llm_listener_config import OpenAIListenerConfig
from phoneme_modelling.util.env import load_env



class OpenAIListenerEvaluator(LLMListenerEvaluator):
    """An OpenAI audio model (Chat Completions with input_audio) listens to each clip and names the sound it hears
    (blind to the target)."""

    def __init__(self, config: OpenAIListenerConfig):
        load_env(config.env_file)
        self.__client = OpenAI()  # reads OPENAI_API_KEY
        super().__init__(config)

    @staticmethod
    def get_name() -> str:
        return "openai_listener"

    def _ask(self, wav_bytes: bytes) -> str:
        response = self.__client.chat.completions.create(
            model=self.config.model,
            modalities=["text"],
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": LISTENER_PROMPT},
                    {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"}},
                ],
            }],
        )
        return response.choices[0].message.content or ""

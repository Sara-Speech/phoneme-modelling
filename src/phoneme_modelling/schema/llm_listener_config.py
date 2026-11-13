from pathlib import Path

from pydantic import BaseModel



class LLMListenerConfig(BaseModel):
    """Settings shared by the LLM listener evaluators (gemini_listener, openai_listener)."""
    model: str
    env_file: Path = Path(".env")
    workers: int = 8  # clips sent in parallel
    max_retries: int = 3  # per clip, for API or parsing errors
    cache_dir: Path = Path("data/llm_listener_cache")  # one jsonl per evaluator+model, so reruns resume


class GeminiListenerConfig(LLMListenerConfig):
    model: str = "gemini-3.8-flash"
    location: str = "us-central1"  # Vertex AI region; the project comes from GCP_PROJECT in env_file


class OpenAIListenerConfig(LLMListenerConfig):
    model: str = "gpt-audio-1.5"  # must accept input_audio in Chat Completions

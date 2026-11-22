from pathlib import Path

from pydantic import BaseModel



class LLMListenerConfig(BaseModel):
    model: str
    cache_dir: Path = Path("data/llm_listener_cache")  # one JSON answer per clip, under <cache_dir>/<evaluator name>
    max_concurrent_requests: int = 8
    max_retries: int = 3


class GeminiListenerConfig(LLMListenerConfig):
    location: str = "global"  # Vertex AI region

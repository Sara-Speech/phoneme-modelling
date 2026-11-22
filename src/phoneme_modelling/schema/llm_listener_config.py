from pathlib import Path

from pydantic import BaseModel



class LLMListenerConfig(BaseModel):
    model: str
    cache_dir: Path = Path("data/llm_listener_cache")  # one JSON answer per clip, under <cache_dir>/<evaluator name>
    max_concurrent_requests: int = 8
    max_retries: int = 3  # retries after an error that isn't a rate limit
    max_rate_limit_retries: int = 8  # retries after a 429 (quota exceeded), each waiting longer than the last
    rate_limit_wait_seconds: float = 10  # first wait after a 429; doubles each time, up to 5 minutes
    request_timeout_seconds: float = 120  # a request with no answer by then fails and is retried, instead of hanging


class GeminiListenerConfig(LLMListenerConfig):
    location: str = "global"  # Vertex AI region

from pathlib import Path

from pydantic import BaseModel



class GoogleTTSConfig(BaseModel):
    env_file: Path = Path(".env")
    voices: list[str]  # Cloud TTS voice names, e.g. "en-US-Neural2-F"
    language_code: str = "en-US"
    hold_repeats: list[int] = [8]  # a held sound repeats its symbol this many times; several values make several takes

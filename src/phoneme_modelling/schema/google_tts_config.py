from pydantic import BaseModel



class GoogleTTSConfig(BaseModel):
    project: str  # GCP project billed for Cloud Text-to-Speech (the API must be enabled there)
    voices: list[str]  # Cloud TTS voice names, e.g. "en-US-Neural2-F"
    language_code: str = "en-US"
    hold_repeats: list[int] = [8]  # a held sound repeats its symbol this many times; several values make several takes

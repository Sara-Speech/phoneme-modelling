from pathlib import Path

from pydantic import BaseModel



class SaraModelEvaluatorConfig(BaseModel):
    local_path: Path = Path("./.hf_cache/sara/wav2vec2-xls-r-300m-timit-phoneme")
    remote_path: str

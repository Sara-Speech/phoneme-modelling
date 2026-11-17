from typing import Optional

from pydantic import BaseModel



class EvaluatorSummaryMetadata(BaseModel):
    evaluation: str
    model: str
    voice: str
    phoneme: str
    accuracy: float
    number_of_samples: int
    undistorted_accuracy: float
    undistorted_number_of_samples: int    
    distorted_accuracy: float
    distorted_number_of_samples: int

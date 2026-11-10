import numpy as np
from dataclasses import dataclass
from pathlib import Path

import wave



@dataclass
class AudioInfo:
    duration: float
    rms_loudness: float


class AudioInspector:

    @classmethod
    def get_info(cls, file_path: Path):
        duration = cls._get_wav_duration(file_path)
        rms_loudness = cls._get_wav_rms_loudness(file_path)
        return AudioInfo(
            duration=duration,
            rms_loudness=rms_loudness
        )

    @staticmethod
    def _get_wav_duration(file_path: Path) -> float:
        with wave.open(str(file_path), "rb") as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            return frames / float(rate)

    @classmethod
    def _get_wav_rms_loudness(cls, file_path: Path) -> float:
        with wave.open(str(file_path), "rb") as audio_file:
            frames = audio_file.getnframes()
            sample_width = audio_file.getsampwidth()
            raw_data = audio_file.readframes(frames)
        dtype = cls.__get_wav_dtype(sample_width)
        audio_array = np.frombuffer(raw_data, dtype=dtype)
        return np.sqrt(np.mean(np.square(audio_array.astype(np.float64))))

    @staticmethod
    def __get_wav_dtype(sample_width: int) -> type:
        if sample_width == 2:
            return np.int16
        if sample_width == 4:
            return np.int32
        if sample_width == 1:
            return np.uint8
        raise ValueError(f"Unsupported sample width for audio: {sample_width}")

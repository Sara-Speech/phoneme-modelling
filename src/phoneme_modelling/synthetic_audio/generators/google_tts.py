"""Google Cloud Text-to-Speech with SSML IPA tags.

The sound is set directly with an IPA tag inside "Please say ___ again.", located in the sentence with forced alignment,
cropped out and put through the same prep step as the other generators (16 kHz mono, centred in 2.5 s of light noise).
Needs Application Default Credentials (``gcloud auth application-default login``, or ``auth.authenticate_user()`` in
Colab) for a project with the Cloud Text-to-Speech API enabled.
"""
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from google.cloud import texttospeech

from phoneme_modelling.schema.google_tts_config import GoogleTTSConfig
from phoneme_modelling.synthetic_audio.generation import SyntheticAudioGenerationVoice, SyntheticAudioGenerator
from phoneme_modelling.synthetic_audio.google_tts_sounds import GOOGLE_TTS_SOUNDS, HOLD
from phoneme_modelling.util.audio import prep_clip, save_wav, seconds
from phoneme_modelling.util.env import load_env
from phoneme_modelling.util.forced_alignment import ALIGNER_SAMPLE_RATE, PhoneAligner


CARRIER_BEFORE = "pliːz seɪ"  # espeak IPA of "please say" and "again", the words around the tag
CARRIER_AFTER = "əɡɛn"
FADE_SECONDS = 0.010
MIN_SECONDS = 0.02  # a shorter crop is effectively empty


class GoogleTTSVoice(SyntheticAudioGenerationVoice):

    def __init__(
        self,
        client: texttospeech.TextToSpeechClient,
        aligner: PhoneAligner,
        voice_name: str,
        language_code: str,
        repeats: int,
        take: int = 1,
        takes: int = 1,
    ) -> None:
        self.client = client
        self.aligner = aligner
        self.voice_name = voice_name
        self.language_code = language_code
        self.repeats = repeats
        self.take = take
        self.takes = takes

    def get_voice_nickname(self) -> str:
        """``en-US-Neural2-F`` -> ``neural2f``. The voice must stay the last part of a clip's name (the distorters read it),
        so several hold lengths are numbered as takes."""
        short = self.voice_name.removeprefix(f"{self.language_code}-").replace("-", "").lower()
        if self.takes > 1:
            return f"{short}_take{self.take}"
        return short

    def __synthesize(self, ipa_sequence: str, word: str) -> np.ndarray:
        ssml = f'<speak>Please say <phoneme alphabet="ipa" ph="{ipa_sequence}">{word}</phoneme> again.</speak>'
        response = self.client.synthesize_speech(
            input=texttospeech.SynthesisInput(ssml=ssml),
            voice=texttospeech.VoiceSelectionParams(language_code=self.language_code, name=self.voice_name),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=ALIGNER_SAMPLE_RATE
            ),
        )
        audio, _ = sf.read(io.BytesIO(response.audio_content), dtype="float32")
        return audio if audio.ndim == 1 else audio.mean(axis=1)

    @staticmethod
    def __crop_with_fades(audio: np.ndarray, start: float, end: float, margin: float) -> np.ndarray:
        first = max(0, int((start - margin) * ALIGNER_SAMPLE_RATE))
        last = min(len(audio), int((end + margin) * ALIGNER_SAMPLE_RATE))
        crop = audio[first:last].copy()
        fade = min(int(FADE_SECONDS * ALIGNER_SAMPLE_RATE), len(crop) // 2)
        if fade:
            crop[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            crop[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return crop

    def _generate_audio(
        self,
        target_file_path: Path,
        text: str,
        ipa: str,
        distortion_intensity: Optional[float] = None,
        distort_to_text: Optional[str] = None,
        distort_to_ipa: Optional[str] = None,
    ) -> bool:
        label = f"{target_file_path.stem} (google_tts, {self.voice_name})"
        sound = GOOGLE_TTS_SOUNDS.get(text.lower())
        if sound is None:
            print(f"FAILED {label}: no Google TTS recipe for sound {text!r}")
            return False
        if sound.kind != HOLD and self.take > 1:
            return False  # only held sounds differ between takes, so one clip is enough
        sequence = sound.ipa_sequence(self.repeats)
        try:
            audio = self.__synthesize(sequence, sound.word)
            span = self.aligner.locate(audio, CARRIER_BEFORE, sequence, CARRIER_AFTER, sound.aligned_as)
        except Exception as e:  # keep going: one failed clip shouldn't stop the run
            print(f"FAILED {label}: {e}")
            return False
        if span is None:
            print(f"FAILED {label}: the aligner did not find the sound")
            return False
        crop = self.__crop_with_fades(audio, span[0], span[1], sound.margin_seconds())
        pcm = (np.clip(crop, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        if seconds(pcm, ALIGNER_SAMPLE_RATE) < MIN_SECONDS:
            print(f"FAILED {label}: the sound is empty")
            return False
        raw_path = target_file_path.parent / "raw" / target_file_path.name
        save_wav(raw_path, pcm, ALIGNER_SAMPLE_RATE)
        prep_clip(raw_path, target_file_path)
        print(f"{label}: sound {seconds(pcm, ALIGNER_SAMPLE_RATE):.2f}s, {sound.kind}, repeats {self.repeats}")
        return True


class GoogleTTSGenerator(SyntheticAudioGenerator):

    def __init__(self, config: GoogleTTSConfig) -> None:
        self.config = config
        super().__init__()

    def _init_voices(self) -> list[SyntheticAudioGenerationVoice]:
        config = self.config
        load_env(config.env_file)
        gcp_project = os.environ.get("GCP_PROJECT")
        client = texttospeech.TextToSpeechClient(client_options={"quota_project_id": gcp_project})
        aligner = PhoneAligner()
        takes = len(config.hold_repeats)
        return [
            GoogleTTSVoice(
                client=client,
                aligner=aligner,
                voice_name=voice_name,
                language_code=config.language_code,
                repeats=repeats,
                take=take,
                takes=takes,
            )
            for voice_name in config.voices
            for take, repeats in enumerate(config.hold_repeats, start=1)
        ]

    def is_distortion_capable(self) -> bool:
        return False

    @classmethod
    def get_config_name(cls) -> str:
        return "google_tts"

    def allows_retries(self) -> bool:
        """Cloud TTS is deterministic, so a retry would give the same clip."""
        return False

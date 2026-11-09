"""Finds a known sound inside a TTS sentence with CTC forced alignment on wav2vec2 (espeak IPA) posteriors."""
import unicodedata
from typing import Optional

import numpy as np


ALIGNER_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
ALIGNER_SAMPLE_RATE = 16000
FRAME_SECONDS = 0.02  # wav2vec2 stride
SPECIAL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>")


def normalize_phone(phone: str) -> str:
    """Drops length marks, stress and diacritics so ``aː`` and ``a`` compare equal."""
    decomposed = unicodedata.normalize("NFD", phone)
    kept = "".join(char for char in decomposed if not unicodedata.combining(char))
    for mark in ("ː", "ˑ", "ʰ", "ˈ", "ˌ"):
        kept = kept.replace(mark, "")
    return kept.strip()


def tokenize_longest(phones: str, vocab_tokens: list[str]) -> list[str]:
    """Greedy longest-match split of an IPA string into vocabulary tokens. Unknown characters are skipped."""
    ordered = sorted(vocab_tokens, key=len, reverse=True)
    remaining, tokens = phones.replace(" ", ""), []
    while remaining:
        for token in ordered:
            if token and remaining.startswith(token):
                tokens.append(token)
                remaining = remaining[len(token):]
                break
        else:
            remaining = remaining[1:]
    return tokens


def ctc_forced_align(log_probs: np.ndarray, token_ids: list[int], blank: int = 0) -> list[tuple[int, int]]:
    """Viterbi alignment of ``token_ids`` to ``log_probs`` (frames x vocabulary).

    Returns one (start_frame, end_frame_exclusive) per token. Raises ValueError if the audio is too short for the tokens.
    """
    frames = log_probs.shape[0]
    count = len(token_ids)
    states = 2 * count + 1
    extended = [blank] * states
    for index, token in enumerate(token_ids):
        extended[2 * index + 1] = token
    negative = -1e30
    scores = np.full((frames, states), negative)
    back = np.zeros((frames, states), dtype=np.int32)
    scores[0, 0] = log_probs[0, blank]
    if states > 1:
        scores[0, 1] = log_probs[0, extended[1]]
    for t in range(1, frames):
        for s in range(states):
            best, source = scores[t - 1, s], s
            if s >= 1 and scores[t - 1, s - 1] > best:
                best, source = scores[t - 1, s - 1], s - 1
            if s >= 2 and extended[s] != blank and extended[s] != extended[s - 2] and scores[t - 1, s - 2] > best:
                best, source = scores[t - 1, s - 2], s - 2
            scores[t, s] = best + log_probs[t, extended[s]]
            back[t, s] = source
    state = states - 1 if (states == 1 or scores[frames - 1, states - 1] >= scores[frames - 1, states - 2]) else states - 2
    path = [0] * frames
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    spans: dict[int, tuple[int, int]] = {}
    for t, state in enumerate(path):
        if state % 2 == 1:
            token_index = state // 2
            start, end = spans.get(token_index, (t, t + 1))
            spans[token_index] = (min(start, t), max(end, t + 1))
    if len(spans) != count:
        raise ValueError("audio too short for the token sequence")
    return [spans[index] for index in range(count)]


class PhoneAligner:
    """Locates a sound in speech whose phones are known. The model loads on first use."""

    def __init__(self, model_name: str = ALIGNER_MODEL) -> None:
        self.__model_name = model_name
        self.__loaded = False

    def __load(self) -> None:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer

        self.__extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.__model_name)
        # do_phonemize=False: the phones are given, so the espeak backend is not needed
        self.__tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(self.__model_name, do_phonemize=False)
        self.__model = Wav2Vec2ForCTC.from_pretrained(self.__model_name).eval()
        self.__vocab = self.__tokenizer.get_vocab()
        self.__vocab_tokens = [token for token in self.__vocab if token not in SPECIAL_TOKENS]
        self.__blank = self.__vocab["<pad>"]
        self.__loaded = True

    def locate(
        self,
        audio: np.ndarray,
        phones_before: str,
        phones: str,
        phones_after: str,
        targets: frozenset[str],
    ) -> Optional[tuple[float, float]]:
        """Start and end (seconds) of the sound ``phones`` spoken between ``phones_before`` and ``phones_after``.

        ``audio`` is 16 kHz mono. The span covers every consecutive token of ``phones`` whose normalized form is in
        ``targets`` (so a repeated symbol is found as one block). Returns None if the sound is not found.
        """
        import torch

        if not self.__loaded:
            self.__load()
        with torch.no_grad():
            inputs = self.__extractor(audio, sampling_rate=ALIGNER_SAMPLE_RATE, return_tensors="pt")
            logits = self.__model(inputs.input_values).logits
            log_probs = torch.log_softmax(logits, dim=-1)[0].numpy()
        skip = len(tokenize_longest(phones_before, self.__vocab_tokens))
        tokens = tokenize_longest(f"{phones_before} {phones} {phones_after}", self.__vocab_tokens)
        spans = ctc_forced_align(log_probs, [self.__vocab[token] for token in tokens], blank=self.__blank)
        for index in range(skip, len(tokens)):
            if normalize_phone(tokens[index]) in targets:
                last = index
                while last + 1 < len(tokens) and normalize_phone(tokens[last + 1]) in targets:
                    last += 1
                return spans[index][0] * FRAME_SECONDS, spans[last][1] * FRAME_SECONDS
        return None

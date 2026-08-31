"""Audio post-processing: end-noise trim, chunk joining, WAV encoding.

csm-maya-exp2 often emits noise at the end of clips (model README known issue);
trim_trailing_noise scans RMS windows from the end and cuts the quiet/noisy tail.
"""

import io

import numpy as np
import soundfile as sf

_WINDOW_MS = 20


def trim_trailing_noise(
    audio: np.ndarray,
    sr: int = 24000,
    threshold_db: float = -40.0,
    keep_ms: int = 50,
) -> np.ndarray:
    window = max(1, int(sr * _WINDOW_MS / 1000))
    threshold = 10 ** (threshold_db / 20)
    end = len(audio)
    while end > window:
        rms = float(np.sqrt(np.mean(np.square(audio[end - window : end]))))
        if rms >= threshold:
            break
        end -= window
    keep = int(sr * keep_ms / 1000)
    return audio[: min(len(audio), end + keep)]


def join_with_gaps(chunks: list[np.ndarray], sr: int = 24000, gap_ms: int = 120) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
    parts = []
    for i, chunk in enumerate(chunks):
        if i:
            parts.append(gap)
        parts.append(chunk.astype(np.float32))
    return np.concatenate(parts)


def to_wav_bytes(audio: np.ndarray, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

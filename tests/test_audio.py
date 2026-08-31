import io

import numpy as np
import soundfile as sf

from maya_csm.audio import join_with_gaps, to_wav_bytes, trim_trailing_noise

SR = 24000


def _tone(seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_trim_removes_quiet_tail():
    audio = np.concatenate([_tone(1.0), _tone(0.5, amp=0.001)])
    trimmed = trim_trailing_noise(audio, sr=SR, threshold_db=-40.0, keep_ms=50)
    assert len(trimmed) < len(audio)
    assert len(trimmed) >= SR  # the loud second is intact
    assert len(trimmed) <= SR + int(SR * 0.1)  # only ~keep_ms of tail retained


def test_trim_keeps_loud_audio_untouched():
    audio = _tone(1.0)
    assert len(trim_trailing_noise(audio, sr=SR)) == len(audio)


def test_trim_handles_all_silence():
    audio = np.zeros(SR, dtype=np.float32)
    trimmed = trim_trailing_noise(audio, sr=SR)
    assert len(trimmed) <= SR  # must not error or grow


def test_join_inserts_gap_between_chunks():
    a, b = _tone(0.5), _tone(0.5)
    joined = join_with_gaps([a, b], sr=SR, gap_ms=120)
    expected = len(a) + int(SR * 0.120) + len(b)
    assert len(joined) == expected
    assert joined.dtype == np.float32


def test_join_single_chunk_has_no_gap():
    a = _tone(0.5)
    assert len(join_with_gaps([a], sr=SR, gap_ms=120)) == len(a)


def test_wav_bytes_roundtrip():
    audio = _tone(0.25)
    data, sr = sf.read(io.BytesIO(to_wav_bytes(audio, sr=SR)))
    assert sr == SR
    assert len(data) == len(audio)
    np.testing.assert_allclose(data, audio, atol=1e-4)

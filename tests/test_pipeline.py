import io
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from maya_csm.config import RefClip, Settings
from maya_csm.pipeline import synthesize

SR = 24000


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate_chunk(self, text, gen_params, context=()):
        self.calls.append((text, dict(gen_params), tuple(context)))
        return np.full(SR // 4, 0.5, dtype=np.float32)  # 0.25s, loud (survives trim)


class FakeParallelModel:
    """A 2-replica engine that records peak concurrency and tags audio by length."""

    num_replicas = 2

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0

    def generate_chunk(self, text, gen_params, context=()):
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
            self.calls.append(text)
        time.sleep(0.03)
        with self._lock:
            self._active -= 1
        return np.full(len(text) * 100, 0.5, dtype=np.float32)


def _nonsilent_run_lengths(data, eps=1e-3):
    mask = np.abs(data) > eps
    runs, i = [], 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        runs.append(j - i)
        i = j
    return runs


def test_returns_valid_wav():
    model = FakeModel()
    wav = synthesize("Hello there.", Settings(), model)
    data, sr = sf.read(io.BytesIO(wav))
    assert sr == SR
    assert len(data) > 0


def test_model_receives_sanitized_text():
    model = FakeModel()
    synthesize('She said "hi!" (quietly)', Settings(), model)
    assert model.calls == [("She said hi quietly", {}, ())]


def test_tag_maps_to_cue_and_gen_params():
    model = FakeModel()
    synthesize("[whispering] Come closer.", Settings(), model)
    text, params, _ = model.calls[0]
    assert text.startswith("Shh...")
    assert "Come closer." in text
    assert params == {"temperature": 0.6}


def test_long_text_is_chunked_and_joined_with_gaps():
    model = FakeModel()
    s = Settings(max_chunk_chars=40, gap_ms=100)
    wav = synthesize("First sentence here. Second sentence here. Third sentence here.", s, model)
    assert len(model.calls) >= 2
    data, _ = sf.read(io.BytesIO(wav))
    n = len(model.calls)
    assert len(data) >= n * (SR // 4) + (n - 1) * int(SR * 0.1)


def test_reference_clips_passed_as_context():
    model = FakeModel()
    clip = RefClip(audio_path="maya.wav", transcript="Hey, it's Maya.")
    synthesize("Hello.", Settings(reference_clips=[clip]), model)
    assert model.calls[0][2] == (clip,)


def test_multi_replica_engine_generates_chunks_concurrently():
    model = FakeParallelModel()
    synthesize("Aaa. Bb. C.", Settings(max_chunk_chars=4), model)
    assert sorted(model.calls) == ["Aaa.", "Bb.", "C."]
    assert model.peak == 2


def test_multi_replica_output_stays_in_chunk_order():
    # slowest-first would reorder a completion-ordered join; map keeps input order
    model = FakeParallelModel()
    wav = synthesize("Aaa. Bb. C.", Settings(max_chunk_chars=4, gap_ms=50), model)
    data, _ = sf.read(io.BytesIO(wav))
    assert _nonsilent_run_lengths(data) == [400, 300, 200]  # len("Aaa.","Bb.","C.") * 100


def test_empty_input_raises():
    with pytest.raises(ValueError):
        synthesize("   ", Settings(), FakeModel())


def test_tags_only_unknown_input_raises():
    with pytest.raises(ValueError):
        synthesize("[backflipping]", Settings(), FakeModel())

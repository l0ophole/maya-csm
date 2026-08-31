import io

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


def test_empty_input_raises():
    with pytest.raises(ValueError):
        synthesize("   ", Settings(), FakeModel())


def test_tags_only_unknown_input_raises():
    with pytest.raises(ValueError):
        synthesize("[backflipping]", Settings(), FakeModel())

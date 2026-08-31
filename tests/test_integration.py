"""Real-model smoke test. Run with: pytest -m model (needs [model] extra + HF_TOKEN)."""

import os

import pytest

pytestmark = pytest.mark.model

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

if not (os.environ.get("HF_TOKEN") or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))):
    pytest.skip("no HF token available for gated sesame/csm-1b", allow_module_level=True)


def test_synthesize_produces_audio():
    from maya_csm.config import Settings
    from maya_csm.model import MayaModel
    from maya_csm.pipeline import synthesize
    import io
    import soundfile as sf

    settings = Settings.from_env()
    model = MayaModel(settings)
    model.load()
    wav = synthesize("Hello there. It is lovely to meet you.", settings, model)
    data, sr = sf.read(io.BytesIO(wav))
    assert sr == 24000
    assert len(data) > sr * 0.3  # more than 0.3s of audio
    assert float(abs(data).max()) > 0.01  # not silence

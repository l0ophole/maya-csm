import pytest
from fastapi.testclient import TestClient

from maya_csm import model as model_mod
from maya_csm import server as server_mod
from maya_csm.config import Settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server_mod, "synthesize", lambda text, settings, model: b"RIFFfake")
    app = server_mod.create_app(Settings(), model=object())  # pre-supplied model: no load
    return TestClient(app)


def test_speech_endpoint_returns_wav(client):
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "maya-csm", "input": "[laughing] Hello!", "voice": "maya"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == b"RIFFfake"


def test_speech_endpoint_minimal_body(client):
    resp = client.post("/v1/audio/speech", json={"input": "Hello."})
    assert resp.status_code == 200


def test_empty_input_is_400(client):
    resp = client.post("/v1/audio/speech", json={"input": "   "})
    assert resp.status_code == 400


def test_unspeakable_input_is_400(client, monkeypatch):
    def boom(text, settings, model):
        raise ValueError("no speakable text")

    monkeypatch.setattr(server_mod, "synthesize", boom)
    resp = client.post("/v1/audio/speech", json={"input": "[backflipping]"})
    assert resp.status_code == 400


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_is_built_via_build_engine(monkeypatch):
    calls = []
    monkeypatch.setattr(model_mod, "build_engine", lambda settings: calls.append(settings) or object())
    monkeypatch.setattr(server_mod, "synthesize", lambda text, settings, model: b"RIFFfake")
    app = server_mod.create_app(Settings(preload=True))
    client = TestClient(app)
    assert client.get("/health").json()["model_loaded"] is True
    assert len(calls) == 1


def test_models_listing(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "maya-csm"


@pytest.mark.parametrize("path", ["/v1/audio/voices", "/v1/voices"])
def test_voices_listing(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert "maya" in resp.json()["voices"]

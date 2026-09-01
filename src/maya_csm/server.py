"""FastAPI server exposing an OpenAI-compatible TTS endpoint for SillyTavern.

SillyTavern's "OpenAI Compatible" TTS provider POSTs /v1/audio/speech with
{model, input, voice, response_format}. We always return WAV regardless of
response_format (ST plays WAV fine) and expose model/voice listings for probes.
"""

from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from .config import Settings
from .pipeline import synthesize

VOICES = ["maya"]


class SpeechRequest(BaseModel):
    input: str
    model: str = "maya-csm"
    voice: str = "maya"
    response_format: str = "wav"
    speed: float | None = None  # accepted, ignored


def _load_model(settings: Settings):
    from .model import build_engine  # deferred: torch/transformers are optional deps

    return build_engine(settings)


def create_app(settings: Settings | None = None, model=None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="maya-csm")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    state = {"model": model}
    lock = threading.Lock()

    def get_model():
        with lock:
            if state["model"] is None:
                state["model"] = _load_model(settings)
            return state["model"]

    if settings.preload and model is None:
        get_model()

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        if not req.input.strip():
            raise HTTPException(status_code=400, detail="input is empty")
        try:
            wav = synthesize(req.input, settings, get_model())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=wav, media_type="audio/wav")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["model"] is not None}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "maya-csm", "object": "model"}]}

    @app.get("/v1/audio/voices")
    @app.get("/v1/voices")
    def voices():
        return {"voices": VOICES}

    return app

"""CSM-1B + csm-maya-exp2 LoRA wrapper.

Load pattern follows the model README / TinkerSpace demo. Heavy deps (torch,
transformers, peft) are imported inside methods so the unit suite and server
module import cleanly without the [model] extra installed.

`build_engine()` is the entry point: it places one merged fp16 replica on every
resolved device and returns a `MayaModel` (one device) or a `ModelPool` (more).
"""

from __future__ import annotations

import contextlib
import math
import queue
import threading

import numpy as np

from .config import RefClip, Settings

# 375 audio tokens covers roughly 200 chars of speech (author's demo settings).
_TOKENS_PER_CHAR = 375 / 200
_MIN_NEW_TOKENS = 125
_MAX_NEW_TOKENS = 750
# Static-cache length when compiling: prompt headroom + _MAX_NEW_TOKENS. A fixed
# value keeps torch.compile from recompiling per input length.
_COMPILE_MAX_LENGTH = 1024

_SAMPLING_DEFAULTS = {
    "do_sample": True,
    "temperature": 0.7,
    "depth_decoder_do_sample": True,
    "depth_decoder_temperature": 0.7,
    "depth_decoder_top_k": 20,
    "depth_decoder_top_p": 0.95,
}


def _resolve_dtype(name: str, device: str) -> str:
    """Pick a dtype safe for the target device.

    T4/Turing has no native bf16, so the caller's choice (fp16 by default) is
    kept on CUDA; CPU falls back to float32 where fp16 matmul is unsupported.
    """
    return "float32" if device.startswith("cpu") else name


def resolve_devices(settings: Settings) -> list[str]:
    """Devices to place model replicas on: explicit setting, else every CUDA
    device, else CPU (also when torch is not installed)."""
    if settings.devices:
        return list(settings.devices)
    try:
        import torch
    except ImportError:
        return ["cpu"]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


class MayaModel:
    num_replicas = 1  # duck-typed marker so pipeline.synthesize can size its dispatch

    def __init__(self, settings: Settings, device: str | None = None):
        self.settings = settings
        self._device_override = device
        self.processor = None
        self.model = None
        self.device = None
        self._gen_lock = threading.Lock()  # serializes generate() in compile mode

    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, CsmForConditionalGeneration

        s = self.settings
        if self._device_override:
            self.device = self._device_override
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _resolve_dtype(s.dtype, self.device)
        try:
            self.processor = AutoProcessor.from_pretrained(s.base_model, token=s.hf_token)
            base = CsmForConditionalGeneration.from_pretrained(
                s.base_model,
                device_map=self.device,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                token=s.hf_token,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not load base model '{s.base_model}'. It is gated on Hugging Face: "
                "accept the terms at https://huggingface.co/sesame/csm-1b and set HF_TOKEN."
            ) from exc
        # Fold the LoRA into the base weights: identical math, no per-forward
        # adapter cost, and a plain module that torch.compile can trace cleanly.
        self.model = PeftModel.from_pretrained(base, s.adapter_path).merge_and_unload()
        self.model.eval()
        if s.compile:
            self._enable_compile()

    def _enable_compile(self) -> None:
        # Per the transformers CSM docs ("Making The Model Go Brrr"), a static
        # cache auto-enables torch.compile(fullgraph=True, mode="reduce-overhead")
        # — CUDA graphs. That needs greedy decoding (no torch.multinomial in the
        # graph; see generate_chunk) and a single thread/stream (see build_engine).
        m = self.model
        m.generation_config.max_length = _COMPILE_MAX_LENGTH
        m.generation_config.max_new_tokens = None
        m.generation_config.cache_implementation = "static"
        m.depth_decoder.generation_config.cache_implementation = "static"

    def generate_chunk(
        self, text: str, gen_params: dict, context: tuple[RefClip, ...] = ()
    ) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("model not loaded; call load() first")
        conversation = [self._context_turn(clip) for clip in context]
        conversation.append(
            {"role": self.settings.speaker_id, "content": [{"type": "text", "text": text}]}
        )
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        if self.settings.compile:
            # CUDA graphs can't sample (torch.multinomial advances the RNG offset
            # outside the graph); greedy only. A fixed generation_config.max_length
            # governs length so the static cache shape stays constant.
            params = {"do_sample": False, "depth_decoder_do_sample": False}
        else:
            params = {**_SAMPLING_DEFAULTS, **gen_params} if self.settings.sampling else dict(gen_params)
            params["max_new_tokens"] = min(
                _MAX_NEW_TOKENS, max(_MIN_NEW_TOKENS, math.ceil(len(text) * _TOKENS_PER_CHAR))
            )
        # CUDA graphs are single-stream: serialize concurrent requests. The first
        # call through here also triggers compilation (~1-3 min) while others wait.
        serialize = self._gen_lock if self.settings.compile else contextlib.nullcontext()
        with serialize, torch.no_grad():
            audio = self.model.generate(**inputs, output_audio=True, **params)
        return audio[0].to(torch.float32).cpu().numpy()

    def _context_turn(self, clip: RefClip) -> dict:
        import soundfile as sf

        audio, _sr = sf.read(clip.audio_path, dtype="float32")
        return {
            "role": self.settings.speaker_id,
            "content": [
                {"type": "text", "text": clip.transcript},
                {"type": "audio", "audio": audio},
            ],
        }


class ModelPool:
    """Generates chunks concurrently across one loaded replica per GPU.

    Exposes the same `generate_chunk` signature as `MayaModel`; callers check the
    `num_replicas` attribute to decide how wide to dispatch. A replica is checked
    out of a queue for the duration of a call and always returned, so concurrent
    HTTP requests and multi-chunk requests share the GPUs without oversubscribing.
    """

    def __init__(self, replicas: list):
        self._replicas = list(replicas)
        self.num_replicas = len(self._replicas)
        self._free = queue.Queue()
        for replica in self._replicas:
            self._free.put(replica)

    def generate_chunk(
        self, text: str, gen_params: dict, context: tuple[RefClip, ...] = ()
    ) -> np.ndarray:
        replica = self._free.get()
        try:
            return replica.generate_chunk(text, gen_params, context=context)
        finally:
            self._free.put(replica)


def build_engine(settings: Settings):
    """Load one replica per resolved device; return a `MayaModel` or `ModelPool`.

    `compile=True` forces a single replica: its CUDA graphs can't be driven from
    `ModelPool`'s worker threads. So `MAYA_COMPILE` and multi-GPU are a choice —
    one fast stream on one GPU, or sampled generation spread across all of them.
    """
    devices = resolve_devices(settings)
    if settings.compile:
        devices = devices[:1]
    replicas = []
    for device in devices:
        replica = MayaModel(settings, device=device)
        replica.load()
        replicas.append(replica)
    return replicas[0] if len(replicas) == 1 else ModelPool(replicas)

"""Unit tests for the torch-free helpers in maya_csm.model.

The model-loading path itself needs weights + heavy deps and is covered by
tests/test_integration.py (`-m model`).
"""

import math
import types

from maya_csm.config import Settings
from maya_csm.model import (
    _COMPILE_MAX_LENGTH,
    _MAX_NEW_TOKENS,
    _MIN_NEW_TOKENS,
    _TOKENS_PER_CHAR,
    _estimate_max_new_tokens,
    MayaModel,
    _resolve_dtype,
    resolve_devices,
)


def test_resolve_dtype_keeps_requested_dtype_on_cuda():
    assert _resolve_dtype("float16", "cuda:0") == "float16"
    assert _resolve_dtype("bfloat16", "cuda:1") == "bfloat16"


def test_resolve_dtype_forces_float32_on_cpu():
    assert _resolve_dtype("float16", "cpu") == "float32"


def test_resolve_devices_uses_explicit_setting_verbatim():
    assert resolve_devices(Settings(devices=("cuda:0", "cuda:1"))) == ["cuda:0", "cuda:1"]


def test_resolve_devices_falls_back_to_cpu_without_torch():
    # torch is not installed in the unit env -> no CUDA to enumerate
    assert resolve_devices(Settings()) == ["cpu"]


def test_estimate_max_new_tokens_scales_with_text_length():
    assert _estimate_max_new_tokens("x" * 150) == math.ceil(150 * _TOKENS_PER_CHAR)


def test_estimate_max_new_tokens_floors_short_text():
    assert _estimate_max_new_tokens("hi") == _MIN_NEW_TOKENS


def test_estimate_max_new_tokens_caps_long_text():
    assert _estimate_max_new_tokens("x" * 100_000) == _MAX_NEW_TOKENS


def test_token_budget_stays_within_compiled_static_cache():
    # A default-sized chunk's frame budget must fit the compile-mode static cache
    # (with room for the prompt), or torch.compile recompiles / truncates.
    assert _estimate_max_new_tokens("x" * 150) + 128 <= _COMPILE_MAX_LENGTH


def test_warmup_is_a_noop_off_cuda():
    m = MayaModel(Settings(), device="cpu")
    m.device = "cpu"
    calls = []
    m.generate_chunk = lambda *a, **k: calls.append((a, k))
    m._warmup()
    assert calls == []


def test_warmup_is_a_noop_when_compiling():
    # compile mode captures a CUDA graph on the calling thread; it must be
    # captured on the serving thread, not the loader thread -> skip at load.
    m = MayaModel(Settings(compile=True), device="cuda:0")
    m.device = "cuda:0"
    calls = []
    m.generate_chunk = lambda *a, **k: calls.append((a, k))
    m._warmup()
    assert calls == []


def test_warmup_runs_on_cuda_and_never_raises():
    m = MayaModel(Settings(), device="cuda:0")
    m.device = "cuda:0"
    calls = []

    def boom(*a, **k):
        calls.append((a, k))
        raise RuntimeError("no real GPU in the unit env")

    m.generate_chunk = boom
    m._warmup()  # must swallow the error, not propagate it
    assert len(calls) == 1


def test_enable_compile_puts_backbone_and_depth_decoder_on_static_cache():
    model = MayaModel(Settings(compile=True))
    model.model = types.SimpleNamespace(
        generation_config=types.SimpleNamespace(),
        depth_decoder=types.SimpleNamespace(generation_config=types.SimpleNamespace()),
    )
    model._enable_compile()
    gc = model.model.generation_config
    assert gc.cache_implementation == "static"
    assert gc.max_length == _COMPILE_MAX_LENGTH
    assert gc.max_new_tokens is None
    assert model.model.depth_decoder.generation_config.cache_implementation == "static"

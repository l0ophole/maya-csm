"""Unit tests for the torch-free helpers in maya_csm.model.

The model-loading path itself needs weights + heavy deps and is covered by
tests/test_integration.py (`-m model`).
"""

import types

from maya_csm.config import Settings
from maya_csm.model import _COMPILE_MAX_LENGTH, MayaModel, _resolve_dtype, resolve_devices


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

"""ModelPool dispatch behaviour and build_engine wiring (torch-free)."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from maya_csm import model as model_mod
from maya_csm.config import Settings
from maya_csm.model import ModelPool, build_engine


class FakeReplica:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def generate_chunk(self, text, gen_params, context=()):
        self.calls.append(text)
        return f"{self.name}:{text}"


class BlockingReplica(FakeReplica):
    """Records how many replicas are mid-generation at once."""

    _lock = threading.Lock()
    _active = 0
    peak = 0

    def generate_chunk(self, text, gen_params, context=()):
        with BlockingReplica._lock:
            BlockingReplica._active += 1
            BlockingReplica.peak = max(BlockingReplica.peak, BlockingReplica._active)
        time.sleep(0.05)
        with BlockingReplica._lock:
            BlockingReplica._active -= 1
        return super().generate_chunk(text, gen_params, context)


def test_num_replicas_reflects_pool_size():
    assert ModelPool([FakeReplica("a"), FakeReplica("b")]).num_replicas == 2


def test_two_replicas_generate_concurrently():
    BlockingReplica.peak = 0
    pool = ModelPool([BlockingReplica("a"), BlockingReplica("b")])
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(lambda t: pool.generate_chunk(t, {}), ["one", "two", "three", "four"]))
    assert BlockingReplica.peak == 2


def test_replicas_are_returned_after_use():
    a, b = FakeReplica("a"), FakeReplica("b")
    pool = ModelPool([a, b])
    for t in ["x", "y", "z", "w"]:
        pool.generate_chunk(t, {})
    # every replica handed back: a fresh full-width batch still succeeds
    with ThreadPoolExecutor(max_workers=2) as ex:
        out = list(ex.map(lambda t: pool.generate_chunk(t, {}), ["p", "q"]))
    assert sorted(out) == ["a:p", "b:q"] or sorted(out) == ["a:q", "b:p"]


def test_replica_returned_to_pool_when_generate_raises():
    class Boom(FakeReplica):
        def generate_chunk(self, text, gen_params, context=()):
            raise RuntimeError("kaboom")

    pool = ModelPool([Boom("a")])
    with pytest.raises(RuntimeError, match="kaboom"):
        pool.generate_chunk("hi", {})
    # replica was not leaked: the next call also gets to run (and raise) rather than hang
    with pytest.raises(RuntimeError, match="kaboom"):
        pool.generate_chunk("hi again", {})


def test_build_engine_returns_bare_model_for_one_device(monkeypatch):
    loaded = []
    monkeypatch.setattr(model_mod.MayaModel, "load", lambda self: loaded.append(self._device_override))
    engine = build_engine(Settings(devices=("cpu",)))
    assert isinstance(engine, model_mod.MayaModel)
    assert loaded == ["cpu"]


def test_build_engine_returns_pool_for_multiple_devices(monkeypatch):
    loaded = []
    monkeypatch.setattr(model_mod.MayaModel, "load", lambda self: loaded.append(self._device_override))
    engine = build_engine(Settings(devices=("cuda:0", "cuda:1")))
    assert isinstance(engine, ModelPool)
    assert engine.num_replicas == 2
    assert loaded == ["cuda:0", "cuda:1"]

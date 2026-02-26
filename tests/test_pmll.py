"""
Tests for pufferlib/PMLL.py — Persistent Memory Logic Loop (PMLL)

Covers: utilities, JSONLStore, PythonBackend, MemoryController,
        Promise, make_backend, and Torch-dependent components
        (cosine_sim_torch, PMLAttention, PMLLTransformer, PMLLPolicyMixin).
"""

import json
import math
import os
import tempfile
import time

import pytest
import numpy as np
import torch
import torch.nn as nn

from pufferlib.PMLL import (
    _stable_json_dumps,
    deterministic_hash,
    MemoryBlock,
    JSONLStore,
    PythonBackend,
    CTypesBackend,
    make_backend,
    Promise,
    MemoryController,
    cosine_sim_torch,
    PMLAttention,
    PMLLTransformer,
    PMLLPolicyMixin,
)


# ============================================================================
# Utilities
# ============================================================================

class TestStableJsonDumps:
    def test_sorted_keys(self):
        assert _stable_json_dumps({"b": 2, "a": 1}) == '{"a":1,"b":2}'

    def test_no_extra_spaces(self):
        result = _stable_json_dumps({"key": "value"})
        assert " " not in result

    def test_nested(self):
        result = _stable_json_dumps({"a": {"c": 3, "b": 2}})
        assert result == '{"a":{"b":2,"c":3}}'


class TestDeterministicHash:
    def test_same_input_same_hash(self):
        h1 = deterministic_hash({"x": 1})
        h2 = deterministic_hash({"x": 1})
        assert h1 == h2

    def test_different_input_different_hash(self):
        h1 = deterministic_hash({"x": 1})
        h2 = deterministic_hash({"x": 2})
        assert h1 != h2

    def test_salt_changes_hash(self):
        h1 = deterministic_hash({"x": 1}, salt="a")
        h2 = deterministic_hash({"x": 1}, salt="b")
        assert h1 != h2

    def test_returns_hex_string(self):
        h = deterministic_hash("hello")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest


class TestMemoryBlock:
    def test_creation(self):
        blk = MemoryBlock(payload={"k": 1}, mid="abc", ts=1.0)
        assert blk.payload == {"k": 1}
        assert blk.mid == "abc"
        assert blk.ts == 1.0
        assert blk.meta is None

    def test_with_meta(self):
        blk = MemoryBlock(payload={}, mid="x", ts=0.0, meta={"ttl": 10})
        assert blk.meta == {"ttl": 10}


# ============================================================================
# JSONLStore
# ============================================================================

class TestJSONLStore:
    def test_append_and_load_from_log(self):
        with tempfile.TemporaryDirectory() as td:
            store = JSONLStore(td)
            blk = MemoryBlock(payload={"a": 1}, mid="m1", ts=1.0)
            store.append(blk)

            loaded = store.load()
            assert len(loaded) == 1
            assert loaded[0].mid == "m1"
            assert loaded[0].payload == {"a": 1}

    def test_snapshot_preferred_over_log(self):
        with tempfile.TemporaryDirectory() as td:
            store = JSONLStore(td)
            blk_log = MemoryBlock(payload={"src": "log"}, mid="log1", ts=1.0)
            store.append(blk_log)

            blk_snap = MemoryBlock(payload={"src": "snap"}, mid="snap1", ts=2.0)
            store.save_snapshot([blk_snap])

            loaded = store.load()
            assert len(loaded) == 1
            assert loaded[0].mid == "snap1"

    def test_load_empty(self):
        with tempfile.TemporaryDirectory() as td:
            store = JSONLStore(td)
            assert store.load() == []

    def test_multiple_appends(self):
        with tempfile.TemporaryDirectory() as td:
            store = JSONLStore(td)
            for i in range(5):
                store.append(MemoryBlock(payload={"i": i}, mid=f"m{i}", ts=float(i)))

            loaded = store.load()
            assert len(loaded) == 5
            assert [b.mid for b in loaded] == [f"m{i}" for i in range(5)]


# ============================================================================
# PythonBackend
# ============================================================================

class TestPythonBackend:
    def test_phi_modulo(self):
        b = PythonBackend()
        assert b.phi(10, 8) == 2
        assert b.phi(0, 5) == 0
        assert b.phi(7, 7) == 0

    def test_utilization_initial(self):
        b = PythonBackend()
        assert b.utilization() == 0.0

    def test_set_utilization_clamped(self):
        b = PythonBackend()
        b._set_utilization(0.5)
        assert b.utilization() == 0.5
        b._set_utilization(1.5)
        assert b.utilization() == 1.0
        b._set_utilization(-0.5)
        assert b.utilization() == 0.0

    def test_process_promise_queue_noop(self):
        b = PythonBackend()
        b.process_promise_queue()  # should not raise

    def test_trigger_compression_noop(self):
        b = PythonBackend()
        b.trigger_compression(0.1)  # should not raise

    def test_vectorized_attention_returns_none(self):
        b = PythonBackend()
        assert b.vectorized_attention(None, None, None) is None


# ============================================================================
# make_backend
# ============================================================================

class TestMakeBackend:
    def test_default_returns_python_backend(self):
        b = make_backend()
        assert isinstance(b, PythonBackend)

    def test_invalid_so_path_falls_back(self):
        b = make_backend(so_path="/nonexistent/path.so")
        assert isinstance(b, PythonBackend)


# ============================================================================
# Promise
# ============================================================================

class TestPromise:
    def test_not_expired(self):
        p = Promise(pid=1, data="x", ttl_s=100.0, importance=0.5, created_ts=time.time())
        assert not p.expired(time.time())

    def test_expired(self):
        p = Promise(pid=1, data="x", ttl_s=1.0, importance=0.5, created_ts=time.time() - 2.0)
        assert p.expired(time.time())

    def test_exact_boundary(self):
        now = time.time()
        p = Promise(pid=1, data="x", ttl_s=5.0, importance=0.5, created_ts=now - 5.0)
        assert p.expired(now)


# ============================================================================
# MemoryController
# ============================================================================

class TestMemoryController:
    def test_write_and_process(self):
        mc = MemoryController(pool_size=64, store_dir=None)
        mc.write(pid=42, data={"hello": "world"}, ttl_s=60.0, importance=0.9)
        mc.process_promises()

        slot = mc.backend.phi(42, 64)
        assert mc.read_slot(slot) == {"hello": "world"}

    def test_utilization_increases(self):
        mc = MemoryController(pool_size=10, store_dir=None)
        assert mc.utilization() == 0.0

        mc.write(pid=0, data="a", ttl_s=60.0, importance=1.0)
        mc.process_promises()
        assert mc.utilization() > 0.0

    def test_expired_promises_not_stored(self):
        mc = MemoryController(pool_size=64, store_dir=None)
        mc.write(pid=99, data="old", ttl_s=0.0, importance=0.5)
        time.sleep(0.01)
        mc.process_promises()

        slot = mc.backend.phi(99, 64)
        assert mc.read_slot(slot) is None

    def test_read_slot_wraps(self):
        mc = MemoryController(pool_size=10, store_dir=None)
        # slot 100 % 10 == 0
        assert mc.read_slot(100) is None

    def test_multiple_writes_same_slot(self):
        mc = MemoryController(pool_size=64, store_dir=None)
        mc.write(pid=5, data="first", ttl_s=60.0, importance=1.0)
        mc.process_promises()
        mc.write(pid=5, data="second", ttl_s=60.0, importance=1.0)
        mc.process_promises()

        slot = mc.backend.phi(5, 64)
        assert mc.read_slot(slot) == "second"

    def test_default_backend(self):
        mc = MemoryController(pool_size=8)
        assert isinstance(mc.backend, PythonBackend)

    def test_pool_size(self):
        mc = MemoryController(pool_size=32)
        assert mc.pool_size == 32
        assert len(mc.pool) == 32

    def test_with_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            mc = MemoryController(pool_size=16, store_dir=td)
            mc.write(pid=7, data="persisted", ttl_s=60.0, importance=0.8)
            mc.process_promises()

            # Check that the log file was written
            assert os.path.exists(mc.store.log_path)
            loaded = mc.store.load()
            assert len(loaded) == 1

    def test_process_empty_queue(self):
        mc = MemoryController(pool_size=8, store_dir=None)
        mc.process_promises()  # should not raise


class _FailingBackend(PythonBackend):
    """Backend whose trigger_compression raises, forcing the python fallback."""
    def trigger_compression(self, rho: float) -> None:
        raise RuntimeError("no native compression")


class TestMemoryControllerCompression:
    def test_trigger_compression_python_fallback(self):
        backend = _FailingBackend()
        mc = MemoryController(pool_size=20, backend=backend, store_dir=None,
                              enable_python_compress_fallback=True,
                              compress_when_util_gt=1.1)  # prevent auto-compress
        # Fill the pool
        for i in range(20):
            mc.write(pid=i, data=f"data_{i}", ttl_s=3600.0, importance=1.0)
        mc.process_promises()

        occupied_before = mc._occupied
        assert occupied_before == 20

        np.random.seed(42)
        mc.trigger_compression(rho=0.5)
        # Some slots should have been cleared by the python fallback
        assert mc._occupied < occupied_before

    def test_compression_noop_with_python_backend(self):
        """PythonBackend.trigger_compression is a no-op, so the controller
        returns early without reaching the fallback path."""
        mc = MemoryController(pool_size=10, store_dir=None,
                              enable_python_compress_fallback=True)
        for i in range(10):
            mc.write(pid=i, data=f"d{i}", ttl_s=3600.0, importance=1.0)
        mc.process_promises()

        mc.trigger_compression(rho=0.5)
        # PythonBackend.trigger_compression is a no-op (doesn't raise),
        # so controller returns immediately without pruning
        assert mc._occupied == 10

    def test_compression_disabled(self):
        backend = _FailingBackend()
        mc = MemoryController(pool_size=10, backend=backend, store_dir=None,
                              enable_python_compress_fallback=False)
        for i in range(10):
            mc.write(pid=i, data=f"d{i}", ttl_s=3600.0, importance=1.0)
        mc.process_promises()

        mc.trigger_compression(rho=0.5)
        # Fallback disabled, backend raises => nothing changes
        assert mc._occupied == 10


class TestMemoryControllerRetrieve:
    def test_retrieve_with_tensor_entries(self):
        mc = MemoryController(pool_size=64, store_dir=None)
        k1 = torch.tensor([1.0, 0.0, 0.0])
        v1 = torch.tensor([10.0, 20.0, 30.0])
        mc.write(pid=1, data=(k1, v1), ttl_s=3600.0, importance=1.0)
        mc.process_promises()

        query = torch.tensor([1.0, 0.0, 0.0])
        results = mc.retrieve_relevant(query, threshold=0.5)
        assert len(results) >= 1

    def test_retrieve_empty_pool(self):
        mc = MemoryController(pool_size=8, store_dir=None)
        query = torch.tensor([1.0, 0.0, 0.0])
        results = mc.retrieve_relevant(query)
        assert results == []

    def test_retrieve_with_threshold(self):
        mc = MemoryController(pool_size=64, store_dir=None)
        # Store an entry orthogonal to query
        k = torch.tensor([0.0, 1.0, 0.0])
        v = torch.tensor([1.0, 1.0, 1.0])
        mc.write(pid=1, data=(k, v), ttl_s=3600.0, importance=1.0)
        mc.process_promises()

        query = torch.tensor([1.0, 0.0, 0.0])
        # With high threshold, orthogonal vectors shouldn't match
        results = mc.retrieve_relevant(query, threshold=0.9)
        assert len(results) == 0

    def test_retrieve_max_items(self):
        mc = MemoryController(pool_size=64, store_dir=None)
        for i in range(10):
            k = torch.randn(4)
            mc.write(pid=i, data=(k, k), ttl_s=3600.0, importance=1.0)
        mc.process_promises()

        query = torch.randn(4)
        results = mc.retrieve_relevant(query, threshold=-1.0, max_items=3)
        assert len(results) <= 3


# ============================================================================
# Torch: cosine_sim_torch
# ============================================================================

class TestCosineSim:
    def test_identical_vectors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        sim = cosine_sim_torch(a, a)
        assert abs(float(sim) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        sim = cosine_sim_torch(a, b)
        assert abs(float(sim)) < 1e-5

    def test_opposite_vectors(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        sim = cosine_sim_torch(a, b)
        assert float(sim) < -0.99


# ============================================================================
# Torch: PMLAttention
# ============================================================================

class TestPMLAttention:
    def test_forward_no_persistent(self):
        mc = MemoryController(pool_size=32, store_dir=None)
        attn = PMLAttention(mc)

        q = torch.randn(2, 8)
        k_local = torch.randn(2, 4, 8)
        v_local = torch.randn(2, 4, 8)

        out = attn(q, k_local, v_local)
        assert out.shape == (8,)

    def test_forward_with_persistent(self):
        mc = MemoryController(pool_size=64, store_dir=None)
        attn = PMLAttention(mc, persistent_threshold=-1.0)

        # Pre-fill memory with entries similar to the query
        q_vec = torch.randn(8)
        for i in range(5):
            k = q_vec + torch.randn(8) * 0.01
            v = torch.randn(8)
            mc.write(pid=i, data=(k, v), ttl_s=3600.0, importance=1.0)
        mc.process_promises()

        q = q_vec.unsqueeze(0)  # [1, 8]
        k_local = torch.randn(1, 3, 8)
        v_local = torch.randn(1, 3, 8)

        out = attn(q, k_local, v_local)
        assert out.shape == (8,)

    def test_attend_1d_query(self):
        mc = MemoryController(pool_size=8, store_dir=None)
        attn = PMLAttention(mc)

        q = torch.randn(4)  # 1D query
        k = torch.randn(3, 4)
        v = torch.randn(3, 4)
        out = attn._attend(q, k, v)
        assert out.shape == (4,)

    def test_extract_kv_tuples(self):
        mc = MemoryController(pool_size=8, store_dir=None)
        attn = PMLAttention(mc)

        entries = [
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
            (torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])),
        ]
        ks, vs = attn._extract_kv(entries)
        assert ks.shape == (2, 2)
        assert vs.shape == (2, 2)

    def test_extract_kv_dicts(self):
        mc = MemoryController(pool_size=8, store_dir=None)
        attn = PMLAttention(mc)

        entries = [
            {"k": torch.tensor([1.0]), "v": torch.tensor([2.0])},
        ]
        ks, vs = attn._extract_kv(entries)
        assert ks.shape == (1, 1)


# ============================================================================
# Torch: PMLLTransformer
# ============================================================================

class TestPMLLTransformer:
    def test_forward_shape(self):
        model = PMLLTransformer(d_model=16, pool_size=32)
        x = torch.randn(4, 16)
        out = model(x)
        assert out.shape == (16,)

    def test_encoder_params(self):
        model = PMLLTransformer(d_model=8, pool_size=16)
        params = list(model.parameters())
        assert len(params) > 0

    def test_multiple_forward_passes(self):
        model = PMLLTransformer(d_model=8, pool_size=64)
        for _ in range(3):
            x = torch.randn(2, 8)
            out = model(x)
            assert out.shape == (8,)


# ============================================================================
# PMLLPolicyMixin
# ============================================================================

class TestPMLLPolicyMixin:
    def test_mixin_write_and_process(self):
        mc = MemoryController(pool_size=32, store_dir=None)
        mixin = PMLLPolicyMixin(mc)

        mixin.pmll_write(pid=10, data="test_data", ttl_s=60.0, importance=0.7)
        mixin.pmll_process()

        slot = mc.backend.phi(10, 32)
        assert mc.read_slot(slot) == "test_data"

    def test_mixin_write_kv(self):
        mc = MemoryController(pool_size=32, store_dir=None)
        mixin = PMLLPolicyMixin(mc)

        k = torch.randn(8)
        v = torch.randn(8)
        mixin.pmll_write_kv(k, v, pid=42, ttl_s=60.0)
        mixin.pmll_process()

        slot = mc.backend.phi(42, 32)
        entry = mc.read_slot(slot)
        assert isinstance(entry, tuple)
        assert torch.allclose(entry[0], k)
        assert torch.allclose(entry[1], v)

    def test_mixin_write_kv_v_none(self):
        mc = MemoryController(pool_size=32, store_dir=None)
        mixin = PMLLPolicyMixin(mc)

        k = torch.randn(4)
        mixin.pmll_write_kv(k, pid=7, ttl_s=60.0)
        mixin.pmll_process()

        slot = mc.backend.phi(7, 32)
        entry = mc.read_slot(slot)
        assert isinstance(entry, tuple)
        # v should default to k
        assert torch.allclose(entry[0], entry[1])

    def test_mixin_step_counter(self):
        mc = MemoryController(pool_size=32, store_dir=None)
        mixin = PMLLPolicyMixin(mc)
        assert mixin._pmll_step == 0
        mixin.pmll_write_kv(torch.randn(4))
        assert mixin._pmll_step == 1
        mixin.pmll_write_kv(torch.randn(4))
        assert mixin._pmll_step == 2

    def test_mixin_with_nn_module(self):
        """Verify the mixin pattern works with nn.Module as documented."""
        mc = MemoryController(pool_size=16, store_dir=None)

        class SimplePolicy(nn.Module, PMLLPolicyMixin):
            def __init__(self):
                nn.Module.__init__(self)
                PMLLPolicyMixin.__init__(self, pmll=mc)
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                out = self.linear(x)
                self.pmll_write_kv(out)
                self.pmll_process()
                return out

        policy = SimplePolicy()
        x = torch.randn(2, 4)
        out = policy(x)
        assert out.shape == (2, 4)


# ============================================================================
# Smoke test (same as __main__ but as a proper test)
# ============================================================================

class TestSmokeTest:
    def test_basic_smoke(self):
        """Mirrors the __main__ smoke test."""
        backend = make_backend()
        mc = MemoryController(pool_size=128, backend=backend, store_dir=None)

        mc.write(pid=123, data={"hello": "world"}, ttl_s=10.0, importance=0.9)
        mc.process_promises()
        slot = backend.phi(123, 128)
        assert mc.read_slot(slot) == {"hello": "world"}

    def test_torch_smoke(self):
        """Mirrors the torch section of __main__."""
        model = PMLLTransformer(d_model=32, pool_size=256)
        x = torch.randn(8, 32)
        y = model(x)
        assert y.shape == (32,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

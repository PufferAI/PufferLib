"""
pmll.py (PufferLib-ready) — Persistent Memory Logic Loop (PMLL)

Refactor goals (vs. your original draft):
- Clean layering: Backend interface + CTypes backend + pure-Python fallback
- Optional native acceleration: libpmll_backend.so (SIMD/intrinsics) if present
- Thread-safe / async-safe memory controller
- PufferLib integration: usable as a plug-in “memory module” inside RL policies
- Torch integration: PML attention block that can be inserted into a network
- No hard dependency on torch/numpy unless you use the corresponding features

Notes:
- This file does NOT assume a specific pufferlib policy API, but provides
  adapters that work with typical “nn.Module policy + forward(obs)” patterns.
- If you have a concrete PufferLib policy base class you’re using, you can
  subclass/mixin the PMLLPolicyMixin below and call its helpers.

License: MIT
"""

from __future__ import annotations

import os
import time
import json
import math
import ctypes
import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable

# Optional deps (only used if available/needed)
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


# =============================================================================
# Utilities: stable hashing + JSONL persistence
# =============================================================================

def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def deterministic_hash(payload: Any, salt: str = "") -> str:
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(_stable_json_dumps(payload).encode("utf-8"))
    return h.hexdigest()

@dataclass
class MemoryBlock:
    payload: Dict[str, Any]
    mid: str
    ts: float
    meta: Optional[Dict[str, Any]] = None

class JSONLStore:
    """
    Simple append-only log + optional periodic snapshot.
    Designed for long runs (RL training) without loading everything each time.
    """
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.log_path = os.path.join(root, "pmll_log.jsonl")
        self.snapshot_path = os.path.join(root, "pmll_snapshot.json")

    def append(self, block: MemoryBlock) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(_stable_json_dumps(block.__dict__) + "\n")

    def save_snapshot(self, blocks: List[MemoryBlock]) -> None:
        with open(self.snapshot_path, "w", encoding="utf-8") as f:
            f.write(_stable_json_dumps([b.__dict__ for b in blocks]))

    def load(self) -> List[MemoryBlock]:
        # Prefer snapshot for faster cold start
        if os.path.exists(self.snapshot_path):
            try:
                with open(self.snapshot_path, "r", encoding="utf-8") as f:
                    arr = json.loads(f.read())
                return [MemoryBlock(**x) for x in arr]
            except Exception:
                pass

        blocks: List[MemoryBlock] = []
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    blocks.append(MemoryBlock(**json.loads(line)))
        return blocks


# =============================================================================
# Backend interface (native or python)
# =============================================================================

@runtime_checkable
class PMLLBackend(Protocol):
    def phi(self, idx: int, n: int) -> int:
        ...

    def process_promise_queue(self) -> None:
        ...

    def trigger_compression(self, rho: float) -> None:
        ...

    def utilization(self) -> float:
        ...

    # Optional accelerated attention (batched)
    def vectorized_attention(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor"
    ) -> Optional["torch.Tensor"]:
        ...


class PythonBackend:
    """
    Pure-Python fallback backend:
    - phi via modular arithmetic
    - no native promise queue (controller handles it)
    - no native compression (controller can do python compression)
    - utilization computed in controller
    """
    def __init__(self):
        self._util = 0.0

    def phi(self, idx: int, n: int) -> int:
        # collision-minimizing slot assignment (simple modulo)
        return idx % n

    def process_promise_queue(self) -> None:
        return

    def trigger_compression(self, rho: float) -> None:
        return

    def utilization(self) -> float:
        return float(self._util)

    def _set_utilization(self, u: float) -> None:
        self._util = max(0.0, min(1.0, float(u)))

    def vectorized_attention(self, q, k, v):
        return None


class CTypesBackend:
    """
    Optional native backend for SIMD/intrinsics acceleration.

    Expects a shared library exposing:
      int phi(int id, int n)
      void process_promise_queue(PromiseQueue*, MemoryPool*)
      void trigger_compression(MemoryPool*, float rho)
      void vectorized_attention(float* q, float* k, float* v, int d)

    Important:
    - This backend only accelerates specific operations; the controller still
      owns the Python-level logic and safety.
    """
    class MemoryPool(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_int),
            ("data", ctypes.POINTER(ctypes.c_void_p)),
            ("utilization", ctypes.c_float),
        ]

    class PromiseQueue(ctypes.Structure):
        _fields_ = [
            ("capacity", ctypes.c_int),
            ("head", ctypes.c_int),
            ("tail", ctypes.c_int),
            ("promises", ctypes.POINTER(ctypes.c_void_p)),
        ]

    def __init__(self, so_path: str):
        self.lib = ctypes.CDLL(so_path)

        # signatures
        self.lib.phi.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.phi.restype = ctypes.c_int

        # Optional exports; allow absent symbols (keep graceful)
        self._has_pq = hasattr(self.lib, "process_promise_queue")
        if self._has_pq:
            self.lib.process_promise_queue.argtypes = [
                ctypes.POINTER(self.PromiseQueue),
                ctypes.POINTER(self.MemoryPool),
            ]
            self.lib.process_promise_queue.restype = ctypes.POINTER(self.MemoryPool)

        self._has_comp = hasattr(self.lib, "trigger_compression")
        if self._has_comp:
            self.lib.trigger_compression.argtypes = [ctypes.POINTER(self.MemoryPool), ctypes.c_float]
            self.lib.trigger_compression.restype = None

        self._has_attn = hasattr(self.lib, "vectorized_attention")
        if self._has_attn:
            self.lib.vectorized_attention.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            self.lib.vectorized_attention.restype = None

        # Allocate minimal structs; you can extend this to mirror your C layout fully.
        self.c_pool = self.MemoryPool(0, None, 0.0)
        self.c_queue = self.PromiseQueue(0, 0, 0, None)

    def phi(self, idx: int, n: int) -> int:
        return int(self.lib.phi(int(idx), int(n)))

    def process_promise_queue(self) -> None:
        if self._has_pq:
            self.lib.process_promise_queue(self.c_queue, self.c_pool)

    def trigger_compression(self, rho: float) -> None:
        if self._has_comp:
            self.lib.trigger_compression(self.c_pool, ctypes.c_float(float(rho)))

    def utilization(self) -> float:
        return float(self.c_pool.utilization)

    def vectorized_attention(self, q, k, v):
        # If no torch or symbol, skip
        if torch is None or not self._has_attn:
            return None
        # Expect float32 contiguous
        qf = q.contiguous().float()
        kf = k.contiguous().float()
        vf = v.contiguous().float()
        d = qf.shape[-1]
        # Only safe for CPU tensors
        if qf.is_cuda or kf.is_cuda or vf.is_cuda:
            return None

        q_ptr = ctypes.cast(qf.data_ptr(), ctypes.POINTER(ctypes.c_float))
        k_ptr = ctypes.cast(kf.data_ptr(), ctypes.POINTER(ctypes.c_float))
        v_ptr = ctypes.cast(vf.data_ptr(), ctypes.POINTER(ctypes.c_float))
        self.lib.vectorized_attention(q_ptr, k_ptr, v_ptr, ctypes.c_int(int(d)))
        return None  # in-place or just an accelerator hint


def make_backend(
    *,
    so_path: Optional[str] = None,
) -> PMLLBackend:
    """
    Factory:
    - If so_path provided and loadable => CTypesBackend
    - Else => PythonBackend
    """
    if so_path:
        try:
            return CTypesBackend(so_path)
        except Exception:
            pass
    return PythonBackend()


# =============================================================================
# Core: Promise + MemoryController (PufferLib-friendly)
# =============================================================================

@dataclass
class Promise:
    pid: int
    data: Any
    ttl_s: float
    importance: float
    created_ts: float

    def expired(self, now: float) -> bool:
        return (now - self.created_ts) >= self.ttl_s


class MemoryController:
    """
    PMLL memory controller with:
    - slotting via backend.phi
    - promise queue (write-behind)
    - compression triggers
    - thread safety

    Intended use in RL:
    - At each environment step, write (state->kv) or (obs->features) as promises
    - Periodically process promises into the persistent pool
    - Retrieve relevant entries for attention / policy decisions
    """
    def __init__(
        self,
        pool_size: int,
        *,
        backend: Optional[PMLLBackend] = None,
        hash_salt: str = "",
        store_dir: Optional[str] = None,
        snapshot_every: int = 500,
        default_ttl_s: float = 3600.0,
        compression_rho: float = 0.10,
        compress_when_util_gt: float = 0.80,
        enable_python_compress_fallback: bool = True,
    ):
        self.pool_size = int(pool_size)
        self.backend = backend or make_backend()
        self.hash_salt = hash_salt

        self.default_ttl_s = float(default_ttl_s)
        self.compression_rho = float(compression_rho)
        self.compress_when_util_gt = float(compress_when_util_gt)
        self.enable_python_compress_fallback = bool(enable_python_compress_fallback)

        self._lock = threading.RLock()

        # Persistent pool: slots of arbitrary python objects
        self.pool: List[Optional[Any]] = [None] * self.pool_size

        # Promise queue: write-behind buffer
        self.promises: List[Promise] = []

        # Persistence (optional)
        self.store = JSONLStore(store_dir) if store_dir else None
        self.snapshot_every = max(1, int(snapshot_every))
        self._append_count = 0

        # Track occupancy for python backend utilization
        self._occupied = 0

    # ------------- Promise write path -------------

    def write(
        self,
        *,
        pid: int,
        data: Any,
        ttl_s: Optional[float] = None,
        importance: Optional[float] = None
    ) -> None:
        """
        Enqueue a promise to write into persistent memory.
        """
        now = time.time()
        ttl = self.default_ttl_s if ttl_s is None else float(ttl_s)
        imp = float(importance) if importance is not None else self._importance_score(data)

        p = Promise(pid=int(pid), data=data, ttl_s=ttl, importance=imp, created_ts=now)
        with self._lock:
            self.promises.append(p)

    def process_promises(self) -> None:
        """
        Flush non-expired promises into pool slots.
        Native backend may also do internal housekeeping.
        """
        now = time.time()

        with self._lock:
            if not self.promises:
                self._update_utilization_locked()
                return

            keep: List[Promise] = []
            for p in self.promises:
                if p.expired(now):
                    continue
                slot = self.backend.phi(p.pid, self.pool_size)
                was_empty = (self.pool[slot] is None)
                self.pool[slot] = p.data
                if was_empty:
                    self._occupied += 1

                # Persist a compact memory block (optional)
                if self.store:
                    blk = MemoryBlock(
                        payload={
                            "pid": p.pid,
                            "slot": slot,
                            "importance": p.importance,
                            "ts": now,
                        },
                        mid=deterministic_hash({"pid": p.pid, "slot": slot, "ts": now}, salt=self.hash_salt),
                        ts=now,
                        meta={"ttl_s": p.ttl_s},
                    )
                    self.store.append(blk)
                    self._append_count += 1
                    if (self._append_count % self.snapshot_every) == 0:
                        # snapshot only metadata log, not full pool
                        # (pool entries may be tensors / non-serializable)
                        self.store.save_snapshot([blk])

                # promise is consumed (write-behind); drop it
            self.promises = keep

            # Let native backend optionally process its internal PQ
            try:
                self.backend.process_promise_queue()
            except Exception:
                pass

            self._update_utilization_locked()

        # Compression check outside lock is fine
        if self.utilization() > self.compress_when_util_gt:
            self.trigger_compression(self.compression_rho)

    # ------------- Read / retrieve path -------------

    def read_slot(self, slot: int) -> Any:
        with self._lock:
            return self.pool[int(slot) % self.pool_size]

    def retrieve_relevant(
        self,
        query: Any,
        *,
        threshold: float = 0.50,
        max_items: int = 64,
        key_extractor: Optional[Any] = None,
        similarity: Optional[Any] = None,
    ) -> List[Any]:
        """
        Retrieve relevant pool entries based on a query.
        - Works with torch tensors if provided
        - key_extractor: maps entry -> key embedding (default assumes entry is (k,v) or dict)
        - similarity: function(query, key) -> float
        """
        if torch is None:
            return []

        if similarity is None:
            similarity = cosine_sim_torch

        if key_extractor is None:
            def key_extractor(x):
                # Default: expect (k, v) tuple
                if isinstance(x, tuple) and len(x) >= 1:
                    return x[0]
                if isinstance(x, dict) and "k" in x:
                    return x["k"]
                return None

        q = query
        hits: List[Tuple[float, Any]] = []
        with self._lock:
            for entry in self.pool:
                if entry is None:
                    continue
                k = key_extractor(entry)
                if k is None:
                    continue
                try:
                    s = float(similarity(q, k))
                except Exception:
                    continue
                if s >= threshold:
                    hits.append((s, entry))

        hits.sort(key=lambda t: t[0], reverse=True)
        return [e for _, e in hits[: int(max_items)]]

    # ------------- Compression -------------

    def trigger_compression(self, rho: float = 0.10) -> None:
        """
        Ask backend to compress; fall back to python compression if enabled.
        """
        try:
            self.backend.trigger_compression(float(rho))
            return
        except Exception:
            pass

        if not self.enable_python_compress_fallback:
            return

        # Simple python compression: drop low-importance slots by heuristic
        if np is None:
            return

        with self._lock:
            scores = np.random.rand(self.pool_size)  # placeholder importance scores
            thresh = float(np.quantile(scores, float(rho)))
            for i in range(self.pool_size):
                if self.pool[i] is None:
                    continue
                if float(scores[i]) < thresh:
                    self.pool[i] = None
            self._occupied = sum(1 for x in self.pool if x is not None)
            self._update_utilization_locked()

    # ------------- Utilization -------------

    def utilization(self) -> float:
        # If native backend reports utilization, prefer it
        try:
            u = float(self.backend.utilization())
            if not math.isnan(u) and u > 0.0:
                return max(0.0, min(1.0, u))
        except Exception:
            pass

        with self._lock:
            return float(self._occupied) / float(self.pool_size)

    def _update_utilization_locked(self) -> None:
        u = float(self._occupied) / float(self.pool_size)
        # python backend can store util for debugging
        if isinstance(self.backend, PythonBackend):
            self.backend._set_utilization(u)

    # ------------- Importance scoring -------------

    def _importance_score(self, data: Any) -> float:
        # Replace with ERS / novelty / recency scoring if you want
        if np is not None:
            return float(np.random.rand())
        return 0.5


# =============================================================================
# Torch: Hybrid PML Attention block (drop-in nn.Module)
# =============================================================================

def cosine_sim_torch(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    a = a.flatten()
    b = b.flatten()
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-9)

class PMLAttention(nn.Module):
    """
    Hybrid attention:
      A = alpha * local_attention + (1 - alpha) * persistent_attention

    Expected shapes:
      q: [B, D] or [D]
      k_local/v_local: [T, D] or [B, T, D] (handled loosely)
      persistent entries: list of (k, v) with k/v compatible with q dims
    """
    def __init__(
        self,
        memory: MemoryController,
        *,
        persistent_threshold: float = 0.50,
        persistent_max_items: int = 64,
        native_attention_threshold: int = 32,
    ):
        super().__init__()
        self.memory = memory
        self.persistent_threshold = float(persistent_threshold)
        self.persistent_max_items = int(persistent_max_items)
        self.native_attention_threshold = int(native_attention_threshold)

    def forward(self, q: torch.Tensor, k_local: torch.Tensor, v_local: torch.Tensor) -> torch.Tensor:
        # Local attention
        a_local = self._attend(q, k_local, v_local)

        # Persistent retrieve
        rel = self.memory.retrieve_relevant(
            q,
            threshold=self.persistent_threshold,
            max_items=self.persistent_max_items
        )
        if not rel:
            return a_local

        k_p, v_p = self._extract_kv(rel)

        # Optional native acceleration hint
        if k_p.shape[0] > self.native_attention_threshold:
            try:
                self.memory.backend.vectorized_attention(q, k_p, v_p)
            except Exception:
                pass

        a_p = self._attend(q, k_p, v_p)
        alpha = self._alpha(q, k_local, k_p)
        return alpha * a_local + (1.0 - alpha) * a_p

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Normalize shapes to (T, D) with q (D,)
        if q.dim() == 2:
            # [B,D] -> use first batch row (common in tiny blocks); extend if you want per-batch
            qv = q[0]
        else:
            qv = q

        if k.dim() == 3:
            kv = k[0]
            vv = v[0]
        else:
            kv = k
            vv = v

        # scores: [T]
        scores = (kv @ qv) / math.sqrt(qv.shape[-1])
        w = torch.softmax(scores, dim=-1)
        out = (w.unsqueeze(-1) * vv).sum(dim=0)
        return out

    def _extract_kv(self, rel: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        ks = []
        vs = []
        for e in rel:
            if isinstance(e, tuple) and len(e) >= 2:
                ks.append(e[0])
                vs.append(e[1])
            elif isinstance(e, dict) and "k" in e and "v" in e:
                ks.append(e["k"])
                vs.append(e["v"])
        return torch.stack(ks, dim=0), torch.stack(vs, dim=0)

    def _alpha(self, q: torch.Tensor, k_local: torch.Tensor, k_p: torch.Tensor) -> torch.Tensor:
        # Simple similarity-based blend (scalar)
        if q.dim() == 2:
            qv = q[0]
        else:
            qv = q
        if k_local.dim() == 3:
            kl = k_local[0]
        else:
            kl = k_local

        sim_local = torch.norm(qv - kl.mean(dim=0), p=2)
        sim_p = torch.norm(qv - k_p.mean(dim=0), p=2)
        return torch.sigmoid(sim_local - sim_p)


# =============================================================================
# PufferLib integration helpers
# =============================================================================

class PMLLPolicyMixin:
    """
    A light-weight mixin for PufferLib-like policies.

    Usage pattern (typical):
      class MyPolicy(nn.Module, PMLLPolicyMixin):
          def __init__(...):
              nn.Module.__init__(self)
              PMLLPolicyMixin.__init__(self, pmll=MemoryController(...))
              self.pmll_attn = PMLAttention(self.pmll)

          def forward(self, obs, state=None):
              features = self.encoder(obs)
              # optionally: store features/kv as promises
              self.pmll_write_kv(features)
              self.pmll_process()  # flush promises periodically
              # optionally: use PML attention
              ...
    """
    def __init__(self, pmll: MemoryController):
        self.pmll = pmll
        self._pmll_step = 0

    def pmll_write(self, pid: int, data: Any, ttl_s: Optional[float] = None, importance: Optional[float] = None) -> None:
        self.pmll.write(pid=pid, data=data, ttl_s=ttl_s, importance=importance)

    def pmll_process(self) -> None:
        self.pmll.process_promises()

    def pmll_write_kv(
        self,
        k: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        *,
        pid: Optional[int] = None,
        ttl_s: Optional[float] = None,
        importance: Optional[float] = None
    ) -> None:
        """
        Convenience: store (k, v) tuple. If v None, store (k, k).
        pid defaults to a rolling step hash.
        """
        if torch is None:
            return
        self._pmll_step += 1
        if pid is None:
            pid = hash((int(self._pmll_step), int(time.time())))
        if v is None:
            v = k
        self.pmll.write(pid=int(pid), data=(k.detach(), v.detach()), ttl_s=ttl_s, importance=importance)


# =============================================================================
# Optional: Minimal “Transformer-like” wrapper
# =============================================================================

class PMLLTransformer(nn.Module):
    """
    A minimal example wrapper showing how to insert PMLAttention into a model.

    This is NOT a drop-in replacement for torch.nn.Transformer.
    It’s a pufferlib-friendly building block: an encoder + memory + attention.
    """
    def __init__(self, d_model: int, pool_size: int = 1024, so_path: Optional[str] = None):
        if nn is None:
            raise ImportError("torch is required for PMLLTransformer.")
        super().__init__()
        self.d_model = int(d_model)
        self.memory = MemoryController(pool_size, backend=make_backend(so_path=so_path))
        self.attn = PMLAttention(self.memory)

        # Small encoder (replace with your policy backbone)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        feats = self.encoder(x)

        # local cache: pretend k_local/v_local are last T=1
        k_local = feats.unsqueeze(1)  # [B,1,D]
        v_local = feats.unsqueeze(1)

        # Use PML attention with q = feats
        out = self.attn(feats, k_local, v_local)

        # Write to persistent memory
        for b in range(x.shape[0]):
            pid = hash((b, time.time_ns()))
            self.memory.write(pid=pid, data=(feats[b].detach(), out.detach()), ttl_s=3600.0)

        self.memory.process_promises()
        return out


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    # Backend selection
    # - If you compiled a native backend, set env PMLL_SO=/path/to/libpmll_backend.so
    so = os.environ.get("PMLL_SO", None)

    backend = make_backend(so_path=so)
    mc = MemoryController(pool_size=128, backend=backend, store_dir=None)

    # Basic write/process/read
    mc.write(pid=123, data={"hello": "world"}, ttl_s=10.0, importance=0.9)
    mc.process_promises()
    slot = backend.phi(123, 128)
    print("slot:", slot, "value:", mc.read_slot(slot))

    # Torch test if available
    if torch is not None:
        model = PMLLTransformer(d_model=32, pool_size=256, so_path=so)
        x = torch.randn(8, 32)
        y = model(x)
        print("out:", y.shape, "util:", mc.utilization())

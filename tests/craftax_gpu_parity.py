"""CPU-env vs device-env parity for the craftax GPU_ENV build.

One process, one VecEnv (gpu=1): the same .so holds both the host envs
(stepped with cpu_step) and the device CuVec (stepped via gpu_step ->
my_gpu_step). Same seed, same action sequence -> obs/rewards/terminals
must match bitwise every step.
"""
import ctypes
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pufferlib import _C  # noqa: E402

N = int(os.environ.get("PARITY_ENVS", 256))
STEPS = int(os.environ.get("PARITY_STEPS", 500))
SEED = 42

class _CudaPtr:
    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {
            "data": (ptr, False), "shape": shape,
            "typestr": typestr, "version": 2,
        }

def cpu_view(ptr, shape, ctype, dtype):
    n = int(np.prod(shape))
    arr = (ctype * n).from_address(ptr)
    return np.frombuffer(arr, dtype=dtype).reshape(shape)

args = {
    "vec": {"total_agents": float(N), "num_buffers": 1.0, "num_threads": 8.0},
    "env": {"seed_offset": float(SEED), "reset_pool_size": 0.0,
            "lazy_floors": 0.0},
}
ve = _C.create_vec(args, 1)
print(f"vec: {ve.total_agents} agents obs_size={ve.obs_size} "
      f"dtype={ve.obs_dtype} act_sizes={ve.act_sizes}")
num_actions = ve.act_sizes[0]

obs_np_dtype = np.uint8 if ve.obs_dtype == "ByteTensor" else np.float32
obs_typestr = "|u1" if obs_np_dtype == np.uint8 else "<f4"
obs_ctype = ctypes.c_uint8 if obs_np_dtype == np.uint8 else ctypes.c_float

ve.reset()

# host-side (CPU env) views: pinned host buffers
h_obs = cpu_view(ve.obs_ptr, (N, ve.obs_size), obs_ctype, obs_np_dtype)
h_rew = cpu_view(ve.rewards_ptr, (N,), ctypes.c_float, np.float32)
h_term = cpu_view(ve.terminals_ptr, (N,), ctypes.c_float, np.float32)

# device-side (GPU env) views
d_obs = torch.as_tensor(_CudaPtr(ve.gpu_obs_ptr, (N, ve.obs_size), obs_typestr))
d_rew = torch.as_tensor(_CudaPtr(ve.gpu_rewards_ptr, (N,), "<f4"))
d_term = torch.as_tensor(_CudaPtr(ve.gpu_terminals_ptr, (N,), "<f4"))

ULP_TOL = int(os.environ.get("PARITY_ULP", 0))

def ulp_diff(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    return np.abs(ai - bi)

def compare(tag):
    torch.cuda.synchronize()
    g_obs = d_obs.cpu().numpy()
    g_rew = d_rew.cpu().numpy()
    g_term = d_term.cpu().numpy()
    ok = True
    if not np.array_equal(h_obs, g_obs):
        if ULP_TOL > 0 and obs_np_dtype == np.float32:
            d = ulp_diff(h_obs, g_obs)
            worst = d.max()
            nbad = int((d > 0).sum())
            if worst > ULP_TOL:
                i, j = np.unravel_index(int(d.argmax()), d.shape)
                print(f"[{tag}] OBS beyond {ULP_TOL} ULP: worst={worst} at "
                      f"env={i} idx={j} cpu={h_obs[i, j]} gpu={g_obs[i, j]} "
                      f"({nbad} cells differ)")
                ok = False
        else:
            bad = np.argwhere(h_obs != g_obs)
            i, j = bad[0]
            print(f"[{tag}] OBS mismatch: {bad.shape[0]} cells, first env={i} "
                  f"byte={j} cpu={h_obs[i, j]} gpu={g_obs[i, j]}")
            ok = False
    if not np.array_equal(h_rew, g_rew):
        bad = np.argwhere(h_rew != g_rew).ravel()
        print(f"[{tag}] REW mismatch: {bad.size} envs, first env={bad[0]} "
              f"cpu={h_rew[bad[0]]} gpu={g_rew[bad[0]]}")
        ok = False
    if not np.array_equal(h_term, g_term):
        bad = np.argwhere(h_term != g_term).ravel()
        print(f"[{tag}] TERM mismatch: {bad.size} envs, first env={bad[0]}")
        ok = False
    return ok

if not compare("reset"):
    print("FAIL at reset")
    sys.exit(1)
print("reset obs identical")

rng = np.random.default_rng(123)
act_host = np.zeros(N, dtype=np.float32)
act_dev = torch.zeros(N, dtype=torch.float32, device="cuda")

fails = 0
for t in range(STEPS):
    a = rng.integers(0, num_actions, size=N).astype(np.float32)
    act_host[:] = a
    act_dev.copy_(torch.from_numpy(a))
    torch.cuda.synchronize()
    ve.cpu_step(act_host.ctypes.data)
    ve.gpu_step(act_dev.data_ptr())
    if not compare(f"step {t}"):
        fails += 1
        if fails >= 3:
            print("FAIL: stopping after 3 mismatching steps")
            sys.exit(1)

term_total = 0
print(f"\n{STEPS} steps x {N} envs: {'PARITY OK' if fails == 0 else 'FAILED'}")
print("log:", ve.log())
sys.exit(0 if fails == 0 else 1)

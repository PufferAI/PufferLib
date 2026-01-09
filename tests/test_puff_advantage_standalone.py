"""
Standalone benchmark for puff_advantage CUDA kernels.

nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC puff_advantage_standalone.cu -o libpuff_advantage.so
python test_puff_advantage_standalone.py
"""

import torch
import ctypes
import time
import os
from pathlib import Path

TEST_CONFIGS = [
    (128, 64, "128x64 (small batch)"),
    (512, 64, "512x64 (medium batch)"),
    (1024, 64, "1024x64 (large batch)"),
    (2048, 64, "2048x64 (xlarge batch)"),
    
    (512, 16, "512x16 (short horizon)"),
    (512, 32, "512x32 (medium horizon)"),
    (512, 128, "512x128 (long horizon)"),
    (512, 256, "512x256 (very long horizon)"),
    
    (4096, 64, "4096x64 (many rows)"),
    (256, 512, "256x512 (very long horizon)"),
]

GAMMA = 0.995
GAE_LAMBDA = 0.90
RHO_CLIP = 1.0
C_CLIP = 1.0

WARMUP_ITERS = 10
BENCHMARK_ITERS = 100
RTOL = 1e-5
ATOL = 1e-6


def load_cuda_lib():
    script_dir = Path(__file__).parent
    lib_path = script_dir / "libpuff_advantage.so"
    
    if not lib_path.exists():
        print(f"ERROR: {lib_path} not found!")
        return None
    
    lib = ctypes.CDLL(str(lib_path))
    
    lib.launch_original.argtypes = [
        ctypes.c_void_p,  # values
        ctypes.c_void_p,  # rewards
        ctypes.c_void_p,  # dones
        ctypes.c_void_p,  # importance
        ctypes.c_void_p,  # advantages
        ctypes.c_float,   # gamma
        ctypes.c_float,   # lambda
        ctypes.c_float,   # rho_clip
        ctypes.c_float,   # c_clip
        ctypes.c_int,     # num_steps
        ctypes.c_int,     # horizon
    ]
    lib.launch_original.restype = None
    
    lib.launch_new.argtypes = lib.launch_original.argtypes
    lib.launch_new.restype = None
    
    lib.sync_device.argtypes = []
    lib.sync_device.restype = None
    
    lib.get_last_error.argtypes = []
    lib.get_last_error.restype = ctypes.c_char_p
    
    return lib


def create_test_tensors(num_steps: int, horizon: int, device: str = "cuda"):
    values = torch.randn(num_steps, horizon, device=device, dtype=torch.float32)
    rewards = torch.zeros(num_steps, horizon, device=device, dtype=torch.float32)
    reward_mask = torch.rand(num_steps, horizon, device=device) < 0.1
    rewards[reward_mask] = torch.randn(reward_mask.sum().item(), device=device)
    
    dones = torch.zeros(num_steps, horizon, device=device, dtype=torch.float32)
    done_mask = torch.rand(num_steps, horizon, device=device) < 0.05
    dones[done_mask] = 1.0
    
    importance = torch.ones(num_steps, horizon, device=device, dtype=torch.float32)
    importance += torch.randn(num_steps, horizon, device=device) * 0.1
    importance = torch.clamp(importance, min=0.1, max=2.0)
    
    return values, rewards, dones, importance


def run_kernel(lib, kernel_fn, values, rewards, dones, importance):
    """Run a kernel and return the advantages tensor."""
    num_steps, horizon = values.shape
    advantages = torch.zeros_like(values)
    
    kernel_fn(
        ctypes.c_void_p(values.data_ptr()),
        ctypes.c_void_p(rewards.data_ptr()),
        ctypes.c_void_p(dones.data_ptr()),
        ctypes.c_void_p(importance.data_ptr()),
        ctypes.c_void_p(advantages.data_ptr()),
        ctypes.c_float(GAMMA),
        ctypes.c_float(GAE_LAMBDA),
        ctypes.c_float(RHO_CLIP),
        ctypes.c_float(C_CLIP),
        ctypes.c_int(num_steps),
        ctypes.c_int(horizon),
    )
    lib.sync_device()
    
    err = lib.get_last_error()
    if err:
        raise RuntimeError(f"CUDA error: {err.decode()}")
    
    return advantages


def benchmark_kernel(lib, kernel_fn, values, rewards, dones, importance, 
                     warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark a kernel with warmup and timing."""
    num_steps, horizon = values.shape
    advantages = torch.zeros_like(values)
    
    for _ in range(warmup):
        kernel_fn(
            ctypes.c_void_p(values.data_ptr()),
            ctypes.c_void_p(rewards.data_ptr()),
            ctypes.c_void_p(dones.data_ptr()),
            ctypes.c_void_p(importance.data_ptr()),
            ctypes.c_void_p(advantages.data_ptr()),
            ctypes.c_float(GAMMA),
            ctypes.c_float(GAE_LAMBDA),
            ctypes.c_float(RHO_CLIP),
            ctypes.c_float(C_CLIP),
            ctypes.c_int(num_steps),
            ctypes.c_int(horizon),
        )
    lib.sync_device()
    
    start = time.perf_counter()
    for _ in range(iters):
        kernel_fn(
            ctypes.c_void_p(values.data_ptr()),
            ctypes.c_void_p(rewards.data_ptr()),
            ctypes.c_void_p(dones.data_ptr()),
            ctypes.c_void_p(importance.data_ptr()),
            ctypes.c_void_p(advantages.data_ptr()),
            ctypes.c_float(GAMMA),
            ctypes.c_float(GAE_LAMBDA),
            ctypes.c_float(RHO_CLIP),
            ctypes.c_float(C_CLIP),
            ctypes.c_int(num_steps),
            ctypes.c_int(horizon),
        )
    lib.sync_device()
    end = time.perf_counter()
    
    return (end - start) / iters * 1000


def test_correctness(lib, num_steps: int, horizon: int):
    """Test that original and new produce matching outputs."""
    values, rewards, dones, importance = create_test_tensors(num_steps, horizon)
    
    adv_original = run_kernel(lib, lib.launch_original, values, rewards, dones, importance)
    adv_new = run_kernel(lib, lib.launch_new, values, rewards, dones, importance)
    
    if not torch.allclose(adv_original, adv_new, rtol=RTOL, atol=ATOL):
        max_diff = (adv_original - adv_new).abs().max().item()
        mean_diff = (adv_original - adv_new).abs().mean().item()
        return False, f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
    
    return True, "OK"


def run_benchmark(lib, num_steps: int, horizon: int, description: str):
    """Run benchmark for a specific tensor size."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"  Shape: [{num_steps}, {horizon}] = {num_steps * horizon:,} elements")
    print(f"{'='*60}")
    
    # Test correctness
    correct, msg = test_correctness(lib, num_steps, horizon)
    print(f"  Correctness: {msg}")
    
    if not correct:
        print("  SKIPPING BENCHMARK - outputs don't match!")
        return None, None
    
    # Create tensors
    values, rewards, dones, importance = create_test_tensors(num_steps, horizon)
    
    # Benchmark
    time_original = benchmark_kernel(lib, lib.launch_original, values, rewards, dones, importance)
    print(f"  Original:    {time_original:.4f} ms")
    
    time_new = benchmark_kernel(lib, lib.launch_new, values, rewards, dones, importance)
    print(f"  New:         {time_new:.4f} ms")
    
    speedup = time_original / time_new if time_new > 0 else float('inf')
    print(f"  Speedup:     {speedup:.2f}x")
    
    return time_original, time_new


def main():
    print("=" * 60)
    print("Standalone Puff Advantage Kernel Benchmark")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Load the library
    lib = load_cuda_lib()
    if lib is None:
        return
    
    print("Loaded libpuff_advantage.so successfully")
    print(f"\nBenchmark config:")
    print(f"  Warmup iterations: {WARMUP_ITERS}")
    print(f"  Benchmark iterations: {BENCHMARK_ITERS}")
    print(f"  gamma={GAMMA}, lambda={GAE_LAMBDA}, rho_clip={RHO_CLIP}, c_clip={C_CLIP}")
    
    # Run benchmarks
    results = []
    for num_steps, horizon, desc in TEST_CONFIGS:
        time_orig, time_new = run_benchmark(lib, num_steps, horizon, desc)
        if time_orig is not None:
            results.append((desc, num_steps, horizon, time_orig, time_new))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<30} {'Elements':>10} {'Original':>12} {'New':>12} {'Speedup':>10}")
    print("-" * 80)
    for desc, num_steps, horizon, time_orig, time_new in results:
        elements = num_steps * horizon
        speedup = time_orig / time_new if time_new > 0 else float('inf')
        print(f"{desc:<30} {elements:>10,} {time_orig:>10.4f}ms {time_new:>10.4f}ms {speedup:>9.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()


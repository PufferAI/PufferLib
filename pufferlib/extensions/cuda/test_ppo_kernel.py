#!/usr/bin/env python3
"""
Test suite for PPO loss kernel optimization.

Compile:
    cd pufferlib/extensions/cuda
    nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC kernels.cu -o kernels.so

Usage:
    python test_ppo_kernel.py
"""

import ctypes
import gc
import time
from pathlib import Path

import torch


def load_extension():
    """Load the precompiled CUDA .so via ctypes."""
    so_file = Path(__file__).parent / "kernels.so"
    
    if not so_file.exists():
        raise FileNotFoundError(
            f"Compiled library not found: {so_file}\n"
            f"Compile it first with:\n"
            f"  cd {so_file.parent}\n"
            f"  nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC kernels.cu -o kernels.so"
        )
    
    print(f"Loading {so_file}...")
    lib = ctypes.CDLL(str(so_file))
    
    # Forward original: launch_ppo_loss_forward_original_f32
    lib.launch_ppo_loss_forward_original_f32.argtypes = [
        ctypes.c_void_p,  # loss_output (float*)
        ctypes.c_void_p,  # saved_for_backward (double*)
        ctypes.c_void_p,  # logits (float*)
        ctypes.c_void_p,  # values_pred (float*)
        ctypes.c_void_p,  # actions (int64_t*)
        ctypes.c_void_p,  # old_logprobs (float*)
        ctypes.c_void_p,  # advantages (float*)
        ctypes.c_void_p,  # prio (float*)
        ctypes.c_void_p,  # values (float*)
        ctypes.c_void_p,  # returns (float*)
        ctypes.c_void_p,  # adv_mean (float*)
        ctypes.c_void_p,  # adv_std (float*)
        ctypes.c_double,  # clip_coef
        ctypes.c_double,  # vf_clip_coef
        ctypes.c_double,  # vf_coef
        ctypes.c_double,  # ent_coef
        ctypes.c_int,     # T_seq
        ctypes.c_int,     # A
        ctypes.c_int,     # N
    ]
    lib.launch_ppo_loss_forward_original_f32.restype = None
    
    # Forward optimized: launch_ppo_loss_forward_optimized_f32
    lib.launch_ppo_loss_forward_optimized_f32.argtypes = [
        ctypes.c_void_p,  # loss_output (float*)
        ctypes.c_void_p,  # saved_for_backward (double*)
        ctypes.c_void_p,  # logits (float*)
        ctypes.c_void_p,  # values_pred (float*)
        ctypes.c_void_p,  # actions (int64_t*)
        ctypes.c_void_p,  # old_logprobs (float*)
        ctypes.c_void_p,  # advantages (float*)
        ctypes.c_void_p,  # prio (float*)
        ctypes.c_void_p,  # values (float*)
        ctypes.c_void_p,  # returns (float*)
        ctypes.c_void_p,  # adv_mean (float*)
        ctypes.c_void_p,  # adv_std (float*)
        ctypes.c_double,  # clip_coef
        ctypes.c_double,  # vf_clip_coef
        ctypes.c_double,  # vf_coef
        ctypes.c_double,  # ent_coef
        ctypes.c_int,     # T_seq
        ctypes.c_int,     # A
        ctypes.c_int,     # N
    ]
    lib.launch_ppo_loss_forward_optimized_f32.restype = None
    
    # Backward original: launch_ppo_loss_backward_original_f32
    lib.launch_ppo_loss_backward_original_f32.argtypes = [
        ctypes.c_void_p,  # grad_logits (float*)
        ctypes.c_void_p,  # grad_values_pred (float*)
        ctypes.c_void_p,  # grad_loss (float*)
        ctypes.c_void_p,  # logits (float*)
        ctypes.c_void_p,  # actions (int64_t*)
        ctypes.c_void_p,  # old_logprobs (float*)
        ctypes.c_void_p,  # advantages (float*)
        ctypes.c_void_p,  # prio (float*)
        ctypes.c_void_p,  # values (float*)
        ctypes.c_void_p,  # returns (float*)
        ctypes.c_void_p,  # saved_for_backward (double*)
        ctypes.c_void_p,  # adv_mean (float*)
        ctypes.c_void_p,  # adv_std (float*)
        ctypes.c_double,  # clip_coef
        ctypes.c_double,  # vf_clip_coef
        ctypes.c_double,  # vf_coef
        ctypes.c_double,  # ent_coef
        ctypes.c_int,     # T_seq
        ctypes.c_int,     # A
        ctypes.c_int,     # N
    ]
    lib.launch_ppo_loss_backward_original_f32.restype = None
    
    # Backward optimized: launch_ppo_loss_backward_optimized_f32
    lib.launch_ppo_loss_backward_optimized_f32.argtypes = [
        ctypes.c_void_p,  # grad_logits (float*)
        ctypes.c_void_p,  # grad_values_pred (float*)
        ctypes.c_void_p,  # grad_loss (float*)
        ctypes.c_void_p,  # logits (float*)
        ctypes.c_void_p,  # actions (int64_t*)
        ctypes.c_void_p,  # old_logprobs (float*)
        ctypes.c_void_p,  # advantages (float*)
        ctypes.c_void_p,  # prio (float*)
        ctypes.c_void_p,  # values (float*)
        ctypes.c_void_p,  # returns (float*)
        ctypes.c_void_p,  # saved_for_backward (double*)
        ctypes.c_void_p,  # adv_mean (float*)
        ctypes.c_void_p,  # adv_std (float*)
        ctypes.c_double,  # clip_coef
        ctypes.c_double,  # vf_clip_coef
        ctypes.c_double,  # vf_coef
        ctypes.c_double,  # ent_coef
        ctypes.c_int,     # T_seq
        ctypes.c_int,     # A
        ctypes.c_int,     # N
    ]
    lib.launch_ppo_loss_backward_optimized_f32.restype = None
    
    lib.sync_device.argtypes = []
    lib.sync_device.restype = None
    
    lib.get_last_error.argtypes = []
    lib.get_last_error.restype = ctypes.c_char_p
    
    print("Loaded successfully!\n")
    return lib


# Test configurations based on production RL workloads
# Format: (N, T, A) where:
#   N = batch size (number of sequences)
#   T = sequence length (timesteps per sequence)
#   A = action space size
TEST_CONFIGS = {
    "tiny": [
        (32, 64, 4),     # simple control
        (32, 64, 6),     # lunar lander style
        (64, 32, 4),
    ],
    "small": [
        (128, 64, 6),    # atari minimal
        (128, 64, 18),   # atari full
        (256, 32, 15),   # procgen
    ],
    "medium": [
        (512, 64, 4),
        (512, 64, 18),
        (1024, 32, 18),
    ],
    "large": [
        (1024, 64, 18),
        (2048, 32, 18),
        (2048, 64, 6),
    ],
    "extreme": [
        (512, 64, 64),   # large action space
        (256, 64, 128),  # very large (nethack-ish)
    ],
}


def create_test_tensors(N: int, T: int, A: int, device: str = "cuda"):
    """
    Create test tensors matching actual PPO kernel inputs.
    
    Args:
        N: Batch size (number of sequences)
        T: Sequence length (timesteps per sequence)
        A: Action space size
        device: Device to create tensors on
    
    Returns:
        dict of tensors
    """
    # logits: (N, T, A) - raw policy outputs
    logits = torch.randn(N, T, A, device=device, dtype=torch.float32)
    
    # values_pred: (N, T) - predicted values from critic
    values_pred = torch.randn(N, T, device=device, dtype=torch.float32)
    
    # actions: (N, T) - actions taken (indices into A)
    actions = torch.randint(0, A, (N, T), device=device, dtype=torch.int64)
    
    # old_logprobs: (N, T) - log probs from behavior policy
    old_logprobs = torch.randn(N, T, device=device, dtype=torch.float32) - 2.0  # make negative
    
    # advantages: (N, T) - GAE advantages
    advantages = torch.randn(N, T, device=device, dtype=torch.float32)
    
    # prio: (N,) - importance weights per sequence
    prio = torch.ones(N, device=device, dtype=torch.float32)
    
    # values: (N, T) - old value predictions (for clipping)
    values = torch.randn(N, T, device=device, dtype=torch.float32)
    
    # returns: (N, T) - computed returns
    returns = torch.randn(N, T, device=device, dtype=torch.float32)
    
    # adv_mean, adv_std: scalars for normalization
    adv_mean = torch.tensor([advantages.mean().item()], device=device, dtype=torch.float32)
    adv_std = torch.tensor([advantages.std().item()], device=device, dtype=torch.float32)
    
    return {
        "logits": logits,
        "values_pred": values_pred,
        "actions": actions,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "prio": prio,
        "values": values,
        "returns": returns,
        "adv_mean": adv_mean,
        "adv_std": adv_std,
    }


def compare_outputs(
    outputs_orig: list[torch.Tensor],
    outputs_new: list[torch.Tensor],
    names: list[str],
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> tuple[bool, dict]:
    """
    Compare two sets of outputs for numerical equivalence.
    
    Returns:
        (all_passed, details_dict)
    """
    assert len(outputs_orig) == len(outputs_new) == len(names)
    
    results = {}
    all_passed = True
    
    for orig, new, name in zip(outputs_orig, outputs_new, names):
        # Check shapes match
        if orig.shape != new.shape:
            results[name] = {
                "passed": False,
                "error": f"Shape mismatch: {orig.shape} vs {new.shape}",
            }
            all_passed = False
            continue
        
        # Check values
        try:
            torch.testing.assert_close(new, orig, rtol=rtol, atol=atol)
            max_diff = (orig - new).abs().max().item()
            mean_diff = (orig - new).abs().mean().item()
            results[name] = {
                "passed": True,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
            }
        except AssertionError as e:
            max_diff = (orig - new).abs().max().item()
            mean_diff = (orig - new).abs().mean().item()
            
            results[name] = {
                "passed": False,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "error": str(e)[:200],
            }
            all_passed = False
    
    return all_passed, results


def cleanup_gpu():
    """Force cleanup of GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


def run_forward_kernel(lib, kernel_func, tensors, N, T, A,
                       clip_coef=0.2, vf_clip_coef=0.2, vf_coef=0.5, ent_coef=0.01):
    """
    Run a PPO forward kernel and return output tensors.
    
    Returns:
        (loss, saved_for_backward)
    """
    device = tensors["logits"].device
    
    # Allocate output tensors
    loss = torch.zeros(1, device=device, dtype=torch.float32)
    saved_for_backward = torch.empty(N * T, 5, device=device, dtype=torch.float64)
    
    # Call kernel
    kernel_func(
        loss.data_ptr(),
        saved_for_backward.data_ptr(),
        tensors["logits"].data_ptr(),
        tensors["values_pred"].data_ptr(),
        tensors["actions"].data_ptr(),
        tensors["old_logprobs"].data_ptr(),
        tensors["advantages"].data_ptr(),
        tensors["prio"].data_ptr(),
        tensors["values"].data_ptr(),
        tensors["returns"].data_ptr(),
        tensors["adv_mean"].data_ptr(),
        tensors["adv_std"].data_ptr(),
        clip_coef,
        vf_clip_coef,
        vf_coef,
        ent_coef,
        T, A, N
    )
    
    # Sync and check for errors
    lib.sync_device()
    err = lib.get_last_error()
    if err and err != b"no error":
        raise RuntimeError(f"CUDA error: {err.decode()}")
    
    return loss, saved_for_backward


def run_backward_kernel(lib, kernel_func, tensors, saved_for_backward, N, T, A,
                        clip_coef=0.2, vf_clip_coef=0.2, vf_coef=0.5, ent_coef=0.01):
    """
    Run a PPO backward kernel and return gradient tensors.
    
    Returns:
        (grad_logits, grad_values_pred)
    """
    device = tensors["logits"].device
    
    # Allocate output tensors
    grad_logits = torch.empty(N, T, A, device=device, dtype=torch.float32)
    grad_values_pred = torch.empty(N, T, device=device, dtype=torch.float32)
    
    # grad_loss is typically 1.0 for .backward()
    grad_loss = torch.ones(1, device=device, dtype=torch.float32)
    
    # Call kernel
    kernel_func(
        grad_logits.data_ptr(),
        grad_values_pred.data_ptr(),
        grad_loss.data_ptr(),
        tensors["logits"].data_ptr(),
        tensors["actions"].data_ptr(),
        tensors["old_logprobs"].data_ptr(),
        tensors["advantages"].data_ptr(),
        tensors["prio"].data_ptr(),
        tensors["values"].data_ptr(),
        tensors["returns"].data_ptr(),
        saved_for_backward.data_ptr(),
        tensors["adv_mean"].data_ptr(),
        tensors["adv_std"].data_ptr(),
        clip_coef,
        vf_clip_coef,
        vf_coef,
        ent_coef,
        T, A, N
    )
    
    # Sync and check for errors
    lib.sync_device()
    err = lib.get_last_error()
    if err and err != b"no error":
        raise RuntimeError(f"CUDA error: {err.decode()}")
    
    return grad_logits, grad_values_pred


def run_forward_correctness_test(
    lib,
    N: int,
    T: int,
    A: int,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """
    Run correctness test for forward pass.
    
    Returns:
        (passed, details)
    """
    try:
        tensors = create_test_tensors(N, T, A)
        
        # Run original
        loss_orig, saved_orig = run_forward_kernel(
            lib, lib.launch_ppo_loss_forward_original_f32, tensors, N, T, A
        )
        
        # Run optimized
        loss_opt, saved_opt = run_forward_kernel(
            lib, lib.launch_ppo_loss_forward_optimized_f32, tensors, N, T, A
        )
        
        # Compare outputs - only compare loss (optimized kernel doesn't use saved_for_backward)
        output_names = ["loss"]
        passed, details = compare_outputs(
            [loss_orig],
            [loss_opt],
            output_names,
            rtol=1e-3,  # looser tolerance for fast math (__expf, __logf)
            atol=1e-3,
        )
        
        return passed, details
    finally:
        cleanup_gpu()


def run_backward_correctness_test(
    lib,
    N: int,
    T: int,
    A: int,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """
    Run correctness test for backward pass.
    
    Returns:
        (passed, details)
    """
    try:
        tensors = create_test_tensors(N, T, A)
        
        # Run forward to get saved_for_backward (use original for consistency)
        loss, saved_for_backward = run_forward_kernel(
            lib, lib.launch_ppo_loss_forward_original_f32, tensors, N, T, A
        )
        
        # Run original backward
        grad_logits_orig, grad_values_pred_orig = run_backward_kernel(
            lib, lib.launch_ppo_loss_backward_original_f32,
            tensors, saved_for_backward, N, T, A
        )
        
        # Run optimized backward
        grad_logits_opt, grad_values_pred_opt = run_backward_kernel(
            lib, lib.launch_ppo_loss_backward_optimized_f32,
            tensors, saved_for_backward, N, T, A
        )
        
        # Compare gradients
        output_names = ["grad_logits", "grad_values_pred"]
        passed, details = compare_outputs(
            [grad_logits_orig, grad_values_pred_orig],
            [grad_logits_opt, grad_values_pred_opt],
            output_names
        )
        
        return passed, details
    finally:
        cleanup_gpu()


def run_forward_benchmark(
    lib,
    N: int,
    T: int,
    A: int,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> dict:
    """
    Benchmark both forward kernels and return timing results.
    """
    try:
        tensors = create_test_tensors(N, T, A)
        
        # Warmup
        for _ in range(warmup_iters):
            _ = run_forward_kernel(lib, lib.launch_ppo_loss_forward_original_f32, tensors, N, T, A)
            _ = run_forward_kernel(lib, lib.launch_ppo_loss_forward_optimized_f32, tensors, N, T, A)
        
        lib.sync_device()
        
        # Benchmark original
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_forward_kernel(lib, lib.launch_ppo_loss_forward_original_f32, tensors, N, T, A)
        lib.sync_device()
        orig_time = (time.perf_counter() - start) / bench_iters * 1000  # ms
        
        # Benchmark optimized
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_forward_kernel(lib, lib.launch_ppo_loss_forward_optimized_f32, tensors, N, T, A)
        lib.sync_device()
        opt_time = (time.perf_counter() - start) / bench_iters * 1000  # ms
        
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        
        return {
            "original_ms": orig_time,
            "optimized_ms": opt_time,
            "speedup": speedup,
        }
    finally:
        cleanup_gpu()


def run_backward_benchmark(
    lib,
    N: int,
    T: int,
    A: int,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> dict:
    """
    Benchmark both backward kernels and return timing results.
    """
    try:
        tensors = create_test_tensors(N, T, A)
        
        # Run forward to get saved_for_backward
        loss, saved_for_backward = run_forward_kernel(
            lib, lib.launch_ppo_loss_forward_original_f32, tensors, N, T, A
        )
        
        # Warmup
        for _ in range(warmup_iters):
            _ = run_backward_kernel(
                lib, lib.launch_ppo_loss_backward_original_f32,
                tensors, saved_for_backward, N, T, A
            )
            _ = run_backward_kernel(
                lib, lib.launch_ppo_loss_backward_optimized_f32,
                tensors, saved_for_backward, N, T, A
            )
        
        lib.sync_device()
        
        # Benchmark original backward
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_backward_kernel(
                lib, lib.launch_ppo_loss_backward_original_f32,
                tensors, saved_for_backward, N, T, A
            )
        lib.sync_device()
        orig_time = (time.perf_counter() - start) / bench_iters * 1000
        
        # Benchmark optimized backward
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_backward_kernel(
                lib, lib.launch_ppo_loss_backward_optimized_f32,
                tensors, saved_for_backward, N, T, A
            )
        lib.sync_device()
        opt_time = (time.perf_counter() - start) / bench_iters * 1000
        
        return {
            "original_ms": orig_time,
            "optimized_ms": opt_time,
            "speedup": orig_time / opt_time if opt_time > 0 else float('inf'),
        }
    finally:
        cleanup_gpu()


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    try:
        lib = load_extension()
    except Exception as e:
        print(f"ERROR: Failed to load extension: {e}")
        return 1
    
    configs = []
    for size_name, size_configs in TEST_CONFIGS.items():
        for cfg in size_configs:
            configs.append((size_name, cfg))
    
    print("=" * 70)
    print("FORWARD CORRECTNESS TESTS")
    print("=" * 70)
    
    all_passed = True
    for size_name, (N, T, A) in configs:
        config_str = f"N={N:4d}, T={T:3d}, A={A:3d}"
        
        try:
            passed, details = run_forward_correctness_test(lib, N, T, A)
        except Exception as e:
            print(f"[{size_name:12s}] {config_str}  EXCEPTION: {e}")
            all_passed = False
            continue
        
        if passed:
            status = "PASS"
        else:
            status = "FAIL"
            failed = [name for name, d in details.items() if not d["passed"]]
            status += f"  (failed: {', '.join(failed)})"
            max_diffs = [f"{name}:{d['max_diff']:.2e}" for name, d in details.items()]
            status += f"\n             max_diffs: {', '.join(max_diffs)}"
            all_passed = False
        
        print(f"[{size_name:12s}] {config_str}  {status}")
    
    print()
    
    print("=" * 70)
    print("FORWARD BENCHMARKS")
    print("=" * 70)
    print(f"{'Config':<30s} {'Original':>12s} {'Optimized':>12s} {'Speedup':>10s}")
    print("-" * 70)
    
    for size_name, (N, T, A) in configs:
        config_str = f"N={N}, T={T}, A={A}"
        
        try:
            bench = run_forward_benchmark(lib, N, T, A)
            print(
                f"{config_str:<30s} "
                f"{bench['original_ms']:>10.3f}ms "
                f"{bench['optimized_ms']:>10.3f}ms "
                f"{bench['speedup']:>9.2f}x"
            )
        except Exception as e:
            print(f"{config_str:<30s} ERROR: {e}")
    
    print()
    
    print("=" * 70)
    print("BACKWARD CORRECTNESS TESTS")
    print("=" * 70)
    
    for size_name, (N, T, A) in configs:
        config_str = f"N={N:4d}, T={T:3d}, A={A:3d}"
        
        try:
            passed, details = run_backward_correctness_test(lib, N, T, A)
        except Exception as e:
            print(f"[{size_name:12s}] {config_str}  EXCEPTION: {e}")
            all_passed = False
            continue
        
        if passed:
            status = "PASS"
        else:
            status = "FAIL"
            failed = [name for name, d in details.items() if not d["passed"]]
            status += f"  (failed: {', '.join(failed)})"
            max_diffs = [f"{name}:{d['max_diff']:.2e}" for name, d in details.items()]
            status += f"\n             max_diffs: {', '.join(max_diffs)}"
            all_passed = False
        
        print(f"[{size_name:12s}] {config_str}  {status}")
    
    print()
    
    print("=" * 70)
    print("BACKWARD BENCHMARKS")
    print("=" * 70)
    print(f"{'Config':<30s} {'Original':>12s} {'Optimized':>12s} {'Speedup':>10s}")
    print("-" * 70)
    
    for size_name, (N, T, A) in configs:
        config_str = f"N={N}, T={T}, A={A}"
        
        try:
            bench = run_backward_benchmark(lib, N, T, A)
            print(
                f"{config_str:<30s} "
                f"{bench['original_ms']:>10.3f}ms "
                f"{bench['optimized_ms']:>10.3f}ms "
                f"{bench['speedup']:>9.2f}x"
            )
        except Exception as e:
            print(f"{config_str:<30s} ERROR: {e}")
    
    print()
    
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())

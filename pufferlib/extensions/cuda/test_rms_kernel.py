#!/usr/bin/env python3
"""
Test suite for RMSNorm kernel optimization.

Compile:
    cd pufferlib/extensions/cuda
    nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC kernels.cu -o kernels.so

Usage:
    python test_rms_kernel.py
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
    
    # Define function signatures
    # Forward original: launch_rmsnorm_forward_original_f32
    lib.launch_rmsnorm_forward_original_f32.argtypes = [
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # inv_norm_buf
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # weight
        ctypes.c_double,  # eps
        ctypes.c_int,     # T_total
        ctypes.c_int,     # H
        ctypes.c_int,     # B
    ]
    lib.launch_rmsnorm_forward_original_f32.restype = None
    
    # Forward optimized: launch_rmsnorm_forward_optimized_f32
    lib.launch_rmsnorm_forward_optimized_f32.argtypes = [
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # inv_norm_buf
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # weight
        ctypes.c_double,  # eps
        ctypes.c_int,     # T_total
        ctypes.c_int,     # H
        ctypes.c_int,     # B
    ]
    lib.launch_rmsnorm_forward_optimized_f32.restype = None
    
    # Backward original: launch_rmsnorm_backward_original_f32
    lib.launch_rmsnorm_backward_original_f32.argtypes = [
        ctypes.c_void_p,  # grad_x
        ctypes.c_void_p,  # grad_weight
        ctypes.c_void_p,  # grad_out
        ctypes.c_void_p,  # inv_norm_buf
        ctypes.c_void_p,  # x_buf
        ctypes.c_void_p,  # weight
        ctypes.c_double,  # eps
        ctypes.c_int,     # T_total
        ctypes.c_int,     # H
        ctypes.c_int,     # B
    ]
    lib.launch_rmsnorm_backward_original_f32.restype = None
    
    # Backward optimized: launch_rmsnorm_backward_optimized_f32
    lib.launch_rmsnorm_backward_optimized_f32.argtypes = [
        ctypes.c_void_p,  # grad_x
        ctypes.c_void_p,  # grad_weight
        ctypes.c_void_p,  # grad_out
        ctypes.c_void_p,  # inv_norm_buf
        ctypes.c_void_p,  # x_buf
        ctypes.c_void_p,  # weight
        ctypes.c_double,  # eps
        ctypes.c_int,     # T_total
        ctypes.c_int,     # H
        ctypes.c_int,     # B
    ]
    lib.launch_rmsnorm_backward_optimized_f32.restype = None
    
    lib.sync_device.argtypes = []
    lib.sync_device.restype = None
    
    lib.get_last_error.argtypes = []
    lib.get_last_error.restype = ctypes.c_char_p
    
    print("Loaded successfully!\n")
    return lib


# Test configurations based on production RL workloads
# Format: (B, T, H) where x is (B, T, H) and weight is (H,)
# H must be <= WARP_SIZE * RMS_THREADS = 32 * 128 = 4096
TEST_CONFIGS = {
    "tiny": [
        (512, 64, 128),
        (512, 64, 256),
        (512, 64, 384),
    ],
    "small": [
        (512, 96, 256),
        (768, 64, 256),
        (768, 64, 512),
    ],
    "medium": [
        (1024, 64, 256),
        (1024, 64, 512),
        (1024, 96, 384),
        (1024, 128, 256),
    ],
    "large": [
        (1536, 64, 512),
        (2048, 64, 256),
        (2048, 64, 512),
        (2048, 96, 512),
    ],
    "edge_cases": [
        (512, 64, 128),
        (512, 128, 512),
        (2048, 64, 256),
        (1024, 91, 384),
        (1536, 77, 512),
    ],
}


def create_test_tensors(B: int, T: int, H: int, device: str = "cuda"):
    """
    Create test tensors matching actual kernel inputs.
    
    Args:
        B: Batch size
        T: Sequence length (T_total)
        H: Hidden size
        device: Device to create tensors on
    
    Returns:
        x: (B, T, H) tensor
        weight: (H,) tensor
    """
    x = torch.randn(B, T, H, device=device, dtype=torch.float32)
    weight = torch.randn(H, device=device, dtype=torch.float32)
    
    return x, weight


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


def run_forward_kernel(lib, kernel_func, x, weight, B, T, H, eps=1e-6):
    """
    Run a forward kernel and return output tensors.
    
    Args:
        lib: ctypes library
        kernel_func: the kernel launch function to call
        x: input tensor (B, T, H)
        weight: weight tensor (H,)
        B, T, H: dimensions
        eps: epsilon for numerical stability
    
    Returns:
        (out, inv_norm_buf)
    """
    device = x.device
    
    # Allocate output tensors
    out = torch.empty(B, T, H, device=device, dtype=torch.float32)
    inv_norm_buf = torch.empty(B * T, device=device, dtype=torch.float32)
    
    # Call kernel
    kernel_func(
        out.data_ptr(),
        inv_norm_buf.data_ptr(),
        x.data_ptr(),
        weight.data_ptr(),
        eps,
        T, H, B
    )
    
    # Sync and check for errors
    lib.sync_device()
    err = lib.get_last_error()
    if err and err != b"no error":
        raise RuntimeError(f"CUDA error: {err.decode()}")
    
    return out, inv_norm_buf


def run_backward_kernel(lib, kernel_func, grad_out, inv_norm_buf, x, weight, B, T, H, eps=1e-6):
    """
    Run a backward kernel and return gradient tensors.
    
    Args:
        lib: ctypes library
        kernel_func: the kernel launch function to call
        grad_out: gradient w.r.t. output (B, T, H)
        inv_norm_buf: cached inverse RMS from forward (B * T,)
        x: original input tensor (B, T, H)
        weight: weight tensor (H,)
        B, T, H: dimensions
        eps: epsilon for numerical stability
    
    Returns:
        (grad_x, grad_weight)
    """
    device = x.device
    
    # Allocate output tensors
    grad_x = torch.empty(B, T, H, device=device, dtype=torch.float32)
    grad_weight = torch.empty(B, T, H, device=device, dtype=torch.float32)
    
    # Call kernel
    kernel_func(
        grad_x.data_ptr(),
        grad_weight.data_ptr(),
        grad_out.data_ptr(),
        inv_norm_buf.data_ptr(),
        x.data_ptr(),
        weight.data_ptr(),
        eps,
        T, H, B
    )
    
    # Sync and check for errors
    lib.sync_device()
    err = lib.get_last_error()
    if err and err != b"no error":
        raise RuntimeError(f"CUDA error: {err.decode()}")
    
    return grad_x, grad_weight


def run_forward_correctness_test(
    lib,
    B: int,
    T: int,
    H: int,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """
    Run correctness test for forward pass.
    
    Returns:
        (passed, details)
    """
    try:
        x, weight = create_test_tensors(B, T, H)
        
        # Run original
        out_orig, inv_norm_orig = run_forward_kernel(
            lib, lib.launch_rmsnorm_forward_original_f32, x, weight, B, T, H
        )
        
        # Run optimized
        out_opt, inv_norm_opt = run_forward_kernel(
            lib, lib.launch_rmsnorm_forward_optimized_f32, x, weight, B, T, H
        )
        
        # Compare outputs
        output_names = ["out", "inv_norm_buf"]
        passed, details = compare_outputs(
            [out_orig, inv_norm_orig],
            [out_opt, inv_norm_opt],
            output_names
        )
        
        return passed, details
    finally:
        cleanup_gpu()


def run_backward_correctness_test(
    lib,
    B: int,
    T: int,
    H: int,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """
    Run correctness test for backward pass.
    
    Returns:
        (passed, details)
    """
    try:
        x, weight = create_test_tensors(B, T, H)
        
        # Run forward to get inv_norm_buf (use original for consistency)
        out, inv_norm_buf = run_forward_kernel(
            lib, lib.launch_rmsnorm_forward_original_f32, x, weight, B, T, H
        )
        
        # Create gradient input
        grad_out = torch.randn_like(out)
        
        # Run original backward
        grad_x_orig, grad_weight_orig = run_backward_kernel(
            lib, lib.launch_rmsnorm_backward_original_f32,
            grad_out, inv_norm_buf, x, weight, B, T, H
        )
        
        # Run optimized backward
        grad_x_opt, grad_weight_opt = run_backward_kernel(
            lib, lib.launch_rmsnorm_backward_optimized_f32,
            grad_out, inv_norm_buf, x, weight, B, T, H
        )
        
        # Compare gradients
        output_names = ["grad_x", "grad_weight"]
        passed, details = compare_outputs(
            [grad_x_orig, grad_weight_orig],
            [grad_x_opt, grad_weight_opt],
            output_names
        )
        
        return passed, details
    finally:
        cleanup_gpu()


def run_forward_benchmark(
    lib,
    B: int,
    T: int,
    H: int,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> dict:
    """
    Benchmark both forward kernels and return timing results.
    """
    try:
        x, weight = create_test_tensors(B, T, H)
        
        # Warmup
        for _ in range(warmup_iters):
            _ = run_forward_kernel(lib, lib.launch_rmsnorm_forward_original_f32, x, weight, B, T, H)
            _ = run_forward_kernel(lib, lib.launch_rmsnorm_forward_optimized_f32, x, weight, B, T, H)
        
        lib.sync_device()
        
        # Benchmark original
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_forward_kernel(lib, lib.launch_rmsnorm_forward_original_f32, x, weight, B, T, H)
        lib.sync_device()
        orig_time = (time.perf_counter() - start) / bench_iters * 1000  # ms
        
        # Benchmark optimized
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_forward_kernel(lib, lib.launch_rmsnorm_forward_optimized_f32, x, weight, B, T, H)
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
    B: int,
    T: int,
    H: int,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> dict:
    """
    Benchmark both backward kernels and return timing results.
    """
    try:
        x, weight = create_test_tensors(B, T, H)
        
        # Run forward to get inv_norm_buf
        out, inv_norm_buf = run_forward_kernel(
            lib, lib.launch_rmsnorm_forward_original_f32, x, weight, B, T, H
        )
        
        grad_out = torch.randn_like(out)
        
        # Warmup
        for _ in range(warmup_iters):
            _ = run_backward_kernel(
                lib, lib.launch_rmsnorm_backward_original_f32,
                grad_out, inv_norm_buf, x, weight, B, T, H
            )
            _ = run_backward_kernel(
                lib, lib.launch_rmsnorm_backward_optimized_f32,
                grad_out, inv_norm_buf, x, weight, B, T, H
            )
        
        lib.sync_device()
        
        # Benchmark original backward
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_backward_kernel(
                lib, lib.launch_rmsnorm_backward_original_f32,
                grad_out, inv_norm_buf, x, weight, B, T, H
            )
        lib.sync_device()
        orig_time = (time.perf_counter() - start) / bench_iters * 1000
        
        # Benchmark optimized backward
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_backward_kernel(
                lib, lib.launch_rmsnorm_backward_optimized_f32,
                grad_out, inv_norm_buf, x, weight, B, T, H
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
    for size_name, (B, T, H) in configs:
        config_str = f"B={B:4d}, T={T:3d}, H={H:3d}"
        
        try:
            passed, details = run_forward_correctness_test(lib, B, T, H)
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
    
    for size_name, (B, T, H) in configs:
        config_str = f"B={B}, T={T}, H={H}"
        
        try:
            bench = run_forward_benchmark(lib, B, T, H)
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
    
    for size_name, (B, T, H) in configs:
        config_str = f"B={B:4d}, T={T:3d}, H={H:3d}"
        
        try:
            passed, details = run_backward_correctness_test(lib, B, T, H)
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
    
    for size_name, (B, T, H) in configs:
        config_str = f"B={B}, T={T}, H={H}"
        
        try:
            bench = run_backward_benchmark(lib, B, T, H)
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

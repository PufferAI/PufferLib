#!/usr/bin/env python3
"""
Test suite for fused scan kernel optimization.

Compile:
    cd pufferlib/extensions/cuda
    nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC kernels.cu -o kernels.so

Usage:
    python test_fused_scan.py
"""

import argparse
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
    
    # Define function signatures for f32 wrappers
    # Forward original: launch_fused_scan_forward_original_f32
    lib.launch_fused_scan_forward_original_f32.argtypes = [
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # next_state
        ctypes.c_void_p,  # a_star
        ctypes.c_void_p,  # s_vals
        ctypes.c_void_p,  # log_values_buf
        ctypes.c_void_p,  # combined
        ctypes.c_void_p,  # state
        ctypes.c_int,     # T_seq
        ctypes.c_int,     # H
        ctypes.c_int,     # B
    ]
    lib.launch_fused_scan_forward_original_f32.restype = None
    
    # Forward checkpointed: launch_fused_scan_forward_checkpointed_f32
    lib.launch_fused_scan_forward_checkpointed_f32.argtypes = [
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # next_state
        ctypes.c_void_p,  # a_star
        ctypes.c_void_p,  # s_vals
        ctypes.c_void_p,  # log_values_buf
        ctypes.c_void_p,  # combined
        ctypes.c_void_p,  # state
        ctypes.c_int,     # T_seq
        ctypes.c_int,     # H
        ctypes.c_int,     # B
    ]
    lib.launch_fused_scan_forward_checkpointed_f32.restype = None
    
    # Backward original: launch_fused_scan_backward_original_f32
    lib.launch_fused_scan_backward_original_f32.argtypes = [
        ctypes.c_void_p,  # grad_combined
        ctypes.c_void_p,  # grad_state
        ctypes.c_void_p,  # grad_out
        ctypes.c_void_p,  # grad_next_state
        ctypes.c_void_p,  # combined
        ctypes.c_void_p,  # state
        ctypes.c_void_p,  # a_star_buf
        ctypes.c_void_p,  # s_buf
        ctypes.c_void_p,  # log_values_buf
        ctypes.c_int,     # T_seq
        ctypes.c_int,     # H
        ctypes.c_int,     # B
    ]
    lib.launch_fused_scan_backward_original_f32.restype = None
    
    # Backward checkpointed: launch_fused_scan_backward_checkpointed_f32
    lib.launch_fused_scan_backward_checkpointed_f32.argtypes = [
        ctypes.c_void_p,  # grad_combined
        ctypes.c_void_p,  # grad_state
        ctypes.c_void_p,  # grad_out
        ctypes.c_void_p,  # grad_next_state
        ctypes.c_void_p,  # combined
        ctypes.c_void_p,  # state
        ctypes.c_void_p,  # a_star_buf (sparse checkpoints from forward)
        ctypes.c_void_p,  # s_buf (sparse checkpoints from forward)
        ctypes.c_void_p,  # log_values_buf (sparse checkpoints from forward)
        ctypes.c_int,     # T_seq
        ctypes.c_int,     # H
        ctypes.c_int,     # B
    ]
    lib.launch_fused_scan_backward_checkpointed_f32.restype = None
    
    lib.sync_device.argtypes = []
    lib.sync_device.restype = None
    
    lib.get_last_error.argtypes = []
    lib.get_last_error.restype = ctypes.c_char_p
    
    print("Loaded successfully!\n")
    return lib


# Test configurations based on production RL workloads
# Format: (B, T, H) where combined is (B, T, 3*H) and state is (B, 1, H)
# Optimized for typical production ranges: B=512-2048, T=64-128, H=256-512
TEST_CONFIGS = {
    "tiny": [
        # Small baseline configs
        (512, 64, 256),   # Minimum production config
        (512, 64, 384),   # Mid hidden
        (512, 64, 512),   # Standard hidden
    ],
    "small": [
        # Standard production configs
        (512, 96, 256),   # Longer sequence
        (768, 64, 256),   # More batches
        (768, 64, 512),   # Standard large
    ],
    "medium": [
        # Typical RL training configs
        (1024, 64, 256),  # High parallelism
        (1024, 64, 512),  # High parallelism + large hidden
        (1024, 96, 384),  # Longer sequence
        (1024, 128, 256), # Max sequence length
    ],
    "large": [
        # Large-scale production configs
        (1536, 64, 512),  # Very high parallelism
        (2048, 64, 256),  # Maximum batches, standard hidden
        (2048, 64, 512),  # Maximum batches, large hidden
        (2048, 96, 512),  # Maximum batches, long sequence
    ],
    "edge_cases": [
        # Edge cases within production ranges
        (512, 64, 256),   # Minimum viable
        (512, 128, 512),  # Max sequence, min batch
        (2048, 64, 256),  # Max batch, standard hidden
        (1024, 91, 384),  # Odd sequence length
        (1536, 77, 512),  # Non-standard dimensions
    ],
}


def create_test_tensors(B: int, T: int, H: int, device: str = "cuda"):
    """
    Create test tensors matching actual kernel inputs.
    
    Args:
        B: Batch size (number of segments)
        T: Sequence length (bptt_horizon)
        H: Hidden size
        device: Device to create tensors on
    
    Returns:
        combined: (B, T, 3*H) tensor with [hidden, gate, proj] layout
        state: (B, 1, H) tensor with initial state (must be positive for log)
    """
    # combined contains [hidden, gate, proj] concatenated on last dim
    # Values should be in reasonable range for the operations
    combined = torch.randn(B, T, 3 * H, device=device, dtype=torch.float32)
    
    # state must be positive since we take log(state) in the kernel
    # Use softplus or abs to ensure positivity
    state = torch.abs(torch.randn(B, 1, H, device=device, dtype=torch.float32)) + 0.1
    
    return combined, state


def compare_outputs(
    outputs_orig: list[torch.Tensor],
    outputs_new: list[torch.Tensor],
    names: list[str],
    rtol: float = 1e-4,
    atol: float = 1e-4,  # Increased to handle fast math intrinsic differences
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
            
            # Find where the max difference occurs
            diff = (orig - new).abs()
            max_idx = diff.argmax().item()
            max_idx_tuple = tuple(
                (max_idx // diff[..., 0].numel()) if i == 0 
                else (max_idx % diff.shape[-1]) 
                for i in range(diff.dim())
            )
            
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


def run_kernel(lib, kernel_func, combined, state, B, T, H):
    """
    Run a kernel and return output tensors.
    
    Args:
        lib: ctypes library
        kernel_func: the kernel launch function to call
        combined: input tensor (B, T, 3*H)
        state: input tensor (B, 1, H)
        B, T, H: dimensions
    
    Returns:
        (out, next_state, a_star, s_vals, log_values_buf)
    """
    T_buf = T + 1
    device = combined.device
    
    # Allocate output tensors
    out = torch.empty(B, T, H, device=device, dtype=torch.float32)
    next_state_out = torch.empty(B, 1, H, device=device, dtype=torch.float32)
    a_star = torch.empty(B, T_buf, H, device=device, dtype=torch.float32)
    s_vals = torch.empty(B, T_buf, H, device=device, dtype=torch.float32)
    log_values_buf = torch.empty(B, T_buf, H, device=device, dtype=torch.float32)
    
    # Call kernel
    kernel_func(
        out.data_ptr(),
        next_state_out.data_ptr(),
        a_star.data_ptr(),
        s_vals.data_ptr(),
        log_values_buf.data_ptr(),
        combined.data_ptr(),
        state.data_ptr(),
        T, H, B
    )
    
    # Sync and check for errors
    lib.sync_device()
    err = lib.get_last_error()
    if err and err != b"no error":
        raise RuntimeError(f"CUDA error: {err.decode()}")
    
    return out, next_state_out, a_star, s_vals, log_values_buf


def run_correctness_test(
    lib,
    B: int,
    T: int,
    H: int,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """
    Run correctness test for a single configuration.
    
    Returns:
        (passed, details)
    """
    try:
        combined, state = create_test_tensors(B, T, H)
        
        # Run original
        outputs_orig = run_kernel(lib, lib.launch_fused_scan_forward_original_f32, combined, state, B, T, H)
        
        # Run optimized (checkpointed)
        outputs_new = run_kernel(lib, lib.launch_fused_scan_forward_checkpointed_f32, combined, state, B, T, H)
        
        # Compare only out and next_state (intermediate buffers may differ due to sparse checkpointing)
        output_names = ["out", "next_state"]
        passed, details = compare_outputs(outputs_orig[:2], outputs_new[:2], output_names)
        
        return passed, details
    finally:
        cleanup_gpu()


def run_benchmark(
    lib,
    B: int,
    T: int,
    H: int,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> dict:
    """
    Benchmark both kernels and return timing results.
    """
    try:
        combined, state = create_test_tensors(B, T, H)
        
        # Warmup
        for _ in range(warmup_iters):
            _ = run_kernel(lib, lib.launch_fused_scan_forward_original_f32, combined, state, B, T, H)
            _ = run_kernel(lib, lib.launch_fused_scan_forward_checkpointed_f32, combined, state, B, T, H)
        
        lib.sync_device()
        
        # Benchmark original
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_kernel(lib, lib.launch_fused_scan_forward_original_f32, combined, state, B, T, H)
        lib.sync_device()
        orig_time = (time.perf_counter() - start) / bench_iters * 1000  # ms
        
        # Benchmark optimized (checkpointed)
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_kernel(lib, lib.launch_fused_scan_forward_checkpointed_f32, combined, state, B, T, H)
        lib.sync_device()
        new_time = (time.perf_counter() - start) / bench_iters * 1000  # ms
        
        speedup = orig_time / new_time if new_time > 0 else float('inf')
        
        return {
            "original_ms": orig_time,
            "optimized_ms": new_time,
            "speedup": speedup,
        }
    finally:
        cleanup_gpu()


def run_backward_kernel(lib, backward_func, grad_out, grad_next_state, combined, state, 
                        a_star_buf, s_buf, log_values_buf, B, T, H):
    """
    Run backward kernel and return gradients.
    
    Both original and checkpointed versions now use buffers:
    - Original: reads dense buffers (every timestep)
    - Checkpointed: reads sparse checkpoints (every CHECKPOINT_INTERVAL timesteps)
    """
    T_buf = T + 1
    device = combined.device
    
    # Allocate output tensors
    grad_combined = torch.zeros(B, T, 3 * H, device=device, dtype=torch.float32)
    grad_state = torch.zeros(B, 1, H, device=device, dtype=torch.float32)
    
    # Call kernel
    # Both original and checkpointed versions now need buffers
    # Original: reads dense buffers (every timestep)
    # Checkpointed: reads sparse checkpoints (every CHECKPOINT_INTERVAL timesteps)
    backward_func(
        grad_combined.data_ptr(),
        grad_state.data_ptr(),
        grad_out.data_ptr(),
        grad_next_state.data_ptr(),
        combined.data_ptr(),
        state.data_ptr(),
        a_star_buf.data_ptr(),
        s_buf.data_ptr(),
        log_values_buf.data_ptr(),
        T, H, B
    )
    
    # Sync and check for errors
    lib.sync_device()
    err = lib.get_last_error()
    if err and err != b"no error":
        raise RuntimeError(f"CUDA error in backward: {err.decode()}")
    
    return grad_combined, grad_state


def run_backward_correctness_test(
    lib,
    B: int,
    T: int,
    H: int,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """
    Test that checkpointed backward produces same gradients as original.
    """
    try:
        # Create inputs
        combined, state = create_test_tensors(B, T, H)
        
        # Run forward to get buffers and outputs
        out, next_state, a_star, s_vals, log_values_buf = run_kernel(
            lib, lib.launch_fused_scan_forward_original_f32, combined, state, B, T, H
        )
        
        # Create gradient inputs (simulate backward from loss)
        grad_out = torch.randn_like(out)
        grad_next_state = torch.randn_like(next_state)
        
        # Run original backward (reads dense buffers)
        grad_combined_orig, grad_state_orig = run_backward_kernel(
            lib, lib.launch_fused_scan_backward_original_f32,
            grad_out, grad_next_state, combined, state,
            a_star, s_vals, log_values_buf, B, T, H
        )
        
        # Run checkpointed backward (reads sparse checkpoints from same buffers)
        grad_combined_ckpt, grad_state_ckpt = run_backward_kernel(
            lib, lib.launch_fused_scan_backward_checkpointed_f32,
            grad_out, grad_next_state, combined, state,
            a_star, s_vals, log_values_buf, B, T, H
        )
        
        # Compare gradients
        output_names = ["grad_combined", "grad_state"]
        passed, details = compare_outputs(
            [grad_combined_orig, grad_state_orig],
            [grad_combined_ckpt, grad_state_ckpt],
            output_names
        )
        
        return passed, details
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
    Benchmark original vs checkpointed backward.
    
    IMPORTANT: Each backward uses buffers from its matching forward:
    - Original backward uses dense buffers from original forward
    - Checkpointed backward uses sparse buffers from optimized forward
    
    This is the fair comparison for production use.
    """
    try:
        # Create inputs
        combined, state = create_test_tensors(B, T, H)
        
        # Run ORIGINAL forward to get dense buffers for original backward
        out_orig, next_state_orig, a_star_orig, s_vals_orig, log_values_buf_orig = run_kernel(
            lib, lib.launch_fused_scan_forward_original_f32, combined, state, B, T, H
        )
        
        # Run OPTIMIZED forward to get sparse buffers for checkpointed backward
        out_opt, next_state_opt, a_star_opt, s_vals_opt, log_values_buf_opt = run_kernel(
            lib, lib.launch_fused_scan_forward_checkpointed_f32, combined, state, B, T, H
        )
        
        grad_out = torch.randn_like(out_orig)
        grad_next_state = torch.randn_like(next_state_orig)
        
        # Warmup
        for _ in range(warmup_iters):
            _ = run_backward_kernel(
                lib, lib.launch_fused_scan_backward_original_f32,
                grad_out, grad_next_state, combined, state,
                a_star_orig, s_vals_orig, log_values_buf_orig, B, T, H
            )
            _ = run_backward_kernel(
                lib, lib.launch_fused_scan_backward_checkpointed_f32,
                grad_out, grad_next_state, combined, state,
                a_star_opt, s_vals_opt, log_values_buf_opt, B, T, H
            )
        
        lib.sync_device()
        
        # Benchmark original backward (reads dense buffers from original forward)
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_backward_kernel(
                lib, lib.launch_fused_scan_backward_original_f32,
                grad_out, grad_next_state, combined, state,
                a_star_orig, s_vals_orig, log_values_buf_orig, B, T, H
            )
        lib.sync_device()
        orig_time = (time.perf_counter() - start) / bench_iters * 1000
        
        # Benchmark checkpointed backward (reads sparse buffers from optimized forward)
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = run_backward_kernel(
                lib, lib.launch_fused_scan_backward_checkpointed_f32,
                grad_out, grad_next_state, combined, state,
                a_star_opt, s_vals_opt, log_values_buf_opt, B, T, H
            )
        lib.sync_device()
        ckpt_time = (time.perf_counter() - start) / bench_iters * 1000
        
        return {
            "original_ms": orig_time,
            "checkpointed_ms": ckpt_time,
            "speedup": orig_time / ckpt_time if ckpt_time > 0 else float('inf'),
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
    print("CORRECTNESS TESTS")
    print("=" * 70)
    
    all_passed = True
    for size_name, (B, T, H) in configs:
        config_str = f"B={B:4d}, T={T:3d}, H={H:3d}"
        
        try:
            passed, details = run_correctness_test(
                lib, B, T, H
            )
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
    print("BENCHMARKS")
    print("=" * 70)
    print(f"{'Config':<30s} {'Original':>12s} {'Optimized':>12s} {'Speedup':>10s}")
    print("-" * 70)
    
    for size_name, (B, T, H) in configs:
        config_str = f"B={B}, T={T}, H={H}"
        
        try:
            bench = run_benchmark(lib, B, T, H)
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
            passed, details = run_backward_correctness_test(
                lib, B, T, H
            )
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
            # Always show max_diffs for failed tests
            max_diffs = [f"{name}:{d['max_diff']:.2e}" for name, d in details.items()]
            status += f"\n             max_diffs: {', '.join(max_diffs)}"
            all_passed = False
        
        print(f"[{size_name:12s}] {config_str}  {status}")
        
    print()
        
    print("=" * 70)
    print("BACKWARD BENCHMARKS")
    print("=" * 70)
    print(f"{'Config':<30s} {'Original':>12s} {'Checkpointed':>14s} {'Speedup':>10s}")
    print("-" * 70)
    
    for size_name, (B, T, H) in configs:
        config_str = f"B={B}, T={T}, H={H}"
        
        try:
            bench = run_backward_benchmark(lib, B, T, H)
            print(
                f"{config_str:<30s} "
                f"{bench['original_ms']:>10.3f}ms "
                f"{bench['checkpointed_ms']:>12.3f}ms "
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

import torch
import time
import numpy as np
import os

from pufferlib import _C

NUM_STEPS = 6
HORIZON = 4

test_values = torch.tensor([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
], dtype=torch.float32).reshape(NUM_STEPS, HORIZON)

test_rewards = torch.tensor([
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
], dtype=torch.float32).reshape(NUM_STEPS, HORIZON)

test_dones = torch.tensor([
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
], dtype=torch.float32).reshape(NUM_STEPS, HORIZON)

test_importance = torch.tensor([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
], dtype=torch.float32).reshape(NUM_STEPS, HORIZON)

def run_benchmark(num_steps, horizon, num_warmup=3, num_runs=10, enable_profiling=False):
    gamma = 0.99
    lambda_ = 0.95
    rho_clip = 1.0
    c_clip = 1.0

    torch.manual_seed(42)
    np.random.seed(42)

    values = torch.randn(num_steps, horizon, dtype=torch.float32)
    rewards = (torch.rand(num_steps, horizon, dtype=torch.float32) - 0.5) * 0.1
    dones = torch.zeros(num_steps, horizon, dtype=torch.float32)
    dones[:, -1] = 1.0
    dones[torch.rand(num_steps, horizon) < 0.1] = 1.0
    importance = torch.rand(num_steps, horizon, dtype=torch.float32) * 2.0 + 0.5

    advantages_cpu = torch.zeros_like(values)
    for _ in range(num_warmup):
        advantages_cpu.zero_()
        torch.ops.pufferlib.compute_puff_advantage(
            values, rewards, dones, importance, advantages_cpu,
            gamma, lambda_, rho_clip, c_clip
        )

    cpu_times = []
    for _ in range(num_runs):
        advantages_cpu.zero_()
        start = time.perf_counter()
        torch.ops.pufferlib.compute_puff_advantage(
            values, rewards, dones, importance, advantages_cpu,
            gamma, lambda_, rho_clip, c_clip
        )
        cpu_times.append((time.perf_counter() - start) * 1000.0)
    cpu_time = sum(cpu_times) / len(cpu_times)

    if not torch.backends.mps.is_available():
        print(f"Benchmark ({num_steps} steps, {horizon} horizon): MPS not available")
        return

    values_mps = values.to('mps').contiguous()
    rewards_mps = rewards.to('mps').contiguous()
    dones_mps = dones.to('mps').contiguous()
    importance_mps = importance.to('mps').contiguous()
    advantages_mps = torch.zeros_like(values_mps)

    torch.mps.synchronize()
    for _ in range(num_warmup):
        advantages_mps.zero_()
        torch.ops.pufferlib.compute_puff_advantage(
            values_mps, rewards_mps, dones_mps, importance_mps, advantages_mps,
            gamma, lambda_, rho_clip, c_clip
        )
        torch.mps.synchronize()


    mps_times = []

    def run_mps():
        for _ in range(num_runs):
            advantages_mps.zero_()
            torch.mps.synchronize()
            start = time.perf_counter()
            torch.ops.pufferlib.compute_puff_advantage(
                values_mps, rewards_mps, dones_mps, importance_mps, advantages_mps,
                gamma, lambda_, rho_clip, c_clip
            )
            torch.mps.synchronize()
            mps_times.append((time.perf_counter() - start) * 1000.0)

    if enable_profiling:
        with torch.mps.profiler.profile():
            if torch.mps.profiler.is_metal_capture_enabled():
                with torch.mps.profiler.metal_capture("pufferlib_advantage.gputrace"):
                    run_mps()
            else:
                run_mps()
    else:
        run_mps()
    mps_time = sum(mps_times) / len(mps_times)

    mps_diff = (advantages_cpu - advantages_mps.cpu()).abs().max().item()
    print(f"Benchmark ({num_steps} steps, {horizon} horizon): CPU={cpu_time:.4f}ms MPS={mps_time:.4f}ms Speedup={cpu_time/mps_time:.2f}x - MPS Diff {mps_diff:.6f} {'✓ PASSED' if mps_diff < 1e-5 else '✗ FAILED'}")

if __name__ == '__main__':
    gamma = 0.99
    lambda_ = 0.95
    rho_clip = 1.0
    c_clip = 1.0

    advantages_cpu = torch.zeros_like(test_values)
    torch.ops.pufferlib.compute_puff_advantage(
        test_values, test_rewards, test_dones, test_importance, advantages_cpu,
        gamma, lambda_, rho_clip, c_clip
    )

    if not torch.backends.mps.is_available():
        print("MPS not available")
        exit(1)

    values_mps = test_values.to('mps').contiguous()
    rewards_mps = test_rewards.to('mps').contiguous()
    dones_mps = test_dones.to('mps').contiguous()
    importance_mps = test_importance.to('mps').contiguous()
    advantages_mps = torch.zeros_like(values_mps)

    torch.ops.pufferlib.compute_puff_advantage(
        values_mps, rewards_mps, dones_mps, importance_mps, advantages_mps,
        gamma, lambda_, rho_clip, c_clip
    )
    torch.mps.synchronize()

    advantages_mps_cpu = advantages_mps.cpu()

    print("Advantages:")
    for i in range(NUM_STEPS):
        for j in range(HORIZON):
            print(f"{advantages_mps_cpu[i, j]:.2f} ", end='')
        print()

    # check that we're getting the same result on cpu & mps
    max_diff = (advantages_cpu - advantages_mps_cpu).abs().max().item()
    print(f"Max difference: {max_diff:.6f}")
    print("✓ PASSED" if max_diff < 1e-5 else "✗ FAILED")
    print()

    enable_profiling = os.getenv("MPS_PROFILE", "0") == "1"
    if enable_profiling:
        print("Metal profiling enabled (set MPS_PROFILE=1)")
        print("To enable Metal capture, also set: MTL_CAPTURE_ENABLED=1")
        print()

    print("Benchmarks:")
    run_benchmark(8192, 64, enable_profiling=enable_profiling)
    run_benchmark(16384, 64, enable_profiling=enable_profiling)
    run_benchmark(100000, 128, enable_profiling=enable_profiling)
    run_benchmark(1000000, 128, enable_profiling=enable_profiling)

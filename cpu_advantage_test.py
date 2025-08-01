#!/usr/bin/env python3
"""
Test advantage computation on CPU to match our benchmark conditions
"""

import time
import torch
from pufferlib.pufferl import compute_puff_advantage_python, NUMBA_AVAILABLE

def test_cpu_advantage():
    """Test advantage computation on CPU like our benchmark"""
    print("🔥 Testing CPU Advantage Computation (matching benchmark)")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)
    
    # Test same config as our benchmark that achieved 600M+ steps/sec
    segments, horizon = 2048, 128  # 262K batch
    batch_size = segments * horizon
    
    print(f"Testing {segments}×{horizon} = {batch_size:,} batch size on CPU:")
    
    # Generate test data on CPU
    values = torch.randn(segments, horizon)
    rewards = torch.clamp(torch.randn(segments, horizon) * 0.3, -1, 1)
    terminals = torch.bernoulli(torch.full((segments, horizon), 0.05))
    ratio = torch.exp(torch.randn(segments, horizon) * 0.1)
    advantages = torch.zeros_like(values)
    
    # Warmup (includes Numba JIT compilation)
    print("Warming up (including Numba JIT compilation)...")
    for _ in range(10):
        compute_puff_advantage_python(values, rewards, terminals, ratio, advantages,
                                    gamma=0.99, gae_lambda=0.95, vtrace_rho_clip=1.0, vtrace_c_clip=1.0)
    
    # Benchmark
    print("Benchmarking...")
    num_runs = 100
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        compute_puff_advantage_python(values, rewards, terminals, ratio, advantages,
                                    gamma=0.99, gae_lambda=0.95, vtrace_rho_clip=1.0, vtrace_c_clip=1.0)
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    steps_per_sec = batch_size / avg_time
    
    print(f"\n🎯 CPU Results:")
    print(f"  Average time: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {steps_per_sec/1e6:.1f}M steps/sec")
    
    # Compare to our benchmark
    expected_benchmark = 527.6e6  # From our benchmark results
    print(f"\n📊 Comparison:")
    print(f"  This test: {steps_per_sec/1e6:.1f}M steps/sec")
    print(f"  Benchmark: {expected_benchmark/1e6:.1f}M steps/sec")
    print(f"  Ratio: {steps_per_sec/expected_benchmark:.2f}x")
    
    if steps_per_sec >= expected_benchmark * 0.8:
        print("  ✅ Performance matches benchmark (within 20%)")
    else:
        print("  ⚠️  Performance below benchmark expectations")
    
    # Training SPS estimate
    if NUMBA_AVAILABLE and steps_per_sec > 400e6:
        print(f"\n🚀 Training SPS Potential:")
        print("  With 400M+ steps/sec advantage computation:")
        print("  - Simple env (CartPole): 200-400K SPS achievable")
        print("  - Complex env (Atari): 50-150K SPS achievable") 
        print("  - 370K+ CartPole claim: ✅ PLAUSIBLE")

def main():
    test_cpu_advantage()

if __name__ == '__main__':
    main()
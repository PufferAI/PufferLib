#!/usr/bin/env python3
"""
Comprehensive benchmark for PufferLib advantage computation implementations.
Tests CPU+Numba vs GPU implementations across different batch sizes and horizons.
"""

import time
import numpy as np
import torch
import psutil
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Import the advantage computation functions
from pufferlib.pufferl import (
    compute_puff_advantage, 
    compute_puff_advantage_python, 
    NUMBA_AVAILABLE
)

# Force disable Numba for pure Python baseline
import pufferlib.pufferl as pufferl_module
original_numba_available = pufferl_module.NUMBA_AVAILABLE

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    implementation: str
    segments: int
    horizon: int
    batch_size: int
    time_ms: float
    memory_mb: float
    steps_per_sec: float
    device: str
    
    def __post_init__(self):
        self.total_steps = self.segments * self.horizon

class AdvantageComputeBenchmark:
    """Comprehensive benchmark suite for advantage computation"""
    
    def __init__(self, device='cpu', warmup_runs=3, benchmark_runs=10):
        self.device = torch.device(device)
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = []
        
    def generate_test_data(self, segments: int, horizon: int) -> Tuple[torch.Tensor, ...]:
        """Generate realistic test data for advantage computation"""
        device = self.device
        
        # Create realistic data distributions
        values = torch.randn(segments, horizon, device=device) * 0.5 + 1.0
        rewards = torch.clamp(torch.randn(segments, horizon, device=device) * 0.3, -1, 1)
        terminals = torch.bernoulli(torch.full((segments, horizon), 0.05, device=device))
        ratio = torch.exp(torch.randn(segments, horizon, device=device) * 0.1)  # Close to 1.0
        advantages = torch.zeros_like(values)
        
        return values, rewards, terminals, ratio, advantages
    
    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024**2
        elif self.device.type == 'mps':
            # MPS doesn't have direct memory monitoring, use system memory
            return psutil.Process().memory_info().rss / 1024**2
        else:
            return psutil.Process().memory_info().rss / 1024**2
    
    def run_pure_python_benchmark(self, segments: int, horizon: int) -> BenchmarkResult:
        """Benchmark pure Python implementation (no Numba)"""
        # Temporarily disable Numba
        pufferl_module.NUMBA_AVAILABLE = False
        
        try:
            values, rewards, terminals, ratio, advantages = self.generate_test_data(segments, horizon)
            
            # Warmup
            for _ in range(self.warmup_runs):
                compute_puff_advantage_python(values, rewards, terminals, ratio, advantages,
                                            gamma=0.99, gae_lambda=0.95, vtrace_rho_clip=1.0, vtrace_c_clip=1.0)
            
            # Benchmark
            memory_before = self.measure_memory_usage()
            
            if self.device.type in ['cuda', 'mps']:
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                
            start_time = time.perf_counter()
            
            for _ in range(self.benchmark_runs):
                compute_puff_advantage_python(values, rewards, terminals, ratio, advantages,
                                            gamma=0.99, gae_lambda=0.95, vtrace_rho_clip=1.0, vtrace_c_clip=1.0)
                
            if self.device.type in ['cuda', 'mps']:
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                
            end_time = time.perf_counter()
            memory_after = self.measure_memory_usage()
            
            avg_time_ms = (end_time - start_time) * 1000 / self.benchmark_runs
            memory_mb = max(0, memory_after - memory_before)
            steps_per_sec = (segments * horizon * self.benchmark_runs) / (end_time - start_time)
            
            return BenchmarkResult(
                implementation="Pure Python",
                segments=segments,
                horizon=horizon,
                batch_size=segments * horizon,
                time_ms=avg_time_ms,
                memory_mb=memory_mb,
                steps_per_sec=steps_per_sec,
                device=str(self.device)
            )
        finally:
            # Restore Numba availability
            pufferl_module.NUMBA_AVAILABLE = original_numba_available
    
    def run_numba_benchmark(self, segments: int, horizon: int) -> Optional[BenchmarkResult]:
        """Benchmark Numba JIT implementation"""
        if not NUMBA_AVAILABLE:
            return None
            
        # Force CPU for Numba
        cpu_device = torch.device('cpu')
        values, rewards, terminals, ratio, advantages = self.generate_test_data(segments, horizon)
        
        # Move to CPU for Numba
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()
        
        # Warmup (includes JIT compilation)
        for _ in range(self.warmup_runs + 2):  # Extra warmup for JIT
            compute_puff_advantage_python(values, rewards, terminals, ratio, advantages,
                                        gamma=0.99, gae_lambda=0.95, vtrace_rho_clip=1.0, vtrace_c_clip=1.0)
        
        # Benchmark
        memory_before = self.measure_memory_usage()
        start_time = time.perf_counter()
        
        for _ in range(self.benchmark_runs):
            compute_puff_advantage_python(values, rewards, terminals, ratio, advantages,
                                        gamma=0.99, gae_lambda=0.95, vtrace_rho_clip=1.0, vtrace_c_clip=1.0)
        
        end_time = time.perf_counter()
        memory_after = self.measure_memory_usage()
        
        avg_time_ms = (end_time - start_time) * 1000 / self.benchmark_runs
        memory_mb = max(0, memory_after - memory_before)
        steps_per_sec = (segments * horizon * self.benchmark_runs) / (end_time - start_time)
        
        return BenchmarkResult(
            implementation="Numba JIT (CPU)",
            segments=segments,
            horizon=horizon,
            batch_size=segments * horizon,
            time_ms=avg_time_ms,
            memory_mb=memory_mb,
            steps_per_sec=steps_per_sec,
            device="cpu"
        )
    
    def run_gpu_benchmark(self, segments: int, horizon: int) -> Optional[BenchmarkResult]:
        """Benchmark GPU implementation with transfer overhead"""
        if self.device.type not in ['cuda', 'mps']:
            return None
            
        # Test GPU implementation with realistic transfer overhead
        cpu_values, cpu_rewards, cpu_terminals, cpu_ratio, cpu_advantages = self.generate_test_data(segments, horizon)
        
        # Move to CPU to simulate realistic scenario
        cpu_values = cpu_values.cpu()
        cpu_rewards = cpu_rewards.cpu()
        cpu_terminals = cpu_terminals.cpu()
        cpu_ratio = cpu_ratio.cpu()
        cpu_advantages = cpu_advantages.cpu()
        
        # Warmup
        for _ in range(self.warmup_runs):
            values = cpu_values.to(self.device, non_blocking=True)
            rewards = cpu_rewards.to(self.device, non_blocking=True)
            terminals = cpu_terminals.to(self.device, non_blocking=True)
            ratio = cpu_ratio.to(self.device, non_blocking=True)
            advantages = cpu_advantages.to(self.device, non_blocking=True)
            
            # GPU computation (using PyTorch fallback since we don't have CUDA kernel)
            compute_puff_advantage_python(values, rewards, terminals, ratio, advantages,
                                        gamma=0.99, gae_lambda=0.95, vtrace_rho_clip=1.0, vtrace_c_clip=1.0)
            
            # Transfer back
            result = advantages.cpu()
        
        # Benchmark including transfer overhead
        memory_before = self.measure_memory_usage()
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        
        for _ in range(self.benchmark_runs):
            # Transfer to GPU
            values = cpu_values.to(self.device, non_blocking=True)
            rewards = cpu_rewards.to(self.device, non_blocking=True)
            terminals = cpu_terminals.to(self.device, non_blocking=True)
            ratio = cpu_ratio.to(self.device, non_blocking=True)
            advantages = cpu_advantages.to(self.device, non_blocking=True)
            
            # GPU computation
            compute_puff_advantage_python(values, rewards, terminals, ratio, advantages,
                                        gamma=0.99, gae_lambda=0.95, vtrace_rho_clip=1.0, vtrace_c_clip=1.0)
            
            # Transfer back to CPU
            result = advantages.cpu()
            
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        memory_after = self.measure_memory_usage()
        
        avg_time_ms = (end_time - start_time) * 1000 / self.benchmark_runs
        memory_mb = max(0, memory_after - memory_before)
        steps_per_sec = (segments * horizon * self.benchmark_runs) / (end_time - start_time)
        
        return BenchmarkResult(
            implementation=f"GPU + Transfer ({self.device.type.upper()})",
            segments=segments,
            horizon=horizon,
            batch_size=segments * horizon,
            time_ms=avg_time_ms,
            memory_mb=memory_mb,
            steps_per_sec=steps_per_sec,
            device=str(self.device)
        )
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across different scales"""
        print("🚀 Starting Comprehensive Advantage Computation Benchmark")
        print(f"Device: {self.device}")
        print(f"Numba Available: {NUMBA_AVAILABLE}")
        print(f"Warmup runs: {self.warmup_runs}, Benchmark runs: {self.benchmark_runs}")
        print("=" * 80)
        
        # Test configurations: (segments, horizons)
        test_configs = [
            # Small batches
            (8, 32), (16, 32), (32, 32), (64, 32),
            (8, 64), (16, 64), (32, 64), (64, 64),
            (8, 128), (16, 128), (32, 128), (64, 128),
            
            # Medium batches  
            (128, 32), (256, 32), (512, 32),
            (128, 64), (256, 64), (512, 64),
            (128, 128), (256, 128), (512, 128),
            (128, 256), (256, 256),
            
            # Large batches
            (1024, 32), (2048, 32), (4096, 32),
            (1024, 64), (2048, 64), (4096, 64),
            (1024, 128), (2048, 128), (4096, 128),
            (1024, 256), (2048, 256),
            
            # Very large batches
            (8192, 32), (16384, 32),
            (8192, 64), (16384, 64),
            (8192, 128), (16384, 128),
        ]
        
        for segments, horizon in test_configs:
            batch_size = segments * horizon
            print(f"\nTesting: {segments} segments × {horizon} horizon = {batch_size:,} batch size")
            
            # Test each implementation
            implementations = []
            
            # Pure Python baseline
            try:
                result = self.run_pure_python_benchmark(segments, horizon)
                implementations.append(result)
                print(f"  Pure Python: {result.time_ms:.2f}ms, {result.steps_per_sec/1e6:.1f}M steps/sec")
            except Exception as e:
                print(f"  Pure Python: ERROR - {e}")
            
            # Numba JIT
            if NUMBA_AVAILABLE:
                try:
                    result = self.run_numba_benchmark(segments, horizon)
                    if result:
                        implementations.append(result)
                        print(f"  Numba JIT: {result.time_ms:.2f}ms, {result.steps_per_sec/1e6:.1f}M steps/sec")
                except Exception as e:
                    print(f"  Numba JIT: ERROR - {e}")
            
            # GPU implementation
            if self.device.type in ['cuda', 'mps']:
                try:
                    result = self.run_gpu_benchmark(segments, horizon)
                    if result:
                        implementations.append(result)
                        print(f"  GPU+Transfer: {result.time_ms:.2f}ms, {result.steps_per_sec/1e6:.1f}M steps/sec")
                except Exception as e:
                    print(f"  GPU+Transfer: ERROR - {e}")
            
            # Find best implementation
            if implementations:
                best = min(implementations, key=lambda x: x.time_ms)
                print(f"  🏆 Best: {best.implementation} ({best.steps_per_sec/1e6:.1f}M steps/sec)")
                
                # Calculate speedups
                if len(implementations) > 1:
                    baseline = next((r for r in implementations if "Pure Python" in r.implementation), None)
                    if baseline:
                        speedup = baseline.time_ms / best.time_ms
                        print(f"  📈 Speedup vs Python: {speedup:.1f}x")
            
            self.results.extend(implementations)
    
    def analyze_results(self) -> Dict:
        """Analyze benchmark results and provide insights"""
        if not self.results:
            return {}
        
        analysis = {
            'crossover_points': {},
            'scaling_characteristics': {},
            'memory_overhead': {},
            'recommendations': []
        }
        
        # Group results by batch size
        by_batch_size = defaultdict(list)
        for result in self.results:
            by_batch_size[result.batch_size].append(result)
        
        # Find crossover points where GPU becomes better than CPU
        for batch_size, results in sorted(by_batch_size.items()):
            if len(results) < 2:
                continue
                
            cpu_results = [r for r in results if 'cpu' in r.device.lower() or 'numba' in r.implementation.lower()]
            gpu_results = [r for r in results if r.device != 'cpu' and 'gpu' in r.implementation.lower()]
            
            if cpu_results and gpu_results:
                best_cpu = min(cpu_results, key=lambda x: x.time_ms)
                best_gpu = min(gpu_results, key=lambda x: x.time_ms)
                
                if best_gpu.time_ms < best_cpu.time_ms:
                    analysis['crossover_points'][batch_size] = {
                        'cpu_time': best_cpu.time_ms,
                        'gpu_time': best_gpu.time_ms,
                        'speedup': best_cpu.time_ms / best_gpu.time_ms
                    }
        
        # Analyze scaling characteristics
        numba_results = [r for r in self.results if 'numba' in r.implementation.lower()]
        if numba_results:
            # Find performance scaling pattern
            numba_by_size = defaultdict(list)
            for result in numba_results:
                numba_by_size[result.total_steps].append(result.steps_per_sec)
            
            # Calculate scaling efficiency
            sizes = sorted(numba_by_size.keys())
            if len(sizes) >= 3:
                small_perf = np.mean(numba_by_size[sizes[0]])
                large_perf = np.mean(numba_by_size[sizes[-1]])
                analysis['scaling_characteristics']['numba'] = {
                    'small_batch_perf': small_perf / 1e6,
                    'large_batch_perf': large_perf / 1e6,
                    'scaling_efficiency': large_perf / small_perf
                }
        
        # Generate recommendations
        if analysis['crossover_points']:
            min_crossover = min(analysis['crossover_points'].keys())
            analysis['recommendations'].append(
                f"Use GPU for batch sizes >= {min_crossover:,} steps"
            )
        else:
            analysis['recommendations'].append(
                "CPU+Numba consistently outperforms GPU+Transfer at tested scales"
            )
        
        if NUMBA_AVAILABLE:
            numba_results = [r for r in self.results if 'numba' in r.implementation.lower()]
            if numba_results:
                avg_numba_perf = np.mean([r.steps_per_sec for r in numba_results])
                analysis['recommendations'].append(
                    f"Numba JIT achieves {avg_numba_perf/1e6:.0f}M steps/sec average performance"
                )
        
        return analysis
    
    def print_summary_table(self):
        """Print a formatted summary table"""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*100)
        print("COMPREHENSIVE BENCHMARK RESULTS")
        print("="*100)
        
        # Header
        header = f"{'Implementation':<20} {'Segments':<8} {'Horizon':<8} {'Batch Size':<12} {'Time (ms)':<12} {'M Steps/sec':<12} {'Memory (MB)':<12}"
        print(header)
        print("-" * len(header))
        
        # Sort by batch size, then by time
        sorted_results = sorted(self.results, key=lambda x: (x.batch_size, x.time_ms))
        
        current_batch_size = None
        for result in sorted_results:
            if current_batch_size != result.batch_size:
                if current_batch_size is not None:
                    print("-" * len(header))
                current_batch_size = result.batch_size
            
            print(f"{result.implementation:<20} {result.segments:<8} {result.horizon:<8} "
                  f"{result.batch_size:<12,} {result.time_ms:<12.2f} "
                  f"{result.steps_per_sec/1e6:<12.1f} {result.memory_mb:<12.1f}")
    
    def save_results(self, filename: str = 'advantage_benchmark_results.csv'):
        """Save results to CSV file"""
        if not self.results:
            return
        
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['implementation', 'segments', 'horizon', 'batch_size', 
                         'time_ms', 'steps_per_sec', 'memory_mb', 'device']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'implementation': result.implementation,
                    'segments': result.segments,
                    'horizon': result.horizon,
                    'batch_size': result.batch_size,
                    'time_ms': result.time_ms,
                    'steps_per_sec': result.steps_per_sec,
                    'memory_mb': result.memory_mb,
                    'device': result.device
                })
        
        print(f"\n📄 Results saved to {filename}")

def main():
    """Run the comprehensive benchmark"""
    print("PufferLib Advantage Computation Benchmark")
    print("==========================================")
    
    # Detect available devices
    devices = ['cpu']
    
    if torch.cuda.is_available():
        devices.append('cuda')
        print(f"CUDA detected: {torch.cuda.get_device_name()}")
    
    if torch.backends.mps.is_available():
        devices.append('mps')
        print("Apple Metal Performance Shaders detected")
    
    print(f"Numba JIT available: {NUMBA_AVAILABLE}")
    print()
    
    # Run benchmarks on each device
    all_results = []
    
    for device in devices:
        print(f"\n🔥 Running benchmark on {device.upper()}")
        print("=" * 50)
        
        benchmark = AdvantageComputeBenchmark(
            device=device,
            warmup_runs=3,
            benchmark_runs=5
        )
        
        benchmark.run_comprehensive_benchmark()
        all_results.extend(benchmark.results)
    
    # Combine all results for analysis
    final_benchmark = AdvantageComputeBenchmark()
    final_benchmark.results = all_results
    
    # Print comprehensive results
    final_benchmark.print_summary_table()
    
    # Analyze and provide recommendations
    analysis = final_benchmark.analyze_results()
    
    print("\n" + "="*60)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*60)
    
    if analysis.get('crossover_points'):
        print("\n🎯 GPU Crossover Points:")
        for batch_size, data in sorted(analysis['crossover_points'].items()):
            print(f"  Batch size {batch_size:,}: GPU {data['speedup']:.1f}x faster")
    else:
        print("\n💡 CPU+Numba outperforms GPU+Transfer at all tested scales")
    
    if analysis.get('scaling_characteristics', {}).get('numba'):
        scaling = analysis['scaling_characteristics']['numba']
        print(f"\n📊 Numba Scaling:")
        print(f"  Small batches: {scaling['small_batch_perf']:.1f}M steps/sec")
        print(f"  Large batches: {scaling['large_batch_perf']:.1f}M steps/sec")
        print(f"  Scaling efficiency: {scaling['scaling_efficiency']:.1f}x")
    
    if analysis.get('recommendations'):
        print(f"\n🎯 Recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Save results
    final_benchmark.save_results()
    
    print(f"\n✅ Benchmark complete! Tested {len(all_results)} configurations.")

if __name__ == '__main__':
    main()
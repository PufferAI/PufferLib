# PufferLib Hardware Optimization Guide

## Automatic Hardware Detection

PufferLib now automatically detects your hardware and applies optimal training configurations. This works for:
- Apple Silicon Macs (M1/M2/M3/M4) with MPS
- NVIDIA GPUs with CUDA
- CPU-only systems

## How It Works

1. **Automatic Detection**: When you run `puffer train`, the system detects your hardware
2. **Dynamic Scaling**: Configuration scales based on CPU cores, GPU memory, and device type
3. **Smart Defaults**: If config values are set to 'auto' or use suboptimal defaults, they're optimized
4. **Respect User Settings**: Explicitly set values are never overridden

## Dynamic Scaling Rules

### CPU Cores → Number of Environments
- **MPS (Apple Silicon)**: `min(max(4, cores-2), 16)` - leaves headroom for system
- **CUDA (NVIDIA)**: `min(cores*4, 128)` - can handle more parallel envs
- **CPU-only**: `min(cores, 16)` - conservative to avoid oversaturation

### GPU Memory → Batch Size
- **MPS**: 
  - <16GB: batch_size=4096, minibatch=2048
  - 16-32GB: batch_size=8192, minibatch=4096  
  - >32GB: batch_size=min(mem*1024, 16384)
- **CUDA**:
  - <10GB: batch_size=8192
  - 10-20GB: batch_size=16384
  - 20-40GB: batch_size=32768
  - >40GB: batch_size=65536

### Network Size → Device Selection
- **MPS**: Use CPU for networks <100K parameters (overhead too high)
- **CUDA**: Use GPU for networks >10K parameters (lower overhead)
- **Memory Scaling**: Batch sizes automatically reduce for very large networks

## Usage

### Automatic (Recommended)
```bash
# Just run normally - hardware optimization is automatic!
puffer train puffer_snake --wandb
```

### Manual Override
```bash
# Force specific settings if needed
puffer train puffer_snake --wandb \
  --train.device mps \
  --train.batch-size 8192 \
  --train.minibatch-size 4096 \
  --vec.num-envs 8
```

## Performance Tips

1. **Monitor MPS Usage**: Check GPU usage in Activity Monitor > Window > GPU History
2. **Ocean Environments**: These have small networks (~250K params), borderline for MPS benefit
3. **Batch Size**: Larger isn't always better on MPS due to memory transfer overhead
4. **CPU Alternative**: For small networks, `--train.device cpu` may be faster

## Debugging

If you see low MPS utilization (e.g., 16%):
- This is normal for small networks
- The overhead of memory transfers dominates
- Consider using CPU for networks <100K parameters

## Technical Details

The hardware optimizer (`pufferlib/hardware_optimizer.py`):
- Detects CPU cores, GPU type, and available memory
- Applies environment-specific adjustments
- Considers network parameter count for device selection
- Enables non-blocking transfers for MPS/CUDA

## Distributed Training & Multi-GPU Experiments

### Running Experiments on Multiple Machines

PufferLib supports multiple ways to parallelize experiments for faster hyperparameter sweeps and training.

### 1. Parallel Hyperparameter Sweeps (Easiest & Most Effective)

Run independent sweep trials across multiple machines for linear speedup:

```bash
# Machine 1
puffer sweep puffer_snake --wandb --max-runs 50 --seed 1000

# Machine 2  
puffer sweep puffer_snake --wandb --max-runs 50 --seed 2000

# Machine 3
puffer sweep puffer_snake --wandb --max-runs 50 --seed 3000
```

**Benefits:**
- 5 machines = 5x faster sweep completion
- All results aggregate in WandB automatically
- Protein optimizer learns from all experiments
- No network setup required

### 2. Multi-GPU Training (Single Machine)

For machines with multiple GPUs:

```bash
# Use 4 GPUs on one machine
torchrun --standalone --nproc-per-node=4 \
  -m pufferlib.pufferl train puffer_snake --wandb
```

### 3. Distributed Training (Multiple Machines)

For large models across multiple machines:

```bash
# Master node (IP: 192.168.1.100)
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=0 \
  --master-addr=192.168.1.100 --master-port=29500 \
  -m pufferlib.pufferl train puffer_snake --wandb

# Worker nodes
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=1 \
  --master-addr=192.168.1.100 --master-port=29500 \
  -m pufferlib.pufferl train puffer_snake --wandb
```

### 4. Cloud-Based Parallel Sweeps

Using Modal.com for easy cloud parallelization:

```python
import modal

app = modal.App("pufferlib-sweep")

@app.function(
    gpu="h100",  # or "a10g" for cheaper
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb")]
)
def run_sweep_trial(trial_id):
    import subprocess
    subprocess.run([
        "puffer", "sweep", "puffer_snake",
        "--wandb", "--max-runs", "1",
        f"--sweep.seed", str(1000 + trial_id)
    ])

@app.local_entrypoint()
def main():
    # Run 50 trials in parallel on cloud GPUs
    run_sweep_trial.map(range(50))
```

### Speedup Examples

**Hyperparameter Sweep (50 trials):**
- 1 M4 Mac: 4.2 hours
- 5 machines: 50 minutes (5x speedup)
- 10 cloud H100s: 10 minutes (25x speedup)

**Large Model Training (100M+ params):**
- 1 M4 Mac: Baseline
- 1 H100: 50-100x faster
- 4 H100s (DDP): 150-300x faster

### Recommendations

1. **For hyperparameter search**: Use parallel sweeps across machines
   - Simple setup (just different seeds)
   - Linear speedup with number of machines
   - All results visible in WandB

2. **For final training of large models**: Use multi-GPU DDP
   - Best for 100M+ parameter models
   - Enables larger batch sizes
   - Requires more setup

3. **For maximum speed**: Use cloud services
   - Modal.com for simplicity
   - AWS Batch for scale
   - Can run 100s of experiments in parallel

### Cost Considerations

- **Local machines**: High upfront cost, free to run
- **Cloud GPUs**: $2-5/hour per GPU, pay as you go
- **Hybrid approach**: Use local for exploration, cloud for final runs
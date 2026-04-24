# Craftax Moonshot

This folder is the isolated experimental target for high-risk CPU layout and SIMD work.
The oracle remains `ocean/craftax`; the current production fast candidate remains
`ocean/craftax_fast`.

## Verification Contract

Every implementation slice must pass the bitwise oracle harness before it is treated
as a candidate:

```bash
uv run tests/craftax_diff.py \
  --baseline-craftax-dir ocean/craftax \
  --candidate-craftax-dir ocean/craftax_moonshot \
  --seeds 16 \
  --steps 2000 \
  --action-seed 123 \
  --reset-pool-size 1024

uv run tests/craftax_diff.py \
  --baseline-craftax-dir ocean/craftax \
  --candidate-craftax-dir ocean/craftax_moonshot \
  --seeds 16 \
  --steps 2000 \
  --action-seed 321 \
  --reset-pool-size 0
```

The current moonshot observation is packed symbolic float format. It is smaller than
the baseline full binary-float observation, so oracle checks must expand it before
comparison:

```bash
uv run tests/craftax_diff.py \
  --baseline-craftax-dir ocean/craftax \
  --candidate-craftax-dir ocean/craftax_moonshot \
  --candidate-obs-format packed_float \
  --seeds 16 \
  --steps 2000 \
  --action-seed 123 \
  --reset-pool-size 1024
```

Then run an env-only sweep and a Puffer training smoke. Training is required because
raw harness SPS can lie about the production path, GPU copy overlap, and learning
health.

## CPU Target

The Ryzen 9 9950X3D has one 96 MiB V-Cache CCD and one 32 MiB CCD. Benchmark env-only
runs with explicit placement first:

```bash
env OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=threads \
  taskset -c 0-7,16-23 \
  uv run scratch/craftax_variant_sweep.py \
    --candidates ocean/craftax_moonshot \
    --candidate-obs-format packed_float \
    --parity-seeds 4 \
    --parity-steps 1000 \
    --agents 2048 4096 8192 \
    --bench-steps 3000
```

For training, check `nvidia-smi` first because this is a shared GPU machine.

## Implementation Order

1. Keep the Puffer boundary stable: same env name shape, obs dtype, action space,
   reward, terminal, and log semantics.
2. Convert hot state into packet-owned storage while retaining legacy state as the
   oracle-compatible backing store.
3. Convert one hot subsystem at a time, starting with read-only observation encode,
   then spawn scan, then mob update.
4. Avoid syncing hot state back to legacy state every step unless the converted slice
   replaces a full contiguous phase. Per-step sync can erase the layout win.
5. Each slice gets measured in three ways: bitwise oracle harness, env-only SPS, and
   short Puffer training SPS/reward smoke.

## Current Packed Observation

`craftax_moonshot` emits `843` float channels instead of the baseline `3021`:

- `99` visible cells x `8` symbolic channels: block id, item+1, visibility, and
  one mob type+1 slot for each of the five mob classes.
- `51` scalar channels copied exactly from the baseline scalar tail.

This preserves exact expandability back to the baseline full observation while cutting
the observation width by `3.58x`.

# Native trainer throughput benchmarks

This harness compares separately built baseline and candidate binaries without
writing to repository logs, checkpoints, resources, or build outputs.

The default `--checkpoint-policy exact` makes this harness the strict C0
correctness gate. Every scored baseline/candidate pair must produce a
byte-identical final checkpoint. It stops at the first mismatch and retains
both checkpoints plus a mismatch report.

Byte identity is appropriate only for changes claimed to preserve operation
order and arithmetic, such as removing unused event nodes. Optimizations that
change GEMM algorithms, reduction order, or floating-point association require
a separate full-budget, paired multi-seed quality gate across every golden
environment; this short harness cannot establish their learning equivalence.
Some production async configurations are not repeatable even when the same
baseline binary is run twice. Use `--checkpoint-policy record` for those
throughput runs so every mismatch is retained without being misattributed to
the candidate. This never turns a mismatch into correctness evidence.

## Cases

- `affine578-h`: the resolved run-578 recipe using the CPU environment.
- `affine578-cu`: run 578 with the CUDA backend's required one-buffer setting.
- `affine1024x4-h`: run 578 with hidden size 1024 and four effective layers.
- `affine1024x4-cu`: the same stress profile on the CUDA environment backend.
- `breakout-h` and `breakout-cu`: small-network controls.
- `g2048-h`: large learner-dominated control.
- `maze-h`: long-horizon recurrent control.
- `boxoban-h`: large-observation control using an explicitly staged map file.

Run 578's raw sweep layer value was `3.93814397`. The native trainer resolves
that value through an integer field, so the fixture passes the effective value
`3`. The `1024x4` profile changes only hidden size and effective layer count.

## Build contract

The harness does not build or delete anything. Build baseline and candidate
binaries into separate directories with these names:

```bash
./build.sh affine_lock /tmp/puffer-baseline/affine_lock_h
./build.sh affine_lock /tmp/puffer-baseline/affine_lock_cu --cu
./build.sh breakout /tmp/puffer-baseline/breakout_h
./build.sh breakout /tmp/puffer-baseline/breakout_cu --cu
./build.sh g2048 /tmp/puffer-baseline/g2048_h
./build.sh maze /tmp/puffer-baseline/maze_h
./build.sh boxoban /tmp/puffer-baseline/boxoban_h
```

Repeat from the candidate source tree with `/tmp/puffer-candidate` outputs.
Use identical compiler, CUDA, architecture, and environment settings.

## Run

Screen the four Affine cases with five paired trials:

```bash
python3 benchmarks/native_throughput.py \
  --baseline-dir /tmp/puffer-baseline \
  --candidate-dir /tmp/puffer-candidate \
  --cases affine \
  --pairs 5
```

Run a deterministic exact-checkpoint canary by disabling async collection and
using one vector thread:

```bash
python3 benchmarks/native_throughput.py \
  --baseline-dir /tmp/puffer-baseline \
  --candidate-dir /tmp/puffer-candidate \
  --cases goldens \
  --pairs 1 \
  --override base.async=0 \
  --override vec.num_threads=1 \
  --checkpoint-policy exact \
  --boxoban-map /tmp/puffer-maps/boxoban_maps_medium.bin
```

Keep production async settings for the paired performance campaign and record,
rather than abort on, intrinsic checkpoint variation:

```bash
python3 benchmarks/native_throughput.py \
  --baseline-dir /tmp/puffer-baseline \
  --candidate-dir /tmp/puffer-candidate \
  --cases all \
  --pairs 5 \
  --checkpoint-policy record \
  --boxoban-map /tmp/puffer-maps/boxoban_maps_medium.bin
```

Every `--override KEY=VALUE` is stored in campaign metadata and each exact
command. Harness-owned timestep, run ID, log, checkpoint, profiling, and
downsample settings are appended afterward and cannot be displaced by an
override.

Inspect a generated plan without executing a binary:

```bash
python3 benchmarks/native_throughput.py \
  --baseline-dir /tmp/puffer-baseline \
  --candidate-dir /tmp/puffer-candidate \
  --cases all \
  --pairs 5 \
  --boxoban-map /tmp/puffer-maps/boxoban_maps_medium.bin \
  --dry-run
```

Every invocation creates a new mode-0700
`/tmp/puffer-throughput-XXXXXXXX/` campaign and prints its path immediately.
It never removes or overwrites a campaign.

## Measurement

- Baseline/candidate order alternates `A/B`, then `B/A`.
- One short unscored preconditioning run is performed for each case and binary.
- Scored Affine runs use 134,217,728 steps, exactly 512 horizons at 4096 agents
  and horizon 64.
- Steady-state SPS is a least-squares slope of agent steps against native
  uptime after removing the first 10% of steps.
- At least two distinct post-warmup metric samples are required. Exact padded
  terminal duplicates are ignored; real non-monotonic samples still fail.
- Process-wall SPS is recorded separately and includes startup/finalization.
- Results include paired ratios, geometric means, median/MAD, deterministic
  bootstrap 95% confidence intervals, binary hashes, commands, and machine/GPU
  metadata.

The initial performance gate rejects a reproducible regression worse than 3%.
Affine improvement confidence intervals must exclude 1.0. Exact canary
checkpoint hashes must match. Async production mismatches are recorded, and
performance results still require a separate full-budget, paired multi-seed
quality gate.

## Safety

- The harness uses explicit binary paths and never invokes `build.sh`.
- Repository `logs/`, checkpoints, resources, and configurations are not
  modified.
- Boxoban requires `--boxoban-map` pointing to a pre-staged immutable map
  binary; the harness never downloads or generates maps.
- Existing GPU compute processes cause an abort unless
  `--allow-foreign-gpu-processes` is supplied.
- No artifact cleanup is automatic.

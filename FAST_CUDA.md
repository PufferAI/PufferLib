# FAST_CUDA.md — GPU-native trainer speed investigation handoff

Repo: `/home/keith/Git/ml/PufferLib-5.0`, branch `affine-5.0-port`, PR #645
(`PufferAI/PufferLib`, base `5.0`, env-only scope: `ocean/affine_lock/*`).

## Goal

Get a real, measured speedup from the GPU-native (`.cu`) ocean env backend on
serious hardware (RTX 5090, RTX 4090, eventually multi-GPU), **without**
changing training results — checkpoints/metrics should match the CPU (`.h`)
backend, just run faster. The affine_lock env's own PR (#645) is env-scoped
only; anything touching `src/pufferl.cu` / `src/algo.cu` is core, shared by
every ocean env, and must live on a **separate branch**, not #645.

## Hardware in play

- **This box**: AMD Ryzen 9 9950X3D, 16 cores / 32 threads, RTX 5090 (32GB,
  driver 580.105.08, compute cap 12.0 / sm_120).
- **Reference box** (a different agent, "the 5060 box"): RTX 5060, only
  **4 CPU cores / 8 threads**. This matters a lot — see Finding 3.
- User also runs "Puffer boxes" on 4090s and wants this to generalize there
  and to multi-GPU eventually.

## TL;DR finding

**On this hardware, the `.cu` backend currently gives no real speedup over
the CPU `.h` backend, at any network size tested.** The reference agent's
advertised "3.46x" was real on their machine but was measuring their CPU
backend being thread-starved (8 threads), not a GPU architecture win. The
actual bottleneck for a real win is in the **shared trainer** (CUDA graph
launch/dispatch overhead in `src/pufferl.cu`'s epoch loop and the Muon
optimizer, likely `src/algo.cu`), not in `ocean/affine_lock/affine_lock.cu`
itself. The env's own GPU kernels are already fast and small — see Finding 5.

## Build verification (confirms build/env are not the problem)

```
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export NVCC_ARCH=sm_120
unset NVCC_PREPEND_FLAGS
./build.sh affine_lock build/puffer_affine_lock_cu --cu
```

`cuobjdump --dump-resource-usage` on the resulting binary confirms real GPU
kernels embedded, `arch = sm_120`, and for
`gpu_affine_lock_shared_step_kernel`: `REG:36 SHARED:17920` (matches the
expected footprint from the other agent's recipe almost exactly — REG 36 vs
their 38, negligible compiler-version noise). **Build is healthy, not a
stale-CPU-binary problem.**

GPU health during a run: P1 pstate, 100% util, ~558/600W (93%), SM clock
~2820MHz, 78°C, no throttling. **Not a hardware/thermal problem either.**

## Finding 1 — runtime args don't move SPS at matched hypers

All tests: `hidden_size=512, num_layers=3.93814397` (the known-best
`dainty-firefly-578` CPU-backend hypers, perf=0.0636), `vec.num_buffers=1`
(hard-required for GPU backend, see Finding 4), `train.total_timesteps=
67108864` canary.

| config | overall SPS |
|---|---|
| default (agents=4096, minibatch=8192, horizon=64) | 2.586M |
| agents=8192 | 2.534M |
| agents=16384 | 2.575M |
| async=0 | 2.549M |
| horizon=128 | 2.420M |
| **minibatch=32768** | **3.556M** |

Only minibatch size moved anything, and it changes the actual training
recipe (fewer, larger optimizer updates), not a free win. Reference agent's
correction on why: it's not that Muon gets more efficient per-update: it's
that **fewer total updates means fewer total kernel-graph-node launches**,
and launch/dispatch overhead is the real cost (Finding 5), not compute.

`replay_ratio=0` (rollout-only) ceiling: **21.887M SPS** — confirms 3x over
578's 2.73M (target ~8.2M) is theoretically reachable; the bottleneck is
specifically in the train/optimizer phase, not rollout.

## Finding 2 — 578's real CPU-backend baseline (for reference)

From wandb (`kinvert-k/affine3`, run `ktlyegxl`, `dainty-firefly-578`):
`vec.gpu_env=0`, `vec.num_buffers=2`, same hypers as above. **SPS: 2.73M
mean/final, uptime 709.7s for 1.937B steps.** This is the number any GPU
win needs to beat by 3x (~8.2M) to matter.

## Finding 3 — apples-to-apples backend comparison, THIS hardware

Same exact matched-hypers method the reference agent used (same seed,
`async=0`, `cudagraphs=1`, `num_buffers=1`, BF16, identical everything else
between the two binaries — only backend changes):

**Large net** (hidden_size=512, 3.93814397 layers~4, agents=4096, horizon=64,
minibatch=8192, replay_ratio≈1.87... — actually see raw log for exact args
used, effectively 578-style):
- CPU: 2.586M SPS (from Finding 1's baseline row)

**Matched net** (hidden_size=256, num_layers=3, agents=4096, horizon=64,
minibatch=8192, replay_ratio=3, 100,139,008 steps — reference agent's exact
matched-comparison recipe):
- CPU: 3.004M SPS (33.335s)
- CUDA: 3.143M SPS (31.859s)
- **Speedup: 1.05x**

**Tiny net** (hidden_size=64, num_layers=1, agents=8192, horizon=32,
minibatch=8192, replay_ratio=3, 100,139,008 steps — the regime where GPU
should win most, since env-stepping cost is a bigger fraction of total time
relative to a tiny policy):
- CPU: 10.902M SPS (9.185s)
- CUDA: 12.879M SPS (7.775s)
- **Speedup: 1.18x**

**Reference agent's own matched result on their 4-core/8-thread box**, same
256-hidden recipe: CPU 0.311M SPS, CUDA 1.077M SPS, **3.46x**. Their CPU
number is ~10x slower than ours on the identical config — that gap is
entirely CPU-thread starvation (4 cores vs our 32 threads), not a GPU
architecture advantage. **The 3.46x does not reproduce on strong CPU
hardware.**

Open question sent to the reference agent, answer still pending: their
best-tuned SPS after block/lane kernel tuning, at both the 512-hidden and
256-hidden recipes, for direct comparison against our numbers above.

## Finding 4 — why `num_buffers` is capped at 1 for GPU-native

`src/pufferl.cu:978`: `assert(vec->buffers == 1 && "GPU env: num_buffers
must be 1");` inside `env_setup()`.

This is not an arbitrary limitation. CPU backend's `num_buffers=2` lets CPU
worker threads step the *next* rollout buffer's environments while the GPU
trains on the *current* buffer — genuine overlap because CPU cores and GPU
cores are separate hardware. GPU-native env-stepping happens **on the same
GPU** that does training, so naively enabling `num_buffers=2` there would
just interleave GPU work on one stream, not add real parallelism. A real fix
requires running env-step (for the next buffer) on a **separate CUDA
stream** concurrently with the training graph's stream, relying on the
GPU's own multi-stream concurrency (Hopper/Blackwell can do this if kernels
don't fight over the same SMs/queues) — genuine new engineering, not a flag.

## Finding 5 — Nsight profiling: where the time actually goes

Setup used (repeat exactly for any follow-up profiling):

```
BIN=./build/puffer_affine_lock_cu
mkdir -p /tmp/affine-nsys/{logs,checkpoints}

for MB in 8192 32768; do
  nsys profile \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    --trace=cuda,nvtx,cublas \
    --sample=none \
    --cpuctxsw=none \
    --output="/tmp/affine-nsys/mb${MB}" \
    "$BIN" train \
      --base.profile=1 --base.async=1 --base.cudagraphs=1 \
      --base.run_id="nsys-mb${MB}" \
      --base.log_dir=/tmp/affine-nsys/logs \
      --base.checkpoint_dir=/tmp/affine-nsys/checkpoints \
      --base.checkpoint_interval=0 --base.eval_episodes=0 --base.seed=73 \
      --vec.total_agents=4096 --vec.num_buffers=1 \
      --policy.hidden_size=512 --policy.num_layers=3.93814 \
      --train.gpus=1 --train.horizon=64 --train.replay_ratio=1.87009 \
      --train.minibatch_size="$MB" --train.total_timesteps=8388608
done
```

(`--base.profile=1` gates real `cudaProfilerStart()`/`Stop()` calls at
`src/pufferl.cu:2064` / `:2081`, so `--capture-range=cudaProfilerApi` scopes
correctly. Exit code 1 with `missing key env/perf` at the end is harmless —
`eval_episodes=0` means nothing to log; the `.nsys-rep` is still complete.)

Reports pulled: `nsys stats --report cuda_api_sum / cuda_kern_exec_sum:base
/ cuda_gpu_kern_sum / nvtx_gpu_proj_sum --force-export=true <rep>`.

**Results, MB8192 (32 epochs, 8.1M steps, ~3.38s uptime):**
- `cudaStreamSynchronize`: 76.7% of *API* time (2.756s total, 96 calls) —
  this is the CPU blocked waiting for GPU graph replay to finish. It is a
  measurement artifact of CPU-side view, not GPU idle time — ignore the
  dashboard's "0ms train" rows for the same reason (CUDA-event timing is
  unreliable inside a captured graph).
- **Real total GPU kernel-busy time (sum of `cuda_gpu_kern_sum`): only
  ~529ms out of ~3.38s wall-clock — about 15%.** The other ~85% is gaps
  between kernel launches / graph-node dispatch overhead.
- No single kernel exceeds **1.6%** of total accounted GPU time. Nothing to
  point at as "the one slow kernel."
- All six Muon-named kernels (`muon_weight_update`, `muon_store_update`,
  `muon_l2_normalize`, `muon_sum_sq_reduce`, `muon_sum_sq_partials`,
  `muon_clip_nesterov`) combined: **~142ms**, a minority of the 529ms.
- `affine_lock.cu`'s own kernels (`gpu_affine_lock_shared_step_kernel` +
  friends): **~15.8ms** — about 3% of real kernel time. **The env kernels
  are not the bottleneck; block/lane tuning there caps at ~1-5% per the
  reference agent's own report and won't move the needle meaningfully.**
- High-launch-count kernels: `splitKreduce_kernel` (cuBLASLt internal
  reduction) fires **69,856 times** in one run at 1-2µs each; many
  `cutlass::Kernel2<...>` GEMM variants (the real Muon/forward/backward
  matmuls) fire thousands of times at a few µs each. **The signature is
  "thousands of tiny kernel launches, each too small to amortize dispatch
  overhead," not "one kernel is slow."** This matches a network this size
  (hidden_size 256-512, ~3 real layers — note `num_layers=3.93814397` in
  config *logs* as that value but **executes as 3 actual layers**, rounds
  down) being fundamentally launch-overhead-bound on a 5090, not
  compute-bound.

**MB32768 comparison**: real kernel time drops to ~275ms (vs 529ms),
`train_forward_backward` NVTX-projected GPU time drops from ~103ms/epoch to
~70.8ms/epoch (1.46x, not the 4.21x reduction in optimizer-update count —
confirms per-update cost went up as batches got bigger, and the net win
comes from fewer total launches, not cheaper Muon math).

**Reference agent's correction worth preserving**: Muon's ~75 GEMMs/update
mostly show up under generic cuBLAS names (`splitKreduce_kernel`, the
`cutlass::Kernel2<...>` variants) rather than the six Muon-named kernels —
so the *node-count* attribution to Muon is likely higher than the
Muon-named-kernel total of 142ms suggests, even though the physical
diagnosis (launch/dispatch overhead, not compute) holds either way.

## Finding 6 — proposed core fixes (not yet started, need a separate branch)

Two concrete pieces of engineering, both **`src/pufferl.cu` / `src/algo.cu`
core**, both affect every ocean env on GPU, **out of scope for PR #645**:

1. **Stream-overlap for GPU-native env-step vs train** (addresses Finding
   4). Run next-buffer env-step on a separate CUDA stream concurrent with
   the training graph's stream, instead of the current single-stream
   serialization that forces `num_buffers=1`. This is the GPU-native
   equivalent of CPU's double-buffering win.
2. **cuBLASLt separate C/D output for Muon** (reference agent's suggestion,
   addresses Finding 5). Muon currently does a copy-before-GEMM step;
   using cuBLASLt's separate C/D outputs could eliminate that copy and
   remove roughly 50 graph nodes per optimizer update, while keeping the
   same five-step optimizer math (i.e. should be numerically
   equivalent — verify with byte-identical-checkpoint testing like the
   reference agent did for their 3.46x claim).

Neither has been started. Correctness bar for both: checkpoints and eval
metrics should match the CPU `.h` backend at matched hypers (see the
reference agent's own validation method in Finding 3 — byte-identical
checkpoints, "effectively identical" eval metrics).

## Artifacts preserved

- `/tmp/affine-nsys/mb8192.nsys-rep`, `/tmp/affine-nsys/mb32768.nsys-rep` —
  full Nsight captures backing Finding 5.
- `/tmp/affine5090-tune/logs/affine_lock/*.ini` — all canary run logs
  backing Findings 1-3 (`baseline-5090*.ini`, `matched-*.ini`,
  `matched-small-*.ini`).
- These are in `/tmp`, not durable — copy them out before they get cleared
  if you want to keep the raw data rather than just this summary.

## What NOT to do

- Don't touch `ocean/affine_lock/affine_lock.cu` expecting a big win — it's
  already fast (Finding 5), that's not where the time goes.
- Don't touch `src/pufferl.cu` / `src/algo.cu` on the `affine-5.0-port`
  branch / PR #645 — that PR is env-scoped only. Use a separate branch.
- Don't accept a minibatch-size change as a "speedup" without flagging that
  it changes the training recipe (fewer/larger optimizer updates) — it's
  not metric-equivalent to the baseline it's compared against.
- Don't compare CPU-vs-GPU SPS on a CPU-thread-starved machine and
  generalize the ratio to strong hardware — that's exactly the mistake in
  the original "3.46x" pitch (real number, wrong causal attribution).

## 2026-08-20 profile correction and core plan

The raw Nsight SQLite data does not support Finding 5's claim that total GPU
kernel-busy time was only about 529 ms or that roughly 85% of wall time was
empty graph-node dispatch gaps. In the MB32768 capture, the union of kernel
execution intervals is 2,244.640 ms across a 2,397.508 ms kernel span (93.6%).
The GPU can still be underfilled while a kernel is active, but the optimization
target is kernel shape, concurrency, fusion, and memory traffic rather than
another layer of host graph submission.

See `FAST_CUDA_CORE_PLAN.md` for the corrected measurements, separate-branch
plan, permissions, optimization roadmap, correctness tiers, and golden
environment benchmark gates.

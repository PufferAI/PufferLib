# CUDA trainer throughput plan

Date: 2026-08-20

Status: investigation and benchmark design only. No core implementation has
started, no branch has been created, and no benchmark has been run under this
plan.

This document is the durable handoff for making the PufferLib CUDA trainer
substantially faster on high-end GPUs. It supplements `FAST_CUDA.md` and
supersedes that document's interpretation that roughly 85% of the profiled
wall time was empty CUDA graph dispatch gaps.

## Objective

Push the training Pareto front forward on the RTX 5090, with a target of at
least 2x end-to-end training throughput where technically achievable, while:

- preserving training behavior at matched hyperparameters for implementation
  speedup claims;
- measuring final training quality, not SPS alone;
- avoiding regressions across established golden environments;
- keeping core trainer work out of the Affine Lock environment PR;
- keeping `logs/*.ini` and other constellation inputs untouched;
- separating strict implementation wins from retuned Pareto-front results.

A 2-4x absolute improvement may be possible as a combined trainer, kernel,
and workload-shaping program. A 2-4x gain attributable only to
`ocean/affine_lock/affine_lock.cu` is not realistic on the 5090 host because
environment stepping is a small part of total runtime.

Shared changes to `src/pufferl.cu` and `src/algo.cu` can move the absolute
Pareto front, but they also benefit CPU-environment (`.h`) builds. Results must
not present a shared learner optimization as an Affine Lock CUDA-environment
speedup.

## Branch and PR boundary

This work is larger than Affine Lock and belongs on a separate core branch.

Recommended branch structure, to be created only after explicit approval:

1. `cuda-trainer-throughput` from a clean `5.0` base. This contains shared
   trainer changes, benchmark tooling, and core documentation.
2. A local-only integration branch or worktree combining
   `cuda-trainer-throughput` with `affine-5.0-port`. This exists only to measure
   Affine Lock before PR #645 lands.
3. No core commits should be added to the Affine Lock PR.
4. Nothing should be pushed until the correctness and golden-environment gates
   in this document pass.

If Affine Lock lands first, the integration branch becomes unnecessary and the
core branch can be updated from the new `5.0` base before final validation.

Creating, switching, merging, rebasing, committing, or pushing branches is a
Git state mutation and requires explicit user approval.

## Corrected profile interpretation

The preserved Nsight databases do not support the earlier conclusion that the
GPU was executing kernels for only about 15% of wall time.

### MB8192 capture

- CUTLASS `Kernel2`: 178,272 launches, 2,830.752 ms summed duration.
- `splitKreduce_kernel`: 69,856 launches, 137.571 ms summed duration.
- Muon optimizer updates: 1,888, inferred from the one-per-update
  `muon_weight_update` count.

The CUTLASS duration alone is much larger than the earlier claimed total of
about 529 ms.

### MB32768 capture

- Total kernel launches: 100,837.
- Summed kernel duration: 2,695.073 ms. This double-counts overlap across
  streams and is not a wall-time share.
- Union of all kernel execution intervals: 2,244.640 ms.
- Span from first kernel start to last kernel end: 2,397.508 ms.
- At least one kernel was active for 93.6% of that span.
- CUTLASS `Kernel2`: 50,112 launches, 1,771.521 ms summed duration.
- `splitKreduce_kernel`: 16,576 launches, 36.644 ms summed duration.
- Muon optimizer updates: 448.

Comparing the two captures gives an exact decomposition of the launch counts:

- 10,240 fixed CUTLASS kernels per captured run outside optimizer updates;
- 89 main GEMM kernels per optimizer update;
- 37 split-K reduction kernels per optimizer update.

The GPU being occupied does not imply that every kernel fills the SMs. Small
serial GEMMs, scans, and elementwise kernels can keep the device nominally busy
while using only part of its resources. The revised working hypothesis is
underfilled or inefficient kernels plus unnecessary intermediate traffic, not
large empty gaps between graph nodes.

Graph-node tracing can perturb a capture, so the absolute timing must be
re-established without node tracing before implementation decisions are
finalized.

## Existing trainer architecture

- GPU environment observations, actions, rewards, terminals, and masks are
  device-resident.
- GPU rollout captures the entire horizon in one CUDA graph.
- Training captures preprocessing and the complete minibatch loop in one CUDA
  graph.
- Async mode already overlaps the next rollout with learner execution on
  separate nonblocking streams.
- Actor parameters are snapshotted separately from learner parameters.
- GPU environments currently require `vec.num_buffers=1`, but the trainer has
  two async rollout/train slots.
- `base.cudagraphs=0` does not disable graphs in the current implementation;
  only a negative value does. Benchmark commands must account for this.

Consequences:

- Capturing another larger host graph is not the primary opportunity.
- Basic rollout/train overlap is already present.
- Host synchronization cleanup may help, but cannot explain a 2x target.
- The learner and model kernels are the primary target.

## Optimization roadmap

### Track 0: repair measurement and attribution

Before changing core code:

1. Capture a matched baseline without CUDA graph node tracing.
2. Use Nsight Compute on representative Muon GEMMs, split-K reductions,
   MinGRU scans, and forward/backward GEMMs.
3. Record tensor-core utilization, achieved occupancy, memory throughput,
   launch count, kernel duration, and workspace/algorithm choice.
4. Attribute wall time to rollout, train preprocessing, model
   forward/backward, Muon, copies, and logging.
5. Keep all binaries, logs, checkpoints, profiles, and reports under a unique
   `/tmp/puffer-fast/<campaign>/` directory.

### Track 1: Muon, formula-preserving first

Current Muon processes five matrix parameters serially. Each matrix performs:

- a norm reduction and normalization;
- five Newton-Schulz rounds;
- three GEMMs and two copies per round;
- a final store/scale operation.

Across five matrices this is 75 Muon GEMMs and 50 copies per optimizer update.

Implement in this order:

1. Replace legacy copy-before-`cublasGemmEx` operations with cuBLASLt matmuls
   using separate `C` input and `D` output buffers. Cache layouts, descriptors,
   algorithms, and workspace at initialization so graph capture stays static.
2. Benchmark whole-graph GEMM algorithms, including non-split-K candidates.
   Removing a split-K reduction can win end-to-end even when its main GEMM is
   slightly slower in isolation.
3. Allocate disjoint per-parameter scratch and process independent matrices
   concurrently. Batch the identically shaped MinGRU weights where profitable.
4. Use one handle and user-owned workspace per concurrent stream, with an
   explicit fan-out/fan-in dependency before the final update.
5. Batch per-parameter norm, reduction, normalization, and store kernels.
6. Fuse the final FP32 weight update with the BF16 parameter write when this can
   preserve the same values and visibility rules.

These operations can preserve the optimizer formula. Byte-identical results
are not assumed because a different cuBLAS algorithm may change floating-point
reduction order. That must be tested rather than claimed.

### Track 2: MinGRU and model kernels

In the MB32768 capture, named MinGRU kernels account for about 764 ms of summed
kernel duration:

- forward and backward scans: about 581 ms;
- gate and add kernels: about 183 ms.

Candidate work:

1. Fuse gate preparation and residual/add traffic where dependencies permit.
2. Improve the fixed-length affine scan with vectorized, coalesced,
   warp/block-level implementations.
3. Avoid materializing intermediate scan operands when a fused kernel can
   consume them directly.
4. Combine compatible projections into wider GEMMs.
5. Overlap independent `dW` GEMMs with critical-path recurrent backward work,
   joining before optimizer execution.
6. Benchmark cached cuBLASLt algorithms for the exact fixed shapes.

Changing scan association, fusing across BF16 stores, or introducing atomics
can change numerical trajectories. Those changes belong in a separately
labeled numerical-equivalence tier.

### Track 3: CUDA-environment-specific cleanup

This track is important for correctly attributing a `.cu` backend benefit, but
it is secondary for the current Affine Lock workload.

1. Remove the three unconditional timing event nodes captured per rollout
   timestep from the production graph.
2. Keep an instrumented graph only for sampled profiling runs.
3. Fuse reward, terminal, mask, and recurrent-state preparation.
4. Eliminate avoidable observation copies and rollout materializations.
5. Let GPU environments write train-ready fields directly where ownership and
   graph-fixed pointers permit it.
6. Consider a persistent multi-step policy-plus-environment kernel only after
   simpler learner work is measured.

Do not retune `ocean/affine_lock/affine_lock.cu` block or lane geometry expecting
a large end-to-end win. Its measured kernel time is too small.

### Track 4: aggressive Pareto work

Only after formula-preserving changes are measured:

1. Rewrite the Newton-Schulz polynomial in Horner form, reducing Muon from 75
   to 50 GEMMs per update. This is algebraically equivalent but changes BF16
   intermediate rounding.
2. Build specialized grouped or persistent tensor-core Newton-Schulz kernels
   for common matrix shapes.
3. Build persistent or more heavily fused MinGRU paths.
4. Increase minibatch size or reshape agents/horizon to fill the 5090 better.
5. Retune the full training recipe and report a new Pareto front.

These are not strict backend-equivalence results and must never be mixed into
the implementation-only speedup table.

## Correctness tiers

Every result must state one of these tiers.

### C0: implementation-identical

- Same hyperparameters, seed, samples, update count, and operation semantics.
- Byte-identical checkpoints are the preferred gate on the same GPU, toolkit,
  compiler, and build flags.
- If checkpoints are not identical, the result cannot be called C0 without a
  documented reason and a stronger equivalence proof.

### C1: numerically equivalent

- Same optimizer and training recipe, but floating-point association or
  library algorithm can differ.
- Compare per-update losses, gradients, parameter differences, logits, and
  evaluation metrics.
- Require predeclared tolerances and multi-seed non-inferiority.

### C2: Pareto-retuned

- Minibatch, agents/horizon, kernel algebra, precision, or other recipe details
  may change.
- Compare score versus samples, score versus wall time, time to threshold, and
  final quality across multiple seeds.
- Report this as a new Pareto point, not a matched-backend speedup.

## Golden benchmark suite

The suite deliberately spans CUDA and CPU environment backends plus very
different learner shapes.

| Environment | Env backends | Canonical learner/workload | Benchmark role |
|---|---|---|---|
| Affine Lock | `.h`, `.cu` | hidden 512, effective 3 layers, MB 8192, horizon 64 | Primary target and backend parity |
| Breakout | `.h`, `.cu` | hidden 32, 4 layers, MB 65536, horizon 32 | Tiny learner and second true CUDA env |
| G2048 | `.h` only | hidden 1024, effective 5 layers, MB 65536, horizon 64 | Large learner dominated control |
| Maze | `.h` only | hidden 512, 5 layers, MB 16384, horizon 256 | Mid-size learner, long horizon, CPU transfer/overlap |
| Boxoban | `.h` only | hidden 1024, 3 layers, MB 65536, horizon 128 | Large observations/model and puzzle quality control |
| Benchmark env | `.h` | synthetic | Learner-focused diagnostic only, not a quality gate |

Important naming and backend facts:

- The repository environment is `boxoban`; there is no separate `sokoban`
  environment. Sokoban is the game family.
- A normal build uses the CPU `.h` environment plus the CUDA learner.
- `build.sh --cpu` is a standalone CPU play/eval binary, not the correct
  CPU-environment trainer baseline.
- `build.sh ENV OUTPUT --cu` requires `ocean/ENV/ENV.cu` and exclusively uses
  that GPU environment implementation.
- G2048, Maze, and Boxoban have no `.cu` backend. They validate shared learner
  changes but cannot validate CUDA-environment-specific speedups.
- Breakout's CUDA backend should be tested without unsupported self-play
  features.
- G2048's fractional configured layer count is stored in an integer and
  executes as five layers.

## Benchmark harness design

Create one dedicated harness rather than modifying or running existing seed
scripts unchanged.

### Artifact isolation

- Default root: `/tmp/puffer-fast/<campaign>/`.
- Separate `bin`, `logs`, `checkpoints`, `profiles`, `maps`, and `results`
  subdirectories.
- Use separately named baseline and candidate binaries with identical compiler
  flags.
- Run from the repository root because configuration lookup uses relative
  `config/` paths.
- Disable external logging and evaluation for throughput canaries.
- Never write to the repository's `logs/` tree.
- Do not delete old campaign directories without explicit approval.
- Hash binaries, configs, and staged datasets for reproducibility.

Existing `profile.sh` must not be used unchanged. It can silently reuse the
wrong `./puffer` binary, writes artifacts in the repository, force-overwrites
outputs, traces graph nodes by default, and reports summed kernel durations as
percentages even when streams overlap.

Existing multi-seed scripts must not be used unchanged because some write
directly into protected `logs/`, use old CLI syntax, or group conditions in an
order vulnerable to thermal drift.

### Performance gate

1. Build baseline and candidate binaries independently.
2. Use the exact same config, seed, compiler, architecture flags, and runtime
   overrides.
3. Alternate paired order as `A/B`, then `B/A` to reduce thermal/order bias.
4. Use at least five pairs for screening and ten pairs before a public claim.
5. Run long enough to amortize initialization and graph capture.
6. Discard warmup and compute throughput as change in `agent_steps` divided by
   change in `uptime`; do not average logged SPS samples.
7. Report per-environment median, MAD, paired speedup ratios, geometric-mean
   speedup, and bootstrap 95% confidence intervals.
8. Record CUDA/compiler versions, GPU clocks, temperature, power state, CPU
   thread count, buffer count, run order, config overrides, and binary hash.
9. Report native wall time for end-to-end claims and `perf/*` splits only for
   diagnosis.
10. Include Affine Lock and Breakout in both `.h` and `.cu` modes. Include
    G2048, Maze, and Boxoban in `.h` mode.

### Quality gate

1. Use paired fixed seeds. The existing Breakout convention
   `11, 22, 33, 44, 55` is a reasonable initial common seed set.
2. Use representative or full training budgets, not short throughput canaries.
3. Compare final score, score-versus-step AUC, score-versus-time AUC, success
   fraction, and time/steps to predefined thresholds.
4. Define non-inferiority tolerances before viewing candidate results.
5. Evaluate checkpoint/trajectory parity for `.h` versus `.cu` Affine Lock and
   Breakout where the semantics support it.
6. Report environment-specific quality metrics.

Environment-specific metrics:

- Affine Lock: solve rate, maximum solved depth, depth-specific solve rates,
  efficiency, and the existing `env/perf` objective.
- Breakout: score, episode return, episode length, and any established
  score/survival thresholds.
- G2048: score, merge score, maximum tile, and tile reach rates.
- Maze: score, episode return, completion/success rate if exposed, and episode
  length.
- Boxoban: targets hit, solve/success rate if exposed, episode return, and
  episode length.

### Boxoban dataset safety

The current repository does not contain the medium map binary required by the
default Boxoban run. A normal run can download levels and generate files under
`resources/boxoban`, which violates artifact isolation.

Before Boxoban benchmarking:

1. Stage a valid map binary under the campaign's `/tmp` map directory.
2. Record its checksum.
3. Pass `--env.map_bin=/tmp/puffer-fast/<campaign>/maps/...` explicitly.
4. Use the identical staged binary for every baseline and candidate run.

## Acceptance criteria

Exact thresholds should be finalized from baseline variance before inspecting
candidate results. Initial policy:

- Primary target: at least 2x Affine Lock CUDA end-to-end SPS at the strictest
  correctness tier that can support it.
- No golden environment may regress by more than 3% median SPS unless the
  paired confidence interval shows the change is noise or the code path is
  explicitly gated away from that environment.
- A shared core optimization should improve the suite geometric mean and not
  merely transfer time between rollout and training counters.
- No quality metric may fail its predeclared non-inferiority tolerance.
- No new nondeterminism, race, OOM, or unsupported backend behavior.
- Memory growth and workspace use must be reported; a throughput win that
  prevents canonical configurations from fitting is a regression.
- Every speed claim names its correctness tier and includes matched baseline
  data.

## Expected gain envelope

- Graph and host synchronization cleanup alone: likely low single digits.
- Muon copy removal, algorithm selection, reduction batching, and independent
  matrix concurrency: plausible 1.3-1.8x end-to-end if profiling confirms
  underfilled Muon work.
- Muon plus MinGRU/model fusion: the best credible route to a strict-recipe 2x.
- Persistent kernels plus Pareto retuning: potentially 2-4x, but a larger
  research effort and not a C0 claim.
- Strict byte-identical 4x on the current recipe: unlikely based on the current
  kernel occupancy evidence.

These are hypotheses, not promised results. The paired benchmark suite is the
gate.

## Operational permissions and protected state

Standing rules for this work:

Allowed without asking:

- Read files required for the investigation.
- Edit or create files inside the PufferLib repository, including core source.
- Research official technical documentation.

Ask first:

- Any `rm`, unlink, truncation, destructive overwrite, or cleanup.
- Any Git state mutation, including add, commit, push, branch, switch, stash,
  reset, rebase, merge, or clean.
- Builds, training runs, benchmark runs, or profiler runs that consume the GPU
  or create artifacts.
- Package, compiler, driver, system, or privilege changes.
- Process termination or edits outside this repository.
- Any external write such as W&B, GitHub, upload, or network publishing.

Protected:

- Treat `logs/*.ini` and the repository `logs/` tree as read-only.
- Never overwrite, move, truncate, or delete constellation logs.
- Route all experimental outputs to an isolated `/tmp` campaign.
- Stop and ask if unexpected external changes appear.

## Proposed execution sequence

1. Obtain approval to create the separate core and local integration branches.
2. Obtain approval for isolated build, baseline, and profiling commands.
3. Build the unified benchmark harness with `/tmp` defaults and no external
   logging.
4. Establish the full golden baseline matrix before core changes.
5. Correct profiling and attribute Muon, MinGRU, model GEMM, rollout, transfer,
   and logging time.
6. Implement the smallest formula-preserving Muon change.
7. Run microbenchmarks, then the performance gate, then the quality gate.
8. Keep or revert the change based on measured suite-wide results.
9. Continue one independently attributable optimization at a time.
10. Start C1/C2 aggressive work only after C0 opportunities are exhausted and
    explicitly authorized.

No speedup should be accepted because one short Affine Lock run got faster.
The unit of success is a reproducible, correctly classified improvement across
the golden matrix with preserved training quality.

## Locked Affine Lock best-run reference

The following configuration is the current best-run quality reference supplied
by the user. Do not silently change its minibatch, replay ratio, horizon,
network, optimizer, or evaluation settings when making an
implementation-equivalent speed claim.

Run identity:

- `base.run_id=sweep_1787042600825_0515`
- `base.env_name=affine_lock`
- `base.seed=73`
- `env.seed=42`
- `train.seed=42`
- `base.wandb=false`

Backend and vectorization:

- CPU environment backend: `vec.gpu_env=0`
- `base.async=1`
- `base.cudagraphs=1`
- `vec.total_agents=4096`
- `vec.num_buffers=2`
- `vec.num_threads=16`
- `train.gpus=1`

Environment:

- `env.start_depth=2`
- `env.max_depth=16`
- `env.num_agents=1`
- `env.num_bots=0`
- `env.step_grace=0`
- `env.perf_weighting=1`

Network:

- `torch.network=MinGRU`
- `torch.encoder=DefaultEncoder`
- `torch.decoder=DefaultDecoder`
- `policy.hidden_size=512`
- `policy.num_layers=3.93814397`
- `policy.expansion_factor=1`

The current native trainer stores `policy.num_layers` in an integer, so this
configuration executes three MinGRU layers. Preserve the current conversion
behavior for matched results.

Training:

- `train.horizon=64`
- `train.minibatch_size=8192`
- `train.replay_ratio=1.87008977`
- `train.total_timesteps=1936646020`
- `train.learning_rate=0.00326743093`
- `train.anneal_lr=1`
- `train.min_lr_ratio=0`
- `train.gamma=0.999899983`
- `train.gae_lambda=0.921891332`
- `train.clip_coef=0.366497159`
- `train.vf_coef=0.100000001`
- `train.vf_clip_coef=0.00100000005`
- `train.ent_coef=0.0405269228`
- `train.anneal_ent_coef=0`
- `train.min_ent_coef_ratio=0.1`
- `train.momentum=0.924182773`
- `train.max_grad_norm=0.881812513`
- `train.vtrace=0`
- `train.verb_eps=0`

Evaluation:

- `base.eval_episodes=10000`
- `base.burnin_games=0`
- `base.reset_every_horizon=0`

This exact run is a CPU-environment quality reference and uses two vector
buffers. The current CUDA environment requires one vector buffer, so it cannot
be presented as a byte-for-byte backend comparison without addressing that
structural difference. Use two distinct comparisons:

1. Baseline versus candidate within the `.h` backend, holding this
   configuration fixed.
2. Baseline versus candidate within the `.cu` backend, holding its valid
   one-buffer configuration fixed.

The final Pareto comparison can compare the best valid `.h` and `.cu` systems,
but must disclose the buffer difference. Increasing minibatch size or reducing
optimizer-update count belongs only in C2 Pareto-retuned results.

## Universal-first optimization policy

The first changes should remove objectively redundant work or preserve an
existing operation and dependency order. Do not replace one global path with a
path that helps medium matrices but slows small or large matrices.

Rules:

1. Retain the existing implementation as a fallback for shape-sensitive work.
2. Select GEMM or concurrency paths per exact shape and architecture rather
   than applying one algorithm globally.
3. Require paired golden-suite measurements before enabling a new path by
   default.
4. Reject or gate a path that causes a reproducible regression greater than 3%
   in any golden environment.
5. Prefer removing copies, redundant event nodes, redundant scalar work, and
   unnecessary synchronization before changing floating-point association.
6. Keep operation-order-changing fusion, grouped algorithms, persistent
   kernels, and workload retuning in later, separately labeled work.

Universal does not require every environment to gain the same percentage. It
means a core change has no material regression on unsupported shapes and uses
the old path when the new path is not a measured win.

## Proposed PR split

Each PR should be independently benchmarkable and should not depend on a large
unreviewable stack.

### PR 1: benchmark and measurement infrastructure

- Add the isolated golden-environment harness.
- Correct overlapping-kernel accounting.
- Add paired A/B statistics and quality gates.
- Add `.h`/`.cu` parity coverage for Affine Lock and Breakout.
- Make no training-math changes.

### PR 2: exact trainer cleanup

- Remove unconditional rollout timing nodes from production graphs.
- Replace avoidable host synchronizations with explicit stream dependencies.
- Make logging and scalar transfers asynchronous where ordering is unchanged.
- Remove other demonstrably redundant work that preserves the same kernels and
  arithmetic order.

### PR 3: Muon copy and GEMM path

- Add cuBLASLt separate `C`/`D` output support.
- Eliminate Newton-Schulz copy-before-GEMM operations.
- Cache deterministic per-shape algorithms and preserve the legacy fallback.
- Treat the PR as C0 only if checkpoint identity actually passes; otherwise
  evaluate and label it as C1.

### PR 4: Muon concurrency and reduction batching

- Add disjoint per-parameter scratch.
- Batch or concurrently execute only shapes with a measured win.
- Batch norm and elementwise work while preserving reduction order where
  possible.
- Keep large, already-saturating matrices on the legacy serial path if that is
  faster.

### PR 5: MinGRU/model kernels

- Optimize scans, gates, model GEMMs, and backward overlap.
- Separate exact scheduling changes from numerically different scan/fusion
  algorithms.
- Require all five golden environments because this code is shared broadly.

### PR 6: CUDA environment pipeline

- Improve direct device staging and remove duplicate rollout materialization.
- Cover Affine Lock and Breakout `.cu` plus their `.h` controls.
- Keep environment-specific kernels out of the shared learner PRs.

### Separate research results: Pareto retuning

- Horner-form Muon, persistent kernels, minibatch changes, and agents/horizon
  reshaping are C1/C2 experiments.
- Do not bundle them with low-risk core cleanup.
- Report them as new Pareto points rather than transparent implementation wins.

The first two PRs provide a bounded go/no-go point before investing in the
deeper Muon and MinGRU work.

## C0 validation record (2026-08-20)

The first exact cleanup candidate removes unused per-step rollout timing event
nodes when CUDA graphs are enabled. Baseline and candidate were built
independently from clean source into a unique mode-0700 campaign tree under
`/tmp/puffer-c0-golden-jCEYhkTm`; repository logs and resources were not used.

- Affine run 578 `.h`/`.cu` and the 1024x4 stress `.h`/`.cu` produced
  byte-identical baseline/candidate checkpoints at 8,388,608 steps.
- Breakout `.h` was not self-repeatable with its production async settings.
  It also was not self-repeatable with async disabled while retaining multiple
  vector threads. A one-thread, async-off clean baseline canary was exactly
  repeatable.
- With that deterministic canary, Breakout `.h`, Breakout `.cu`, G2048 `.h`,
  and Maze `.h` produced byte-identical baseline/candidate checkpoints at
  67,108,864 steps. The retained campaign is
  `/tmp/puffer-c0-golden-jCEYhkTm/puffer-throughput-rndn2pa8`.
- The short Affine campaign geomean was 1.0019x with a noisy worst pair of
  0.9870x. The deterministic non-Affine campaign geomean was 0.9974x. These
  one-pair C0 screens establish neither a gain nor a regression.
- Boxoban `.h` was self-repeatable and produced byte-identical
  baseline/candidate checkpoints at 33,554,432 steps using an immutable staged
  450,000-puzzle map. The archive SHA256 is
  `fbd7b1efb4e7dd77e06d390051d60fbcc61f11efe127f0c5edaf9ec1547a417b` and
  map SHA256 is
  `87bf4fc7c180895f4b3a75d0393df9909feb8f97de76494c978273ec72bddb53`.
  The retained campaign is
  `/tmp/puffer-c0-golden-jCEYhkTm/puffer-throughput-jm7mqcmo`.
- The Boxoban corpus came from mutable upstream `main`; the recorded hashes
  freeze this campaign but do not create a repository-defined canonical
  corpus. A future fixture should pin the upstream revision or release.

Exact canaries and production performance runs are deliberately separate.
Production async runs use paired statistics and record checkpoint variation;
full-budget, multi-seed learning-quality results remain mandatory before a
core trainer PR can merge.

## Exact optimization experiment log

These candidates were tested against the committed exact-cleanup binaries.
They are retained here even when rejected so later work does not repeat a
failed optimization.

### Muon scalar broadcast: rejected

Computing the clip coefficient and matrix inverse norm once per CUDA block
instead of once per element preserved byte-identical Affine checkpoints. A
balanced two-pair production screen measured `0.9983x` on Affine 578 CUDA,
`1.0060x` on 1024x4 CUDA, and `1.0021x` combined. This is below a credible
signal and not a universal win, so the source change was removed. Artifacts:
`/tmp/puffer-muon-scalar-DOOmXEUF/puffer-throughput-dcmitjed`.

### Batched Muon norm launches: rejected

Batching the identical per-matrix partial reduction, final reduction, and
normalization work from `3P` graph nodes to three preserved byte-identical
checkpoints. A balanced two-pair production screen measured `1.0040x` on
Affine 578 CUDA, `0.9983x` on 1024x4 CUDA, and `1.0011x` combined; the
1024x4 pairs disagreed in direction. The source change was removed. Artifacts:
`/tmp/puffer-muon-batchnorm-pKIiyQFT/puffer-throughput-0t98qn5h`.

### Actor snapshot host-wait removal: rejected as non-C0

Replacing the per-epoch host synchronization with explicit actor-ready
stream events passed the Affine 578 CUDA checkpoint but changed the 1024x4
CUDA checkpoint under production async settings. The speed screen was skipped
and the source change was removed. Artifacts:
`/tmp/puffer-actor-ready-iHjn6riQ/puffer-throughput-4vrgbtor`.

These results reinforce the corrected profile interpretation: fixed graph
bookkeeping is not the 5090 bottleneck. Further work must reduce or overlap
the dominant Muon GEMMs/copies or optimize MinGRU computation while continuing
to apply the exact-checkpoint gate first.

### Concurrent per-matrix Muon: promoted

Running each independent 2D parameter's unchanged Muon pipeline on a disjoint
stream, handle, workspace, norm scratch, and NS scratch produced the first
clear C0 gain. The main stream forks after global clip/Nesterov and rejoins all
lanes before the unchanged flat weight update.

- All Affine, Breakout, G2048, Maze, and Boxoban deterministic canaries
  produced byte-identical baseline/candidate checkpoints.
- The final balanced two-pair production build measured `1.1926x` on run 578
  CUDA and `1.0665x` on 1024x4 CUDA, for `1.1278x` combined. Every pair
  improved and every checkpoint matched. Artifacts:
  `/tmp/puffer-muon-final2-b6gIHRz7/puffer-throughput-9y3xw5y0`.
- A deterministic Breakout CUDA sentinel measured `1.1565x`. The broader
  exact promotion screen measured `1.1963x` Breakout CUDA, `1.0243x`
  Breakout CPU, `1.0995x` Maze, and `1.0095x` Boxoban in one pair each.
- G2048 initially measured `0.9973x` in a balanced two-pair production run. A
  shape-based saturated-workload fallback now selects the legacy serial path
  only when there are at least seven matrices, at least five heavy
  `3072x1024`-class lanes, and at least 15 Mi matrix elements. With that gate,
  G2048 produced byte-identical checkpoints and `1.0009x` across two balanced
  pairs during selector qualification. Artifacts:
  `/tmp/puffer-muon-concurrent-gated-Bp5B17Q6/puffer-throughput-35l7l3c4`.

The final two-pair production golden screen measured `1.1129x` Breakout CUDA,
`1.0292x` Breakout CPU, and `1.0281x` Maze. Boxoban at `0.9972x` and the
G2048 serial fallback at `0.9953x` were statistically flat; all pairs remained
above `0.987x`. Production checkpoint variation was recorded rather than
treated as candidate divergence because those async configurations are not
self-repeatable. The deterministic final-build canaries were byte-identical in
all cases. Artifacts:
`/tmp/puffer-muon-final2-b6gIHRz7/puffer-throughput-_vlevnd0`.

The selected implementation uses one private 32 MiB cuBLAS workspace per
enabled matrix. Concurrency is capped at eight matrices, bounding private
workspace use at 256 MiB; larger models use the allocation-free serial path.
The two legacy 32 MiB allocations were removed because their per-call
`cublasSetStream` reset them to the default pool before every GEMM. A fixed
three-lane experiment reduced memory but lost `5.5%` relative throughput on
Affine 512. An 8 MiB-per-lane experiment stayed exact but lost about `0.5-0.7%`
relative throughput. The 32 MiB per-matrix version remains the measured speed
Pareto point for this 5090 campaign.
## Concurrent Muon cleanup validation

Removed the experimental CUDA/cuBLAS/host assertion-and-abort wrappers. The
optimization now follows the existing direct-call style; the max-eight lane
gate, saturated-workload serial fallback, workspace policy, scheduling,
fork/join topology, and optimizer arithmetic are unchanged.

Fresh binaries:

- `/tmp/puffer-clean-bin-dW4YM5Cg`

Deterministic exact-checkpoint results against the immutable pre-optimization
baseline `/tmp/puffer-c0-golden-jCEYhkTm/candidate-bin`:

- Affine 578 `.h` and `.cu`: exact.
- Affine 1024x4 `.h` and `.cu`: exact.
- Breakout `.h` and `.cu`: exact.
- G2048 `.h`: exact.
- Maze `.h`: exact.
- Boxoban `.h`: exact.
- Affine artifact (the combined campaign stopped after these successful cases
  when the following short case lacked enough uptime samples):
  `/tmp/puffer-clean-validate-kFHz6jz5/puffer-throughput-7jkeeli3`.
- Breakout artifact:
  `/tmp/puffer-clean-breakout-v9mapnzs/puffer-throughput-6mh11ek9`.
- G2048/Maze/Boxoban artifact:
  `/tmp/puffer-clean-heavy-goldens-DJVTxiOy/puffer-throughput-78q3o_ak`.

Balanced production throughput:

- Affine 578 CUDA: `1.176870159x` geomean, pair range
  `1.153879229x` to `1.200319180x`.
- Affine 1024x4 CUDA: `1.076241413x` geomean, pair range
  `1.072833497x` to `1.079660154x`.
- Combined Affine geomean: `1.125431651x`.
- All Affine checkpoints matched exactly.
- Artifact:
  `/tmp/puffer-clean-affine-balanced-nqAzWvrl/puffer-throughput-50ps5sl8`.

The single deterministic G2048 timing pair was noisy (`0.962639308x`), so the
unchanged serial-fallback case was repeated for three production pairs. The
repeat measured `1.000669099x` geomean with a `0.997054066x` worst pair and
matching checkpoints:
`/tmp/puffer-clean-g2048-repeat-lwKRoN0E/puffer-throughput-l4dudlth`.
## Minimum-code concurrent Muon ablations

Goal for this pass: retain only code that is required for measured throughput,
bit-identical arithmetic, CUDA-graph fork/join, or bounded resources. New
defensive wrappers and unused generalizations are not part of the optimization.

### Retained cleanup

- Removed all custom CUDA/cuBLAS/allocation assert-and-abort wrappers.
- Restored the original legacy cuBLAS initializer and call sites.
- Removed redundant pointer initialization and optional-workspace plumbing.
- Removed the abandoned shared-lane load balancer, matrix-to-lane indices,
  max-scratch aggregation, and duplicate warmup search.
- Replaced heap metadata/lane arrays with fixed eight-entry storage, matching
  the measured and qualified eight-lane cap.
- Folded parameter discovery and matrix descriptor construction into one scan.
- Replaced duplicate concurrency state with `num_lanes` (`0` means serial).
- Removed the redundant total-element saturation threshold; five heavy
  `3072x1024`-or-larger matrices already imply the same 15 Mi-element bound.
- Reused the existing maximum-dimension scan for the 4096 qualification gate.
- Replaced unused tensor-shaped lane scratch metadata with raw device pointers.
- Packed each lane's 256 norm partials and one norm scalar into one allocation.
- Kept the original shared serial scratch allocation/registration unchanged.
- Removed the separate lane cuBLAS initializer. Lane setup uses the original
  initializer, binds the private stream, then restores the private workspace
  because `cublasSetStream` resets it.

### Ablation: remove lane GEMM warmup - kept removed

The entire per-lane cuBLAS warmup helper and calls were deleted. Fresh lane
handles successfully captured and replayed without it. Affine 578 CUDA remained
byte-identical and measured `1.006785243x` versus the already-clean concurrent
candidate in the initial canary.

Artifact:
`/tmp/puffer-min-canary-XSgjuhth/puffer-throughput-9bnq23jd`.

Decision: keep the warmup deleted.

### Ablation: remove largest-first scheduling - rejected

Removing `MuonMatrix::work` and the stable insertion sort preserved exact
checkpoints but reduced throughput versus the sorted minimum-code candidate:

- Affine 578 CUDA: `0.974554593x` (`-2.54%`).
- Affine 1024x4 CUDA: `0.987948112x` (`-1.21%`).
- Combined: `0.981228501x`.

Artifact:
`/tmp/puffer-nosort-ab-dBHD6INB/puffer-throughput-hcr26xye`.

Decision: restore the small work field and stable largest-first insertion sort.
Those lines have measured value and remain in the implementation.

### Final minimum-code validation

Fresh binaries:
`/tmp/puffer-min-final-bin-e985GY0q`.

Deterministic checkpoint comparison against the immutable pre-optimization
baseline passed exactly for Affine 578 `.h/.cu`, Affine 1024x4 `.h/.cu`,
Breakout `.h/.cu`, G2048 `.h`, Maze `.h`, and Boxoban `.h`.

- Affine/G2048/Maze/Boxoban artifact:
  `/tmp/puffer-min-final-exact-Gwxuuy1H/puffer-throughput-zthgmrb4`.
- Breakout artifact:
  `/tmp/puffer-min-final-breakout-JHe4038Y/puffer-throughput-_ib_t7di`.

Three-pair production Affine result against the immutable pre-optimization
baseline:

- Affine 578 CUDA: `1.167157783x` geomean, range `1.159652395x` to
  `1.181321136x`.
- Affine 1024x4 CUDA: `1.073290530x` geomean, range `1.063182301x` to
  `1.079850177x`.
- Combined Affine geomean: `1.119240544x`.
- Every paired checkpoint matched exactly.
- Artifact:
  `/tmp/puffer-min-final-affine-yxtO2Tb8/puffer-throughput-ud3m_3ae`.

## 2026-08-20: post-Muon hotspot profiling and strict-C0 ablations

Baseline for this round: commit `9a2afdae` (`cuda: overlap independent Muon matrix updates`) on RTX 5090 with CUDA 13.1. All candidates below used the native throughput harness, isolated `/tmp` binaries/artifacts, and exact checkpoint comparison. No candidate was retained in `src/algo.cu`; the committed Muon implementation remains the source baseline.

### Nsight Systems attribution

| GPU work class | Affine 578 | Affine 1024x4 |
| --- | ---: | ---: |
| Model/non-Muon GEMMs | 43.21% | 46.95% |
| Muon lane GEMMs | 31.26% | 35.29% |
| MinGRU custom kernels | 14.24% | 10.09% |
| Muon lane D2D copies | 4.23% | 2.57% |
| Muon lane custom kernels | 2.46% | 1.61% |

Reports: `/tmp/puffer-nsys-affine578-AeLPq6DN/affine578.nsys-rep` and `/tmp/puffer-nsys-affine1024-xrrO5zxs/affine1024.nsys-rep`. MinGRU scan backward was about 7.6% of summed kernel time on 578 and 4.7% on 1024x4; forward was about 3.8% and 2.4%. Nsight Compute hardware-counter collection was attempted but rejected by the driver with `ERR_NVGPUCTRPERM`; no permission or driver setting was changed. Log: `/tmp/puffer-ncu-gemm-4bbRUQRm/ncu.log`.

### Ablation ledger

| Candidate | Exact checkpoints | Result | Decision |
| --- | --- | --- | --- |
| Rollout MinGRU state update in place, deleting one D2D copy per layer | Yes | Affine suite `1.000075x`; isolated Boxoban at original 256-thread scans `0.992227x` over 3 pairs | Reject: not universally free |
| Ordinary MinGRU scans at 128 threads, initially stacked on in-place rollout | Yes | Short Affine test: 578 `1.002227x`, 1024x4 `1.010714x`, suite `1.006461x` | Promising short result, required clean isolation |
| Ordinary MinGRU scans at 512 threads, stacked on in-place rollout | Yes | 578 `1.009817x`, 1024x4 `0.999703x`, suite `1.004747x` | Reject: size-dependent |
| Ordinary MinGRU scans at 64 threads, stacked on in-place rollout | Yes | Versus in-place 256: 578 `1.013120x`, 1024x4 `1.013406x`; versus committed baseline over 3 pairs: suite `1.010901x` | Reject: Boxoban `0.985530x` over 3 pairs |
| Ordinary MinGRU scans at 128 threads with the in-place change removed | Yes | Affine 578 `0.999373x`, 1024x4 `0.997186x`, suite `0.998279x`; Boxoban `0.998644x` | Reject: no repeatable gain |

The positive short 128/64 results were not accepted because the better-isolated repeats contradicted them or exposed an environment regression. This is why throughput changes need paired repeats and the golden suite even when arithmetic is trivially bit-preserving.

Ordinary scan shapes were: Affine 578 `B*H=65,536,T=64`; Affine 1024x4 `131,072,64`; Breakout `65,536,32`; G2048 `1,048,576,64`; Boxoban `524,288,128`. Maze (`32,768,256`) uses the existing row-scan path and was unaffected. All ordinary totals divide both 128 and 256 exactly, so the rejected geometry experiments changed scheduling only, not arithmetic or tail behavior.

### Golden results for the rejected 64-thread/in-place candidate

All nine deterministic backend/config checkpoints matched exactly: Affine 578 `.h/.cu`, Affine 1024x4 `.h/.cu`, Breakout `.h/.cu`, G2048 `.h`, Maze `.h`, and Boxoban `.h`. The full one-pair artifact roots are `/tmp/puffer-hotspots-golden-UecOhctF/standard/puffer-throughput-f9kt97kw` and `/tmp/puffer-hotspots-golden-UecOhctF/breakout/puffer-throughput-l9sef9qv`. Repeat artifacts are `/tmp/puffer-hotspots-regression-check-c4BKFlvl/maze-boxoban/puffer-throughput-uwk0wf_t` and `/tmp/puffer-hotspots-regression-check-c4BKFlvl/breakout-h/puffer-throughput-kpyhrbuu`.

### Muon copy-overlap result

The private-copy-stream experiment preserved every GEMM, copy, coefficient, and checkpoint bit, but it did not clear the performance gate. Over two Affine pairs, 578 was noisy at `1.008304x`, 1024x4 consistently regressed to `0.997829x`, and the suite result was only `1.003053x`. The added stream/events were reverted. Artifact: `/tmp/puffer-muon-copy-overlap-ab-I2JJdU62/puffer-throughput-7p1jziev`.

### Next direction

Further work should target arithmetic-preserving work removal rather than more dW scheduling. Trace parsing found a median final dW tail of only `9.920 us` on Affine 578 (about `0.79%` of iteration time) and `7.745 us` on 1024x4 (about `0.12%`). The previous dW was still running when the next layer became ready only once in 7,552 opportunities on 578 and never in 5,577 opportunities on 1024x4, so neither a second dW stream nor stream-priority tuning is justified. cuBLASLt separate-C/D and GEMM autotuning remain deferred because different kernels/reduction orders are not guaranteed bit-identical.

### MinGRU training-fusion ablations

Inter-layer backward-add fusion was also rejected. Instead of materializing each upper-layer `BF16_round(FP32(dX) + FP32(highway))`, the next lower scan reconstructed that exact rounded value from the two BF16 sources before its unchanged arithmetic. The bottom add remained materialized for encoder backward, and both ordinary and row scans were supported. The cleaned implementation was roughly 25-30 net lines. A three-pair run measured Affine 578 `1.008689x`, 1024x4 `1.001612x`, suite `1.005144x`, with exact checkpoints and worst pair `0.995648x`. All nine final golden checkpoints matched exactly, but the repeated Breakout-h performance gate measured `0.995334x` with a `0.960713x` worst pair. That is not enough gain or stability to justify the code. Affine artifact: `/tmp/puffer-grad-add-clean-ab-MZ967oyD/puffer-throughput-s2g89pvp`; golden artifacts: `/tmp/puffer-grad-add-clean-golden-ObwQjjBP/standard/puffer-throughput-2lplq623` and `/tmp/puffer-grad-add-clean-golden-ObwQjjBP/breakout/puffer-throughput-mxvbxckj`; repeat artifact: `/tmp/puffer-grad-add-clean-regression-tFpJLK6m/breakout-h/puffer-throughput-tswg4v27`.

An additional saved-input-copy fusion was rejected. It raw-copied the exact loaded `precision_t` input from each forward scan into the existing backward-save buffer, removing one serialized D2D node per layer while preserving allocator layout and bits. Combined with backward-add fusion it measured 578 `1.011174x` but 1024x4 `0.995644x`, including a `0.980087x` pair; the extra scan-store pressure outweighed the removed copy on the larger net. Artifact: `/tmp/puffer-scan-fusions-ab-HdiVwMZc/puffer-throughput-x68n0j3o`.

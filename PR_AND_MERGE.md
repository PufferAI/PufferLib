# Bat PR And Merge Notes

## Determinism Terms

- **Old-baseline trajectory equivalence**: same code/config/seed reproduces the
  current exact training trajectory, scalar signature, checkpoint behavior, and
  level-10 eval behavior.
- **Deterministic reproducibility**: same code/config/seed reproduces the same
  result after we intentionally change behavior.

For cleanup before the first merge, preserve old-baseline trajectory equivalence
unless we explicitly decide a change belongs in the new deterministic baseline.
Later behavior-breaking cleanups are allowed, but each one needs a fresh
reproducible training/eval signature.

## Most Embarrassing Review Targets

| Area | Why Joseph might call it out | Cleanup class |
| --- | --- | --- |
| `c_step()` terminal/reward flow | Over-budget chirp, collision, success, and timeout are detected in different branches, then partially consolidated later. It is correct enough, but less direct than Breakout/Boxoban/G2048. | Try to preserve old baseline first; larger reshaping may break it. |
| `compute_observations()` side effects | The name says observations, but it also consumes echo buckets and sets `tick_bug_echo_path`, which later affects reward. This is now direct in the function instead of hidden behind a one-use helper. | Preserve old baseline unless the echo reward order is deliberately changed. |
| `schedule_echo()` size | It mixes heading math, ear directivity, path/range checks, Doppler, attenuation, and queue writes in one function. | Preserve old baseline by extracting repeated left/right queueing only. |
| `reset_bug_motion()` and `update_bug()` | Three maneuver modes, inbound special cases, sign state, bounce repair, and multiple curriculum helpers are too much. A single sine wave with curriculum-ramped amplitude would be cleaner. | New deterministic baseline. This will likely break old behavior. |
| Spawn helpers | Exact-distance spawn and fallback quadrant spawn now live in one function, but the fallback loops remain defensive. | New deterministic baseline if RNG order changes. |
| Obstacle generation | `rects_overlap()`, `obstacle_clear()`, 96 attempts, and fallback placements are probably more safety than we need. | New deterministic baseline. Remove if overlapping random obstacles are acceptable. |
| Curriculum difficulty logs | `curriculum_distance_difficulty()`, `curriculum_obstacle_difficulty()`, `curriculum_motion_difficulty()`, and `curriculum_difficulty()` are a lot of code for diagnostics/objective shaping. | Likely behavior/metric breaking; do later. |
| Chirp efficiency / chirps-used logs | `chirps_used_ratio()` is still an observation, but reward/log helpers around sparse chirping are low conviction. | Keep observation if needed; remove reward/log parts in new baseline. |
| Demo defaults in `bat.c` | `set_demo_defaults()` duplicates `config/bat.ini`, which can drift. | Cleanup after deciding how the human demo path should load defaults. |
| Magic constants | Echo strengths, attenuation constants, spawn attempts, obstacle margins, first chirp defaults, and render colors are mostly unnamed. | Rename constants where it clarifies intent; avoid sweeping constant churn. |

## Preserve Old Baseline First

These changes should be attempted one at a time with the full gate:

1. Reduce duplicated left/right queueing in `schedule_echo()`.
2. Keep `compute_observations()` order stable. Echo bucket observation copying is
   now direct in the function.
3. Simplify local renderer helpers and repeated static reflector drawing.
4. Remove dead fields, dead constants, and obviously unreachable guards.
5. Keep `c_step()` reward order stable unless we deliberately decide to break
   old-baseline trajectory equivalence.

Gate after each code change:

```bash
source .venv/bin/activate && ./build.sh bat
source .venv/bin/activate && bash ocean/bat/tests/run_all.sh
.venv/bin/python -m pufferlib.pufferl train bat --train.gpus 1
timeout 45s env DISPLAY=:0 .venv/bin/python -m pufferlib.pufferl eval bat --load-model-path latest --env.curriculum-initial-level 10 --env.curriculum-successes-per-level 1000000
```

Known old-baseline training signature:

- `perf 0.556`
- `base_perf 0.950`
- `timeout 0.009`
- `chirps_emitted 5.191`

## Current Safe-Cleanup Notes

- Keep the normal `init()`, `allocate()`, `c_close()`, and `free_allocated()`
  shape. Breakout and G2048 use this pattern too, even when Bat currently has
  less heap-owned state after obstacle arrays became fixed-size.
- Breakout resets `terminals[0]` and `rewards[0]` at the top of `c_step()`;
  Boxoban increments `tick`, clears terminal/reward, then handles success and
  timeout as separate early-return branches. Bat's current terminal block is
  somewhat more abstract, but changing the reward/terminal order belongs in the
  new-baseline phase unless we intentionally stop matching the old run.
- Avoid changing reward arithmetic order in `c_step()` while preserving the old
  baseline. The one-line reward fold already proved it can break trajectory
  equivalence.
- `schedule_ear_echo()` is worth keeping for now. It is a small helper that
  removed duplicated left/right attenuation and receive-time logic.
- `norm_bin()` is small, but it names the action-bin normalization used by three
  chirp fields. Inlining it would save little and may make `try_emit_chirp()`
  less readable.
- `compute_observations()` now copies due echo buckets directly into
  observations and updates `tick_bug_echo_path` in place. Build, tests,
  training signature, and level-10 eval all preserved the old baseline.
- The fallback spawn cleanup removed the one-use `sample_spawns()` helper and
  kept the same RNG order inside `sample_spawns_at_distance()`. Build, tests,
  training signature, and level-10 eval all preserved the old baseline.
- Chirp slice scheduling no longer pre-fills every future source slot at emit
  time, and constructs the per-slice echo source explicitly instead of copying
  the whole `ChirpEvent`. Build, tests, training signature, and level-10 eval
  all preserved the old baseline.
- Expected bug echo timing now reads the just-emitted chirp source directly
  instead of asking for slice `0` before any slices are scheduled. Build, tests,
  training signature, and level-10 eval all preserved the old baseline.

## Next Old-Baseline Candidates

These are candidates only after the latest visual gate is confirmed:

1. Revisit any remaining one-use render helpers, but only if removal reduces
   lines without making `draw_freq_history_panel()` harder to scan.
2. Look for dead test-only exposure caused by removed helpers. The tests should
   assert behavior, not preserve helpers just because they were previously
   callable.
3. Review tiny math helpers one at a time. Keep helpers that name a real domain
   concept (`chirp_slice_ticks`, `chirp_age_norm_denominator`); consider
   inlining helpers that merely restate one field expression.
4. Leave `c_step()` structural reshaping for later. It is one of the highest
   review-value areas, but it is also one of the easiest ways to break old
   trajectory equivalence.

## New Deterministic Baseline Later

These are probably the real pre-merge quality wins, but they should be grouped
after we are ready to stop matching the current trajectory exactly:

1. Replace bug maneuver modes with one always-active sine-wave path and
   curriculum-ramped amplitude.
2. Simplify spawn and obstacle generation, including removing overlap checks if
   overlapping obstacles are acceptable.
3. Remove low-conviction curriculum difficulty and chirp efficiency logs/reward
   shaping.
4. Rework `c_step()` into a direct step, reward, done, log/reset shape matching
   the simpler reference envs.
5. Reconsider the initial chirp observation defaults instead of pretending a
   chirp happened before the episode starts.

# Bat Curriculum Strategy

This note explains how Bat curriculum should work, why `curriculum_perf` can
appear pinned around `0.2`, and how to extend difficulty toward maneuvering bugs
without adding artificial reward dials.

## Current Diagnosis

The old `curriculum_perf` behavior was mostly structural.

The old code computed:

```text
curriculum_difficulty = (distance_norm + obstacle_norm) / 2
curriculum_perf = success * curriculum_difficulty
```

With the current defaults:

```ini
curriculum_initial_level = 1
BAT_CURRICULUM_START_OBSTACLES = 0
BAT_CURRICULUM_MAX_OBSTACLES = 3
curriculum_obstacle_step = 8
curriculum_start_bug_distance = 8.438
BAT_CURRICULUM_MAX_BUG_DISTANCE = 40.0
BAT_CURRICULUM_BUG_DISTANCE_STEP = 2.0
BAT_CURRICULUM_INBOUND_MAX_BUG_DISTANCE = 56.0
BAT_CURRICULUM_INBOUND_BUG_DISTANCE_STEP = 4.0
```

At `curriculum_level ~= 5`, the bug starts around distance `28`, giving:

```text
distance_norm ~= (28 - 8) / (56 - 8) ~= 0.42
obstacle_norm = 0.0 because obstacle count is still 1
curriculum_difficulty ~= (0.42 + 0.0) / 2 ~= 0.21
```

So `curriculum_perf ~= 0.2` did not necessarily mean the policy hit an
impossible wall. It means the metric gives half of its difficulty credit to
obstacles, but obstacle difficulty stays zero until level `18`.

The current code logs split difficulty components and computes
`curriculum_difficulty` from active distance and obstacle components:

```text
distance_norm = normalize(start_bug_dist)
obstacle_norm = normalize(num_obstacles)
motion_norm = 0 until bug maneuvers are added

curriculum_difficulty =
    (0.50 * distance_norm +
     0.50 * obstacle_norm) / active_weight
```

Chirp-budget pressure is intentionally excluded from curriculum difficulty.
Motion difficulty is logged separately as `0`, but it does not lower the metric
ceiling before maneuver curricula exist.

## Design Principles

- Keep `base_perf` pure: `1.0` for catching the bug, `0.0` otherwise.
- Use composite `perf` for sweep ranking, but always sanity-check it against
  `base_perf`, `curriculum_level`, and failure modes.
- Curriculum should change task distribution, not secretly define behavior with
  dense reward shaping.
- Chirp timing pressure should come from physical constraints:
  finite budget, cooldown, and overlapping echo ambiguity.
- Movement pressure should come from task dynamics, not a hover exploit. The
  bat must keep a minimum forward speed so harder levels cannot collapse into
  stationary timeout policies.
- Do not add generic timing-efficiency reward.
- Difficulty metrics should be smooth enough that W&B does not show fake
  plateaus caused by integer schedule thresholds.
- Every curriculum rung should be reversible and testable.

## Recommended Metrics

Add or keep these logs:

```text
base_perf
perf
curriculum_perf
curriculum_level
curriculum_distance_difficulty
curriculum_obstacle_difficulty
curriculum_chirp_budget_difficulty (legacy fixed zero)
curriculum_difficulty
bug_motion_mode
bug_motion_speed
num_obstacles
chirp_budget
chirps_used_ratio
chirp_overlap_fraction
collision
timeout
SPS
```

Keep the W&B export list capped at 31 explicit metrics because PufferLib appends
`n` as the 32nd metric. Diagnostics such as `budget_difficulty`,
`curriculum_motion_difficulty`, chirp far/near fractions, and redundant inverse
ratios should stay internal unless they are needed for a specific sweep.
Do not remove `score` from `binding.c`; PufferLib's train worker reads
`metrics["env/score"]` when a child process exits. If the cap is tight, drop
`episode_return` instead.

The key change is splitting `curriculum_difficulty` into components. If
`curriculum_perf` is low, we should be able to tell whether the policy is stuck
on distance, obstacles, or later motion.

## Recommended Difficulty Formula

Use an explicit weighted difficulty. Until motion curriculum exists, renormalize
over active components:

```text
distance_norm = normalize(start_bug_dist)
obstacle_norm = normalize(num_obstacles)
motion_norm = normalize(bug maneuver difficulty)

curriculum_difficulty =
    0.50 * distance_norm +
    0.50 * obstacle_norm

curriculum_difficulty /= active_weight

curriculum_perf = base_perf * curriculum_difficulty
```

The exact weights can change, but the important property is that no inactive
component silently cuts the maximum metric in half. If `motion_norm` is not
enabled yet, either log it as `0` and accept the lower ceiling, or renormalize
active components. For sweeps, renormalizing active components is easier to
interpret.

## Recommended Stage Order

### Stage 0: Known-good baseline

Purpose:

- Preserve the current solved rung as a fallback.

Task:

- No obstacles at level 0.
- Moderate starting bug distance.
- Current forward-only bat dynamics.
- Configurable minimum forward speed; brake cannot stop the bat below this
  floor.
- Current chirp budget and overlap penalty.

Gate:

- `base_perf >= 0.80`
- collision not exploding
- timeout not dominating

### Stage 1: Distance curriculum

Purpose:

- Make the bat solve larger spatial uncertainty before adding more clutter.

Schedule:

```text
start_bug_distance = 8 + level * distance_step
```

Recommendation:

- Keep `BAT_CURRICULUM_BUG_DISTANCE_STEP` hardcoded at `2.0`; Bat9's best runs
  clustered there, and the inbound curriculum already expands later distances.
- Log `curriculum_distance_difficulty` directly.

Gate:

- advance after a small success count, not one lucky catch.
- current `curriculum_successes_per_level = 21` is conservative; for 50M-step
  runs, sweep lower values such as `4..16`.

### Stage 2: Obstacle curriculum

Purpose:

- Avoid the current metric plateau where obstacle difficulty is invisible until
  level `18`.

Recommendation:

- Reduce default `curriculum_obstacle_step`.
- The current default is `4`, with only level 0 obstacle-free:

```text
level 0:     0 obstacles
level 1..4:  1 obstacle
level 5..8:  2 obstacles
level 9+:    3 obstacles
```

Alternative:

- Keep the count schedule but add obstacle size or reflector strength as a
  smoother sub-rung.

Gate:

- Do not reduce chirp budget as obstacle count rises; clutter legitimately
  requires reacquisition chirps.

### Stage 3: Chirp budget curriculum

Purpose:

- Force useful chirp timing without artificial timing reward.

Current behavior:

- Observation includes `chirps_used / chirp_budget`.
- Chirping after the last chirp causes `-1` terminal.
- Budget is fixed across curriculum levels.
- Valid chirps before the previous max echo window clears get a physical
  overlap penalty.

Recommendation:

- Keep terminal/logging pressure as the primary signal:

```text
chirp_perf = clamp(1.0 - chirps_emitted / 15.0, 0.05, 1.0)
perf = base_perf * curriculum_difficulty * chirp_perf
```

- Keep `chirp_overlap_penalty` small and sweepable.
- Treat `chirp_overlap_fraction` as a diagnostic, not the main objective.
- Keep `budget_difficulty` and `chirp_efficiency` as diagnostics, but do not
  multiply them into `perf`; the fixed 15-chirp reference gives cleaner Protein
  ranking pressure across 10, 8, and 6 chirp policies.

### Stage 4: Constant-velocity moving bug

Purpose:

- Make Doppler and reacquisition matter while keeping motion predictable.

Current/near-term model:

- Fixed speed and heading.
- Wall bounce:
  - vertical wall flips `vx`
  - horizontal wall flips `vy`

Recommended knobs:

```text
bug_speed
bug_wall_bounce_enabled
```

Gate:

- Require maintained `base_perf` and non-collapsing `chirps_used_ratio`.
- Motion should not start before distance and obstacle rungs are stable.

### Stage 5: Simple bug maneuvers

Purpose:

- Add nontrivial pursuit without jumping directly to adversarial behavior.

Recommended maneuver order:

1. Sine lateral motion
   - Bug keeps forward velocity but adds low-frequency lateral acceleration.
   - Knobs: `bug_maneuver_amplitude`, `bug_maneuver_period`.

2. Smooth heading drift
   - Bug heading changes slowly with bounded turn rate.
   - Knobs: `bug_turn_rate`, `bug_turn_period`.

3. Circular or oval path segments
   - Bug follows simple parametric curves.
   - Knobs: `bug_orbit_radius`, `bug_orbit_period`.

4. Piecewise constant heading changes
   - Bug chooses a new heading every N ticks.
   - Knobs: `bug_heading_change_interval`, `bug_heading_change_angle`.

5. Mild evasive steering
   - Bug slowly biases away from bat only at higher curriculum.
   - This should come late because it changes the task from tracking to pursuit.

Do not expose bug mode, bug position, or true velocity in observations. The bat
should infer motion from echoes.

## Proposed Curriculum State

Keep the current single integer `curriculum_level`, but derive separate
sub-difficulties from it:

```text
distance_level = level
obstacle_level = max(0, level - obstacle_start_level)
budget_level = max(0, level - budget_start_level)
motion_level = max(0, level - motion_start_level)
```

Recommended starting points:

```ini
obstacle_start_level = 4
budget_start_level = 0
motion_start_level = 10
maneuver_start_level = 16
```

This keeps one scalar progression while preventing all difficulty knobs from
turning on at once.

## Eval Requirements

Eval should support fixed curriculum levels so we can inspect specific rungs.

Useful modes:

```text
default latest curriculum level
fixed level 0
fixed level 6
fixed level 12
fixed level 18
fixed maneuver mode
```

This matters because an aggregate training metric can look fine while a
specific rung has bad behavior.

## TDD Targets

Add focused tests before changing curriculum code:

```text
curriculum_difficulty_logs_distance_and_obstacle_components
curriculum_obstacles_advance_before_level_18
curriculum_budget_reduces_monotonically_with_level
curriculum_motion_stays_zero_before_motion_start_level
bug_wall_bounce_flips_x_or_y_velocity
bug_sine_maneuver_keeps_speed_bounded
eval_fixed_curriculum_level_overrides_training_level
```

## Near-Term Recommendation

Before adding maneuvers, do this:

1. Add split difficulty logs:
   - `curriculum_distance_difficulty`
   - `curriculum_obstacle_difficulty`
   - `curriculum_chirp_budget_difficulty`
   - `curriculum_motion_difficulty`

2. Change obstacle schedule so it starts contributing around level `6`, not
   level `18`.

3. Consider reducing `curriculum_successes_per_level` default from `21` to a
   faster value such as `8..12`, while keeping it sweepable.

4. Add fixed-level eval override so visual checks can inspect harder rungs
   directly.

5. Only then add simple bug maneuvers, starting with constant velocity wall
   bounce and then sine/heading drift.

The goal is a ladder where each rung is visibly harder, metrics explain why,
and the bat must improve sensing behavior without reward terms that directly
script the desired chirp timing.

## Current Curriculum Cleanup

Chirp-budget pressure is no longer mixed into curriculum difficulty.

Rationale:

- More obstacles legitimately require more chirps. A cluttered arena should not
  rank worse just because the bat used more chirps than it would in open space.
- Chirp count is a sensing-efficiency metric, not a world-difficulty metric.
- A shrinking chirp budget can make later curriculum levels impossible before
  we know whether the policy has learned robust obstacle disambiguation.

Current curriculum split:

```text
level 0:
  no obstacles
  moving bug only

later levels:
  increase bug start distance
  introduce the first obstacle immediately at level 1
  increase obstacle count/clutter every few levels
  then add maneuvering bug motion
```

Current curriculum difficulty:

```text
curriculum_difficulty =
    0.5 * distance_difficulty +
    0.5 * obstacle_difficulty
```

If a component has not been activated yet, renormalize over active components
instead of letting inactive components cap the score. Once obstacle curriculum
is active, the two-component `0.5 / 0.5` interpretation is easy to explain:
half distance, half clutter.

Current chirp handling:

```text
BAT_MAX_CHIRPS_PER_EPISODE = 15
chirp_budget does not decrease with curriculum level
chirp_budget_difficulty is removed from curriculum difficulty
```

Reward/perf pressure for chirps should focus on intelligent use, not simply
fewer chirps everywhere:

- Keep a finite chirp budget so chirps are not unlimited.
- Keep overlap penalty because overlapping echo returns are physically
  ambiguous.
- Consider rewarding successful catches with a chirp-use bonus based on
  `chirps_emitted / 15`, but avoid dense shaping that scripts exact chirp
  timing.
- Keep `chirp_perf = clamp(1.0 - chirps_emitted / 15.0, 0.05, 1.0)` as a sweep
  diagnostic/objective term, but interpret it together with obstacle difficulty
  and not as an absolute "fewer chirps is always better" rule.

This cleanup changes how `perf` compares to older Bat sweep runs. Compare old
and new runs through component logs (`base_perf`, `curriculum_perf`,
`chirp_perf`) when needed.

## Bat3 Partial Sweep Notes

These notes are from an in-progress `bat3` W&B peek on June 9, 2026. Treat them
as directional, not final.

Early top-`perf` runs show:

- `bat_min_speed` tends toward the low end, usually near `2.0`.
- `bat_turn_rate` tends toward the high end, often near `3pi`.
- `sound_speed` tends toward the high end, often `175..180`.
- `progress_reward_scale` tends toward the high end, around `0.11..0.12`.
- Good policies often catch with roughly `6..8` chirps.
- Highest `base_perf` runs can exceed `0.90`, but may use more chirps and rank
  lower by `perf`.
- Highest `curriculum_perf` runs reach around levels `8..9`, but often pay with
  higher collision/timeout and around `10` chirps.

Behavior read:

- The current interesting behavior is circle-search followed by full-speed
  dash/intercept after apparent target acquisition.
- Do not remove this behavior unless harder fixed-level evals prove it is an
  exploit. It may be a useful active-sensing search pattern.

Metric implication:

- `chirp_perf` is working as a sweep ranking term, but it also confirms that
  "fewer chirps" cannot be the whole story once obstacles increase.
- More clutter can legitimately require more chirps, so chirp count should stay
  separate from world/curriculum difficulty.

Post-sweep PR gate:

1. Update `config/bat.ini` defaults from the best sane run, not merely highest
   `perf`.
2. Run one normal training pass with those defaults.
3. Eval fixed levels `0`, `4`, `7`, and `10`.
4. Commit only if visual behavior remains sane and the policy does not regress
   to hover/spin/collision farming.

Visual eval diagnosis from `fresh-wood-149`:

- Default/easier eval looks good: circle-search followed by dash/intercept.
- Fixed level `10` performs poorly.
- Fixed level `7` reveals the likely failure mode:
  - the bat spends many chirps during early search before acquiring the bug,
  - once it finally gets a useful bug signal, the remaining chirp budget is low,
  - it dashes toward the last known/acquired bug direction,
  - if the bug moves enough after the final echoes, the bat keeps flying blind
    and misses.

Implication:

- The next bottleneck is acquisition/reacquisition under finite chirp budget,
  not basic forward flight.
- Be careful with any metric or reward that simply minimizes chirp count. At
  harder distances or with obstacles, useful policies may need more search and
  reacquisition chirps.
- This supports the current cleanup: keep a fixed chirp budget for now and keep
  chirp pressure separate from curriculum difficulty before adding harder
  motion or more clutter.

## Reward-Shaping Guardrails

Bug-echo progress shaping is allowed, but it must not pay for passive target
motion. Compare the current bug echo path against the previous bug echo only
after the bat has displaced by at least `bug_echo_min_displacement`. If the echo
path is shorter, reward by `bug_echo_reward_scale`; if it is longer, apply a
weaker penalty using `bug_echo_farther_penalty_scale`, currently defaulting to
`0.10`.

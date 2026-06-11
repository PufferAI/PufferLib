# Bat Environment Spec

Status: draft baseline; ready for implementation planning after review

Workspace: `/home/claude/pathfinder`

Target branch: `bat`

Target env name: `bat`

Detailed sonar observation design note:

- `BAT_SONAR_OBSERVATION_NOTES.md`

## Intent

Build a single-agent PufferLib Ocean environment inspired by bat echolocation.
The agent controls a bat flying in a 2D arena with walls, static obstacles, and
a moving bug target. The bat must avoid collisions and catch the bug using
binaural acoustic returns from self-generated chirps rather than direct map or
position observations.

The first version should copy the small native-C env style used by Breakout:
fixed-size observations, a compact action space, simple deterministic physics,
and enough instrumentation to make training failures debuggable.

The core challenge is active sensing. The policy must learn both how to move
and how to emit useful chirps. The environment should make chirping meaningful
without turning v1 into a full acoustic wave simulator.

## Research Grounding

Range cue:

- Echolocating bats primarily estimate target distance from the delay between
  an emitted call and the returning echo.
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC9157489/
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC7888678/

Binaural direction cue:

- Left/right ear differences are biologically plausible and useful. Bats use
  binaural and spectral cues, including head-related transfer effects, to infer
  sound direction.
- Source: https://pubmed.ncbi.nlm.nih.gov/15658710/
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC4555857/

Chirp design:

- Linear frequency-modulated chirps are standard in radar and sonar because
  matched filtering can compress a long emitted pulse into a sharp return peak.
  Bandwidth controls range resolution, and the time-bandwidth product controls
  processing gain.
- Source: https://rfessentials.com/rf-knowledge-base/how-does-pulse-compression-improve-the-range-resolution-and-sensitivity-of-a-rad/

Doppler:

- Doppler shift is a useful velocity cue, especially for moving targets and
  insect-like prey. Some bat species actively compensate call frequency to keep
  important echo bands in a sensitive range.
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC2438418/
- Source: https://www.nature.com/articles/s41598-018-22880-y

Fast signal processing:

- FFTW is the high-performance reference point for C FFT design, but v1 should
  avoid adding FFTW as a dependency. A fixed-size radix-2 FFT or precomputed
  analytic matched-filter bins are preferred.
- Source: https://www.fftw.org/fftw2_doc/fftw_1.html
- Source: https://web.stanford.edu/class/cme324/classics/cooley-tukey.pdf

Reflection model:

- Full wave acoustics is out of scope for v1. A geometric echo model is the
  right first approximation: sound travels in straight paths, reflects from
  objects, and returns with delay, angle-dependent ear gain, attenuation, and
  optional Doppler.
- Source: https://au.mathworks.com/help/audio/ug/room-impulse-response-simulation-with-image-source-method-and-hrtf-interpolation.html

## Environment Model

World:

- 2D continuous rectangular arena.
- Arena dimensions are fixed by config.
- Boundaries are solid walls.
- Static obstacles are axis-aligned rectangles.
- The bug is a moving circular target.
- The bat is a moving circular agent with heading, speed, turn rate, and
  collision radius.

Physics:

- Fixed control/physics timestep, default `1/60` second.
- Bat motion is acceleration-limited and turn-rate-limited.
- The bat has a configurable minimum forward speed. It cannot hover; brake
  only reduces speed down to this stall-speed floor.
- Bug motion uses a simple deterministic or seeded random policy.
- The bug reflects from walls and obstacles.
- The bat collides with walls and obstacles.
- Catch success occurs when bat and bug circles overlap.
- The `1/60` second tick is not the acoustic sample rate. Echo delays are
  computed analytically with fractional timing inside each env step.

Acoustics:

- Walls, obstacles, and the bug reflect chirps.
- Static reflectors provide range and direction cues.
- The bug is the only moving reflector, so it is the main Doppler source.
- The env computes compact acoustic features analytically instead of storing or
  convolving high-rate audio samples.
- Sound speed is configurable and artificial. The default should be much slower
  than real air acoustics so echo timing is learnable in a small game arena.
- Start with `sound_speed = 60.0` world units per second. At the default
  `1/60` second env tick and current ear spacing, this gives broadside echoes
  enough artificial time-of-arrival separation for one ear to be able to hear a
  return about one tick before the other.
- `ear_separation_scale` controls the artificial distance between ears as a
  multiple of `BAT_RADIUS`. Keep it bounded; the implementation clamps it to
  `[0.25, 2.0]` and the default sweep range is `[0.5, 2.0]`.
- Every echo contribution has:
  - two-way distance from mouth/source to reflector to each ear,
  - delay derived from speed of sound,
  - amplitude falloff from distance and reflector strength,
  - left/right ear gain from relative azimuth,
  - Doppler shift from reflector radial velocity.

Point-reflector renderer:

- v1 should represent walls and obstacle surfaces as stationary point
  reflectors.
- Sample each wall and obstacle edge at fixed spacing,
  `BAT_REFLECTOR_SPACING = 8.0` world units.
- The bug contributes one moving circular/point reflector at its center.
- This avoids wavefront bookkeeping while preserving range, angle, and Doppler
  learning signals.

First-order echoes only:

- v1 should include direct echo paths from visible surfaces and the bug.
- Multiple-bounce reverberation is out of scope for v1.
- Occlusion can be approximated by ray intersection against the nearest
  obstacle along the bat-to-reflector path.
- Segment-level specular reflection and raw waveform propagation are later
  variants, not the v1 baseline.

## Chirp Model

The policy controls chirp parameters rather than emitting arbitrary audio.

Chirp parameters:

- `chirp_start_freq`
- `chirp_end_freq`
- `chirp_duration`

Derived fields:

- `chirp_bandwidth = abs(chirp_end_freq - chirp_start_freq)`
- `chirp_slope = (chirp_end_freq - chirp_start_freq) / chirp_duration`
- `chirp_age_ticks = ticks since most recent emitted chirp`

Defaults:

- Frequency range is normalized in the policy/action interface and mapped to a
  narrow ultrasonic band in the env.
- Duration is normalized in the policy/action interface and mapped to a small
  tick/subtick window.
- Up-chirps and down-chirps are both legal.
- A zero-amplitude/no-chirp action should be available so the bat is not forced
  to emit every tick.

Implementation direction:

- Start with analytic range/Doppler bins, not literal audio buffers.
- If an FFT is needed, use a fixed power-of-two size with precomputed twiddle
  factors.
- Prefer precomputed chirp templates or direct bin accumulation for v1 because
  this env will run thousands of agents in parallel.
- The v1 observation bins are not raw FFT bins. They are compact
  matched-filter-like echo features derived from chirp parameters, delay,
  amplitude, and normalized Doppler.
- RayLib eval rendering may play an audible debug version of emitted chirps.
  This is render-only and must not run in headless training. The audible sound
  maps the normalized chirp band to a human-hearable swept sine while preserving
  the selected start frequency, end frequency, and duration.
- RayLib eval rendering also supports `env.render_target_fps`; default `60`,
  and `0` leaves RayLib uncapped. This is for visualization/audio inspection
  only and should not be used in training or sweep interpretation. Audible
  debug chirp duration scales as `max(1, 60 / render_target_fps)` so low-FPS
  inspection preserves chirp ordering while making each sweep easier to hear.

## Action Space

Use a small multi-discrete action space.

Recommended v1 action heads:

- `move`: 3 values
  - `0`: no thrust
  - `1`: thrust forward
  - `2`: brake / reduce forward speed
- `turn`: 3 values
  - `0`: no turn
  - `1`: turn left
  - `2`: turn right
- `chirp_start_freq`: discrete bins, default `8`
- `chirp_end_freq`: discrete bins, default `8`
- `chirp_duration`: discrete bins, default `4`
- `chirp_emit`: 2 values
  - `0`: do not emit a chirp this tick
  - `1`: emit chirp using selected chirp parameters

Initial action sizes:

- `ACT_SIZES {3, 3, 8, 8, 4, 2}`
- `NUM_ATNS 6`

Rationale:

- Multi-discrete actions let the agent combine flight and active sensing.
- Discrete chirp bins keep the policy simple and cheap.
- Bat movement is scalar forward speed plus heading. The velocity vector is
  recomputed as `heading * speed` every tick.
- Brake clamps speed at `bat_min_speed`. The bat cannot fly backward or hover.
- Strafe/lateral velocity is intentionally unavailable. This avoids sideways
  spiral policies and makes the visual behavior match the game fantasy better
  than a full inertial top-down spacecraft model.
- Continuous actions can be a later variant after the first training baseline
  is understood.

## Observation Space

Do not expose absolute position, absolute bug position, obstacle map, or global
heading.

Observation layout:

1. `left_freq_bins[16]`
2. `right_freq_bins[16]`
3. `chirp_age_norm`
4. `chirp_cooldown_norm`
5. `last_chirp_start_freq_norm`
6. `last_chirp_end_freq_norm`
7. `last_chirp_duration_norm`
8. `chirps_used_norm = chirps_used / chirp_budget`
9. `forward_speed_norm`
10. `turn_rate_norm`
11. `timer_norm = elapsed_steps / BAT_MAX_STEPS`, clamped to `[0, 1]`

Initial observation size:

- `OBS_SIZE = 41`

Timer normalization:

- The timer starts at `0.0` on reset.
- With `BAT_MAX_STEPS = 512`, after step `N` the observation is
  `N / 512.0`.
- The observed timer is clamped to `[0.0, 1.0]`.

Echo bins:

- Each ear receives 16 frequency-intensity bins.
- Bins represent the summed intensity arriving at that ear during the current
  env tick.
- Values are capped to `[0.0, 1.0]` before policy input.
- No explicit delay/range bins are exposed.
- No chirp means no new echo energy, aside from any later noise model.
- Range must be inferred from when frequency energy returns after an emitted
  chirp.
- Doppler shifts move return energy across nearby frequency bins instead of
  appearing in a separate Doppler observation channel.

Echo timing:

- Chirps schedule analytic echo-arrival events.
- Each event has a receive time, ear, normalized frequency, and intensity.
- On each tick, all events arriving in that tick window are summed into the
  corresponding ear frequency bins.
- Multiple reflectors can contribute to the same bin on the same tick.
- Echoes beyond `BAT_MAX_ECHO_RANGE` are ignored.
- Implementation should use a fixed future-tick accumulator, not a full active
  event scan every env step. The current design buckets each echo by
  `ceil(receive_tick)` into `BAT_ECHO_QUEUE_TICKS = 256`, sums by
  `[ear][freq_bin]`, and processes only the current tick's bucket.
- The accumulator is an implementation detail only. It must preserve the
  observation semantics: current-tick per-ear frequency intensities are summed
  and capped to `[0.0, 1.0]`; no range/delay axis is exposed.

Chirp metadata:

- The agent receives the last emitted chirp start frequency, end frequency, and
  duration because interpreting a return depends on knowing the transmitted
  signal.

Current implementation note:

- The range/Doppler scaffold has been retired in favor of per-tick left/right
  frequency spectra generated by analytic echo-arrival events.
- Range is inferred from echo timing and chirp age rather than exposed as an
  observation axis.
- See `BAT_SONAR_OBSERVATION_NOTES.md` before changing acoustic observations.
- `chirp_age_norm` lets the policy distinguish fresh echo windows from stale or
  silent intervals.

Self-motion:

- `forward_speed_norm` and `turn_rate_norm` are proprioceptive signals.
- `forward_speed_norm` is normalized scalar speed and should stay in `[0, 1]`.
- These do not reveal map coordinates or target location.
- They reduce unnecessary burden on recurrent policy memory.

Model memory note:

- PufferLib has recurrent policy support through `MinGRU`, `GRU`, and `LSTM`.
- The default config currently uses `MinGRU`, but v1 should not require the
  policy to remember chirp identity just to interpret the current acoustic
  observation.

## Reward and Termination

Reward shaping is intentionally simple in v1. It should make pursuit learnable
without leaking any privileged information through observations.

Default reward model:

- `+1.0` for catching the bug.
- Small negative step cost to encourage efficient pursuit.
- Dense progress reward based on reduction in true bat-to-bug distance.
- `-1.0` for hitting walls or obstacles, terminal.
- `-1.0` for timeout, terminal.
- `-1.0` for attempting a chirp after `chirps_used_norm` reaches `1.0`,
  terminal.
- Tiny chirp cost so constant chirping is not fully free without causing
  chirp collapse.
- Chirping again before the prior chirp's expected bug reflection has returned
  gets a small physical overlap penalty. This is not a generic timing-efficiency
  reward; it represents self-induced acoustic ambiguity from overlapping bug
  returns without forcing the bat to wait for every static wall or obstacle
  reflection.
- Solve-time chirp efficiency reward:
  - `chirp_efficiency = 0.5 + 0.5 * (1.0 - chirps_used / chirp_budget)`,
  - a catch after spending the full budget gets efficiency `0.5`,
  - a catch with very few chirps approaches efficiency `1.0`,
  - `chirp_efficiency_reward` scales this bonus and should be sweepable.
- Sound-derived bug echo progress reward:
  - when a bug echo returns with a shorter acoustic path than the previous bug
    echo, add a small shaped reward,
  - this reward only applies if the bat has moved at least
    `BAT_BUG_ECHO_MIN_DISPLACEMENT` since the previous scored bug echo, so a
    stationary bat cannot farm reward from the bug moving closer by itself,
  - farther bug echoes update the previous bug echo path and receive a weaker
    penalty scaled by `bug_echo_farther_penalty_scale`, default `0.10`,
  - static wall and obstacle echoes do not receive this reward.
- Optional silence bonus or energy budget should wait until the basic task
  trains.

Progress reward:

- Track previous true bat-to-bug distance internally.
- Reward positive distance reduction.
- Penalize distance increase by the same or smaller scale.
- Do not expose the true distance in observations.
- Default formula:
  - `reward += progress_reward_scale * (prev_bug_dist - bug_dist)`
  - `reward -= step_cost`
  - `reward -= BAT_CHIRP_COST` when a chirp is emitted; this is hardcoded to
    zero for the current Bat defaults
  - `reward -= chirp_overlap_penalty * bug_echo_wait_fraction` when a valid
    chirp is emitted before the previous chirp's expected bug reflection has
    returned
  - `reward += chirp_efficiency_reward * chirp_efficiency` on catch
  - `reward += bug_echo_reward_scale * echo_path_reduction / BAT_MAX_ECHO_RANGE`
    when a returning bug echo indicates the bug is closer than the previous bug
    echo and the bat has moved enough since that previous echo
  - `reward -= bug_echo_reward_scale * bug_echo_farther_penalty_scale *
    echo_path_increase / BAT_MAX_ECHO_RANGE` when a later moved-enough bug echo is
    farther away
- Default starting values:
  - `progress_reward_scale = 0.05`
  - `step_cost = 0.001`
  - `chirp_efficiency_reward = 1.0`
  - `chirp_overlap_penalty = 0.004`
  - `bug_echo_reward_scale = 0.02`

Important caveat:

- Dense distance reward is privileged training signal. It is acceptable for v1
  if the goal is to get learning started, but it should be easy to disable or
  scale down once the acoustic policy learns basic pursuit.

Termination:

- Success: bat catches bug.
- Failure: bat collides with a wall or obstacle.
- Failure: bat attempts to chirp after exhausting the chirp budget.
- Timeout: `tick >= BAT_MAX_STEPS`.

Reset:

- New episode samples arena layout, bat spawn, bug spawn, and bug velocity.
- Bat and bug should not spawn overlapping obstacles or each other.
- Initial bug distance should support curriculum.

W&B exported metrics:

- Export at most 31 explicit `dict_set(out, ...)` metrics from `binding.c`.
  PufferLib appends `n`, giving the 32-key cap. Keep lower-value diagnostics
  internal unless they are actively needed for sweep decisions.

- `perf`
  - composite sweep objective:
    `base_perf * curriculum_difficulty * chirp_perf`
- `base_perf`
  - pure catch rate: `1.0` for catching the bug, `0.0` otherwise
- `curriculum_level`
- `curriculum_difficulty`
  - weighted normalized episode difficulty from split curriculum components
- `curriculum_perf`
  - `base_perf * curriculum_difficulty`; useful diagnostic for level progress
    without chirp-budget weighting
- `curriculum_distance_difficulty`
- `curriculum_obstacle_difficulty`
- `curriculum_chirp_budget_difficulty`
  - legacy diagnostic; fixed at `0.0` because chirp budget no longer decays
    with curriculum
- `score`
  - required by PufferLib train worker; do not remove from `binding.c`
- `episode_length`
- `success`
- `collision`
- `timeout`
- `bug_distance_start`
- `bug_distance_final`
- `bug_distance_delta`
- `num_obstacles`
- `chirps_emitted`
- `chirp_budget`
- `chirps_used_ratio`
- `chirp_efficiency`
  - `0.5` if the full budget was spent, approaching `1.0` when few chirps were
    used
- `chirp_perf`
  - sweep-objective chirp multiplier:
    `clamp(1.0 - chirps_emitted / 15.0, 0.05, 1.0)`
  - this uses a fixed 15-chirp reference instead of the current per-level
    budget so 6-chirp and 8-chirp policies remain meaningfully separated
- `chirp_overlap_fraction`
  - fraction of emitted chirps that were sent before the previous chirp's max
    return window cleared
- `far_chirp_rate`
- `near_chirp_rate`
- `chirp_tempo_ratio`
  - `near_chirp_rate / far_chirp_rate`, clamped to `[0, 10]`; values above
    `1.0` indicate chirps are denser near the bug than far away
- `first_chirp_tick_norm`
- `mean_chirp_tick_norm`
- `mean_chirp_duration`
- `mean_chirp_bandwidth`
- `mean_echo_energy_left`
- `mean_echo_energy_right`
- `n`

## Curriculum

The first curriculum should keep obstacles present but make target behavior
simple before adding maneuvering.

Recommended stages:

- Stage 0: fixed arena, boundary walls, simple fixed obstacles, slow bug with
  fixed velocity and bounce behavior.
- Stage 1: same layout class, faster bug with fixed velocity and bounce
  behavior.
- Stage 2: randomized obstacles, slow bug with fixed velocity and bounce
  behavior.
- Stage 3: randomized obstacles, faster bug with small seeded random turns.
- Stage 4: randomized obstacles, faster bug that can maneuver or flee.
- Stage 5: lower progress reward scale and higher chirp cost.

Config knobs:

- `obstacle_min_size`
- `obstacle_max_size`
- `ear_separation_scale`
- `bat_max_speed`
- `bat_min_speed`
- `bat_accel`
- `bat_turn_rate`
- `sound_speed`
- `chirp_cooldown_ticks`
- `chirp_freq_bins`
- `chirp_duration_bins`
- `chirp_efficiency_reward`
- `chirp_overlap_penalty`
- `bug_echo_farther_penalty_scale`
- `step_cost`
- `progress_reward_scale`
- `collision_penalty`
- `curriculum_initial_level`
- `curriculum_stage`

## PufferLib Integration

Expected files after spec approval:

- `ocean/bat/bat.h`
- `ocean/bat/bat.c`
- `ocean/bat/binding.c`
- `ocean/bat/tests/`
- `config/bat.ini`

Follow the Breakout-style native env shape:

- Define `Log`.
- Define env struct `Bat`.
- Store required pointers:
  - `float* observations`
  - `float* actions`
  - `float* rewards`
  - `float* terminals`
  - `int num_agents`
  - `Log log`
  - `unsigned int rng`
- In `binding.c`, start with:
  - `OBS_SIZE 41`
  - `NUM_ATNS 6`
  - `ACT_SIZES {5, 3, 8, 8, 4, 2}`
  - `OBS_TENSOR_T FloatTensor`
  - `Env Bat`

Testing expectations:

- Unit tests for chirp parameter normalization.
- Unit tests for audible chirp waveform helper math. Rendering playback itself
  stays a RayLib eval concern, not a training dependency.
- Unit tests for echo delay and per-tick frequency-bin placement.
- Unit tests for left/right ear asymmetry from azimuth.
- Unit tests for Doppler sign on approaching vs receding bug.
- Unit tests for collision and catch termination.
- Unit tests for progress reward sign.
- Unit tests that wall collision returns `-1.0` and terminates.
- Unit tests that obstacle reflectors create boundary-approach signals.

## Open Design Questions

Reward shaping:

- The first implementation should use the default shaping constants above.
- After the first trainability pass, decide whether to clip progress reward,
  anneal privileged progress reward down, or increase chirp cost.

Acoustic representation:

- v1 uses 16 current-tick frequency-intensity bins per ear.
- A later variant can test more frequency bins, a flattened range-Doppler grid,
  or literal FFT bins.

Bug behavior:

- v1 starts with fixed-velocity bounce behavior.
- Later curriculum stages add seeded random turns and maneuvering.

Obstacle reflections:

- v1 samples walls and obstacle edges into point reflectors.
- Later variants can compare analytic segment reflections or multiple-bounce
  reflections.

## Training and Sweep Operations

- Curriculum design notes are tracked in `BAT_CURRICULUM.md`. Keep that file
  updated when changing level progression, difficulty metrics, or bug motion
  rungs.
- Current curriculum cleanup is documented in `BAT_CURRICULUM.md`: level 0
  starts with no obstacles, chirp-budget pressure is separate from curriculum
  difficulty, and curriculum difficulty uses distance/obstacles only.
- Keep `base_perf` as pure catch rate. Use composite `perf` as the sweep
  objective. It rewards catching harder curriculum levels with fewer chirps
  without changing in-episode reward shaping:
  `perf = base_perf * curriculum_difficulty * chirp_perf`.
- `chirp_perf` uses a fixed 15-chirp reference:
  `clamp(1.0 - chirps_emitted / 15.0, 0.05, 1.0)`. This intentionally gives
  strong sweep-ranking separation between 10, 8, and 6 chirps. Do not multiply
  `perf` by both `budget_difficulty` and `chirp_efficiency`; that made the
  metric harder to reason about and double-counted chirp pressure.
- Keep `score` exported in `binding.c`. If the 31-metric cap is tight, drop
  `episode_return` before dropping `score`; PufferLib reads `metrics["env/score"]`
  when train workers finish.
- Reward terms are training scaffolding and should remain sweepable. `progress_reward_scale` is true-distance shaping and should usually stay below `bug_echo_reward_scale`, which is based on closer received bug reflections.
- Forward-only movement dynamics should be swept with bounded ranges:
  `env.bat_max_speed` in `[8.0, 22.0]`, `env.bat_min_speed` in `[2.0, 6.0]`,
  `env.bat_accel` in `[40.0, 90.0]`, and `env.bat_turn_rate` in `[4.0, 3pi]`.
- Do not remove the minimum forward speed invariant. If the bat can hover at
  zero velocity, PPO can learn a bad local optimum where it avoids collision
  and timeout-shapes instead of exploring movement.
- Bug-echo progress shaping must be gated on bat displacement. Closer bug
  echoes can reward, and farther bug echoes can weakly penalize, but neither
  should pay out when the bat has not moved enough since the prior bug echo.
- Acoustic scale terms should be swept before increasing model size. Current bounded acoustic sweep knobs are `env.sound_speed` in `[80.0, 180.0]` and `env.ear_separation_scale` in `[1.0, 3.0]`.
- The June 9, 2026 `bat1` sweep strongly improved after the forward-only
  dynamics change. Best observed run was `sage-cherry-92` with `perf ~= 0.953`,
  `SPS ~= 2.06M`, collision `~= 0.031`, and timeout `~= 0.016`. The old default
  had higher SPS but poor `perf`, so use `perf` first and SPS only as a
  tie-breaker.
- That sweep pushed several bounds upward: `bat_accel`, `bat_turn_rate`,
  `sound_speed`, `ear_separation_scale`, `progress_reward_scale`,
  `replay_ratio`, and often `ent_coef`. It pushed `step_cost` and
  `valid_chirp_reward` down. Defaults in `config/bat.ini` now track the best
  high-perf region rather than the highest-SPS failed default.
- Train workers should use CUDA with `--train.gpus 1`.
- Protein/sweep control does not need CUDA. Run sweeps with `--sweep.use-gpu ""` so the optimizer stays off CUDA and avoids CUDA IPC/resource-handle failures.
- Do not override training duration with ad hoc `--train.total-timesteps`. Put duration ranges in `config/bat.ini`.
- Keep Bat sweep ranges bounded so a sweep cannot accidentally launch huge slow models.
- Do not use `sweep_only` in Bat config. Keep the config clean and bound the
  actual sweep sections/defaults instead.
- The default Bat sweep does not sweep policy model size; it keeps `policy.hidden_size = 128` and `policy.num_layers = 4`. Current cost-sensitive sweep bounds cap training duration at `50_000_000`, rollout horizon at `128`, replay ratio at `1.25`, and `vec.num_buffers` at `8`.
- Do not add broad model-size sweep ranges. If model size must be swept later, require explicit human approval and keep a hard ceiling of `policy.hidden_size <= 256` and `policy.num_layers <= 4` unless there is a measured SPS reason to widen it.
- Keep PufferLib core stock for Bat. If sweep parsing conflicts with inherited default sweep keys, solve it through Bat config or command-line args, not core edits.
- Checkpoints trained before the forward-only action model are stale. After
  changing action dimensions or movement semantics, run a normal `train bat`
  before `eval bat --load-model-path latest`.
- On this PufferLib branch, `sweep bat --sweep.max-runs 2` is not enough to
  exercise suggested hyperparameters: the first two launched experiments use
  the current config defaults, and `sweep_obj.suggest(...)` is only called for
  later runs. Use at least `--sweep.max-runs 3` for one actual suggestion, or
  run explicit bounded comparison trains when testing a small acoustic grid.
- Curriculum difficulty should not advance on a single lucky catch. `env.curriculum_successes_per_level` gates advancement so each env must catch the bug multiple times at the current level before increasing bug distance or obstacle count.

## Near-Term Roadmap

Keep these changes small and reversible. Use TDD for env behavior changes,
train/eval after each rung, and commit each known-good rung separately.

1. Harder curriculum and eval difficulty.
   - Plain eval starts from a fresh env at curriculum level 0, so the bug can
     look too close even when training eventually reaches harder levels.
   - Add a configurable initial curriculum level so eval can start at a
     representative harder level without requiring manual in-session catches.
   - Increase the maximum curriculum bug distance so longer runs can keep
     getting harder after the current successful range.
   - Preserve monotonic progress: once an env advances above the configured
     initial level, resets must not drop it back down.

2. Finite chirp budget.
   - Keep the low-curriculum budget below the old `20`-chirp setting; `20`
     proved too easy and should not be part of the default sweep.
   - Keep the chirp budget fixed across curriculum levels. Harder levels and
     clutter legitimately need reacquisition chirps, so budget decay made
     later levels fail for the wrong reason.
   - Track `chirps_used / chirp_budget` as a normalized `0..1` observation.
   - When the budget is exhausted, terminate with a `-1.0` failure penalty if
     the policy attempts another chirp. Do not terminate immediately after the
     last valid chirp, so the final echo can still matter.
   - Log chirp budget, used ratio, remaining ratio, and efficiency to W&B so
     sweeps can distinguish successful policies that waste every chirp from
     successful policies that catch the bug with useful chirp timing.
   - Add a sweepable solve-time efficiency reward where spending the full
     budget scores `0.5` on the efficiency component and using very few chirps
     approaches `1.0`.

3. Later bug motion curriculum.
   - Keep the current fixed-velocity bounce bug as the base rung.
   - Add later stages for sine/cosine perturbations, circular/arc paths, and
     simple maneuvers.
   - Sweep bug speed and maneuver amplitude only after harder curriculum and
     chirp budget are stable.

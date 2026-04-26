# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Safety - CRITICAL

**NEVER run `git add`, `git commit`, `git push`, `git reset`, `git rebase`, `git amend`, `git stash`, or `gh pr create` without the user explicitly asking you to do that specific action in that specific moment.**

- "Here's what I did" / "I committed X" / "I pushed Y" from the user is NOT permission. They are telling you what THEY did, not asking you to commit/push.
- Phrases like "ok let's commit" or "commit this" or "go ahead and commit" ARE permission. Ambiguous phrasing: ask.
- Do not evade permission prompts with alternative command forms. The `ask` list in `.claude/settings.local.json` has patterns for these commands; if you find yourself choosing a form that happens to skip a prompt (e.g. `git -C <path> commit` vs `git commit`), stop and ask the user anyway.
- Rule of thumb: if the command modifies git history, remote state, or stages content, the user must have just asked you to do it.

**NEVER use `git stash -u` or `git stash -a` or `git stash --include-untracked` or `git stash --all` without explicit user approval.**

These flags stash ALL untracked files including documentation, 3D assets, trained weights, config files, and everything else not in .gitignore. A previous agent used `git stash -u` and stashed 40+ files including critical documentation, model weights, and assets.

**Safe**: `git stash` or `git stash push -m "message"` (no `-u` or `-a` flags)

If you need a clean working directory, use `git stash` (tracked changes only) or selectively stage/stash specific files. Never vacuum up all untracked files without asking.

## Current Goal: dogfight4 port (started 2026-04-20)

Porting the dogfight env + training stack from `/home/keith/Git/ml/PufferLib`
(branch `dogfight`, based on PufferLib 3.0) into this repo on branch `dogfight4`,
based on 4.0.

**Ground rules:**
- `/home/keith/Git/ml/PufferLib` is READ-ONLY. Never modify it. Source material only.
- 4.0 is immutable ground truth. Do NOT bend 4.0 to match 3.0; bend dogfight to fit 4.0.
- Port manually in small testable steps — NOT `git cherry-pick`. Read the 3.0 dogfight diffs (`git log 3.0..dogfight`, `git diff 3.0..dogfight -- <file>`), identify a discrete feature ("oh we added a JSON reader / a new obs scheme / a spawn level"), then apply that same change by hand to the 4.0 codebase, adapting to 4.0 conventions. Don't bite off more than you can chew — one small feature per iteration, rebuild and smoke-test between each.
- Plan: `~/.claude/plans/ok-so-now-let-reflective-russell.md`
- 4.0 reorg reference: `~/.claude/projects/-home-keith-Git-ml-PufferLib/memory/porting_env_3_to_4.md`

**Phases:**
1. Copy env C/H/assets into `ocean/dogfight/`, get `./build.sh dogfight` green.
2. Smoke-test with stock 4.0 training (no self-play yet).
3. Port custom behavior feature-by-feature (physics defines → observations → curriculum → rewards → league/eval → self-play loop) — manual changes, not `git cherry-pick`.

Do not start phase N+1 until phase N is green.

## Human created todo list for agents (can be removed when done)
- **Self-play opponent observations were BROKEN through df24 - FIXED (2026-02-07)**: `compute_opponent_observations()` always used the generic 16-obs `compute_obs_momentum_for_plane()` regardless of `obs_scheme`. In self-play, the opponent got wrong-size, wrong-layout observations — e.g. for scheme 0 (17 obs), g_force slot had azimuth, azimuth slot had elevation, etc. Half of all training data was garbage. Fixed by creating `_for_plane` variants for all 5 schemes and dispatching by `obs_scheme` in `compute_opponent_observations()`. **All df24 self-play results are suspect — the policy was training on mismatched observation layouts.**
- **Control oscillation / bang-bang control - IMPLEMENTED (2026-01-29)**: Agent learned to spam controls (aileron 37% bang-bang, rudder 41%). Fixed with squared action rate penalty (`control_rate_penalty=0.05` in config). See `pufferlib/ocean/dogfight/CONTROL_OSCILLATION_ANALYSIS.md` for full analysis. **TODO: Train and verify smooth control, tune penalty scale if needed.**
- **Curriculum logging**: The `[CURRICULUM]` print statements (MASTERED, TARGET, step diagnostics) are swallowed by the TUI during training. Consider switching to `sys.stderr` or a logging framework so messages are visible in real-time. The curriculum works correctly but you can't see progression messages during a run.
- **Train/eval reward penalty**: G-force reward penalty now uses actual `p->g_force` (see G_FORCE_FIX.md). Train and observe: does agent climb toward enemy (good) or dive toward enemy (bad)?
- **Possible gimbal lock / roll snap at vertical**: When plane pitches to near-vertical (up or down), roll may snap 180° instantly. Could be quaternion singularity, euler angle conversion issue, or rendering artifact. Investigate when plane is nose-up or nose-down at ~90° pitch. Test: g_limit_positive shows this behavior.
  - **HUMAN CONFIRMED (2026-01-21)**: Observed in `pitch_direction` test at 5 FPS. Plane pitched down correctly, but at approximately 90° nose-down, roll INSTANTLY snapped 180° in a single frame. Nose stayed correctly pointed down, only roll flipped. Investigate later.
- **Rudder/Yaw - PARTIALLY FIXED, STILL UNCERTAIN (2026-01-23)**:
  - **Fixed**: `CN_DELTA_R` sign flipped from -0.015 to +0.015 (right rudder → nose right)
  - **Fixed**: `CL_DELTA_R` sign flipped from +0.003 to -0.003 (right rudder → left roll, since rudder is above roll axis)
  - **Fixed**: PID sign error in `test_rudder_only_turn` - was commanding wrong aileron direction
  - **Still uncertain**:
    - Not 100% sure yaw direction is correct now (hard to tell visually with small heading change)
    - Not sure if yaw AMOUNT is realistic (only ~1° heading change with full rudder over 6 seconds)
    - Not sure if rudder-roll coupling sign is correct. Physically: right rudder creates sideforce on tail, rudder surface is ABOVE roll axis, so torque should roll plane LEFT. But sideslip itself creates dihedral effect (CL_BETA) which may dominate.
  - **To verify**: Need more careful visual tests or reference data from real P-51D
- Aileron might also be backwards. python pufferlib/ocean/dogfight/test_flight.py --render --fps 10 --test sustained_turn rendered as turning left, rightly or wrongly. Not sure yet. But I saw it turn left. Also, a separate issue, I saw the plan losing altitude. Perhaps elevators not doing their job of keeping the nose on the horizon? Two things to check for this test. python pufferlib/ocean/dogfight/test_flight.py --render --fps 10 --test turn_60 also turned left by the way. Elevator maybe better in that one, but I'm not sure.
- **Elevator authority vs pitch stability - INVESTIGATED (2026-01-23)**:
  - `knife_edge_pull` test: Plane tries to pull up but oscillates - pitch stability fights elevator
  - Root cause: `CM_ALPHA = -1.2` (pitch stability) overpowers `CM_DELTA_E = -0.5` (elevator) above ~8.4° AoA
  - Full elevator gives Cm_de = -0.175, but at 15° AoA stability gives Cm_alpha = -0.31
  - Result: Can't sustain high-G pulls - nose gets pushed back down by "arrow weathervane" effect
  - **To fix**: Either increase CM_DELTA_E or decrease CM_ALPHA magnitude
  - **To verify**: Check JSBSim P-51D values for realistic tuning
- **knife_edge_flight - IMPROVED (2026-01-23)**:
  - Old issue: "nose rose VERY STRONG" - was unrealistic
  - Current behavior: Nose now DROPS (correct - no vertical lift at knife-edge)
  - Plane rolls past 90° toward inverted - this is because test applies ZERO aileron during Phase 2
  - Test might need adjustment to actively hold 90° bank if we want true knife-edge
  - Overall: Physics is now MORE realistic than before
- **Vertical spawn scenarios NOT YET WIRED INTO TRAINING (2026-02-13)**:
  - 5 vertical spawn levels exist in C (apex, past-vertical, mid-climb, merge, pre-merge) in `dogfight.h`
  - Controlled by `vertical_spawn_prob` (float 0-1) and `vertical_level` (int 0-4)
  - **Currently 0% chance**: `vertical_spawn_prob` defaults to 0.0 and nothing in `train_dual_selfplay.py` sets it
  - **`vertical_level` is a single int, not randomized**: picks ONE level every time, not random across all 5
  - **To activate**: Python code in `train_dual_selfplay.py` needs to call `binding.vec_set_vertical_spawn(c_envs, prob, level)` during self-play
  - **To randomize levels**: either randomize `vertical_level` periodically from Python, or change C code to pick a random level each episode
  - The spawn condition requires `selfplay_active && stage == CURRICULUM_AUTOACE` — so it only fires during true self-play at the final curriculum stage
  - Test with: `python pufferlib/ocean/dogfight/test_vertical_spawn.py --level N --fps 10` (N=0-4)
- **Stage 11 (ZOOM_ATTACK) unstable flight (2026-01-27)**:
  - Player spawns 75° nose-up at high speed (~120-130 m/s after reduction from 140-150)
  - Plane oscillates and deviates increasingly - becomes unrealistic
  - Likely related to pitch stability vs elevator authority at high AoA during zoom climb
  - **TODO**: Write a flight test to reproduce and diagnose the instability


## Build

```bash
# We use UV and that's where we get Python
source .venv/bin/activate

# The actual build command
python setup.py build_ext --inplace --force
```

Do NOT use environment variables before the command (e.g., `NO_TORCH=1 python ...`). The build is configured via `setup.py` directly.

## Porting Ocean Envs 3.0 → 4.0

~5 min mechanical port per env. See `/home/keith/.claude/projects/-home-keith-Git-ml-PufferLib/memory/porting_env_3_to_4.md` for the checklist (actions/terminals → float, RNG → unsigned + `rand_r`, minor binding.c format change).

## Ocean Environment Structure

**Full guide**: `pufferlib/ocean/ENV_GUIDE.md` - comprehensive patterns, code templates, and checklist

Each environment in `pufferlib/ocean/{name}/` needs:
- `{name}.h` - Main C header with environment logic
- `binding.c` - Python-C binding using `env_binding.h` template
- `{name}.py` - Python wrapper inheriting `pufferlib.PufferEnv`

Config file: `pufferlib/config/ocean/{name}.ini`

### Standard Env Struct Pattern
```c
typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct EnvName {
    float* observations;      // or char* for discrete
    int* actions;             // or float* for continuous
    float* rewards;
    unsigned char* terminals;
    Log log;
    Client* client;           // raylib rendering (NULL if no render)
    // ... env-specific state
} EnvName;
```

### Required Functions
- `init(Env* env)` - Initialize state
- `allocate(Env* env)` - Allocate buffers (if not using pre-allocated)
- `free_allocated(Env* env)` - Free buffers
- `c_close(Env* env)` - Close resources
- `compute_observations(Env* env)` - State → observations
- `c_reset(Env* env)` - Reset episode
- `c_step(Env* env)` - Step simulation
- `c_render(Env* env)` - Raylib rendering

### Key Reference Files
- `pufferlib/ocean/env_binding.h` - Binding template (read first)
- `pufferlib/ocean/drone_race/` - Best template for continuous actions (Box)
- `pufferlib/ocean/drone_swarm/` - Multi-agent continuous pattern
- `pufferlib/ocean/snake/snake.h` - Simple multi-agent discrete example
- `pufferlib/ocean/impulse_wars/` - Physics-based with Box2D

### Action Space Notes
- All C implementations use `float*` for actions regardless of Python space type
- Discrete/MultiDiscrete actions get converted to floats in Python before passing to C
- For continuous control (flight, drones, physics): use `gymnasium.spaces.Box`
- For multi-agent: `num_agents = num_envs * agents_per_env`, slice buffers accordingly

## External Dependencies
- Raylib 5.5 - rendering (auto-downloaded)
- Box2D - physics (auto-downloaded, used by impulse_wars)
- NumPy <2.0

## Wandb MCP Integration

Wandb MCP server is configured globally (`~/.claude/.mcp.json`). After Claude Code restart, you can query W&B data directly using MCP tools:

- `query_wandb_tool` - Access runs, metrics, experimental data
- `query_wandb_entity_projects` - List projects in entity
- `create_wandb_report_tool` - Generate W&B reports

Example uses: "show runs from df5 sweep", "compare perf metrics across runs", "list wandb projects"

**Note:** Requires `WANDB_API_KEY` in environment (added to `~/.bashrc`).

## Protein Sweep Override

Protein sweeps support injecting hyperparameters mid-sweep via `override.json` file.

**Docs**: `pufferlib/SWEEP_PERSISTENCE.md` - full documentation for persistence and override features.

**CRITICAL: Always perturb override values by 1-5%** — never submit identical or near-identical continuous parameter vectors. The Protein GP will crash (`NotPSDError`) if two override points share the same continuous params (e.g. same config with only `obs_scheme` changed). Jitter every float/int param randomly by 1-5%, then **clamp to the sweep's `[min, max]` bounds** from the INI file (`pufferlib/config/ocean/{env}.ini` and `pufferlib/config/default.ini`). Values outside bounds crash with `ValueError: math domain error`.

## Dogfight Environment

### Documentation
- `pufferlib/ocean/dogfight/SPEC.md` - specification and physics
- `pufferlib/ocean/dogfight/PLAN.md` - implementation phases and checklist
- `pufferlib/ocean/dogfight/RENDERING.md` - rendering implementation notes
- `pufferlib/ocean/dogfight/TRAINING_IMPROVEMENTS.md` - training analysis and improvement ideas
- `pufferlib/ocean/dogfight/OBSERVATION_EXPERIMENTS.md` - observation scheme comparison
- `pufferlib/ocean/dogfight/REALISTIC_SCHEMES.md` - realistic observation variants for sim-to-real transfer
- `pufferlib/ocean/dogfight/BISECTION.md` - training regression bisection
- `pufferlib/ocean/dogfight/SWEEP_SIMPLIFICATION.md` - **df22 analysis: which params to hardcode vs sweep**

- `pufferlib/ocean/dogfight/baselines/BASELINE_SUMMARY.md` - training baselines for comparison
- `pufferlib/ocean/dogfight/physics_log.md` - historical physics test results by commit
- `pufferlib/ocean/dogfight/AIRCRAFT_PERFORMANCE_RL_GUIDE.md` - aircraft performance for RL context
- `pufferlib/ocean/dogfight/P51d_REFERENCE_DATA.md` - P-51D Mustang reference data
- `pufferlib/ocean/dogfight/AUTOPILOT_TODO.md` - autopilot system tech debt and future work
- `pufferlib/ocean/dogfight/G_FORCE_FIX.md` - G-force calculation (FIXED)
- `pufferlib/ocean/dogfight/PHYSICS_AUDIT.md` - **Comparison of dogfight vs drone_race physics (integration, control authority, damping, gimbal lock)**
- `pufferlib/ocean/dogfight/PID_TUNING_GUIDE.md` - **PID tuning for autopilots: JSBSim P-51D validation, tuning methods, cascaded control architecture**
- `pufferlib/ocean/dogfight/PHYSICS_SIMPLIFICATION_PLAN.md` - physics simplification (completed)
- `pufferlib/ocean/dogfight/CONTROL_OSCILLATION_ANALYSIS.md` - **Bang-bang control diagnosis and RL solutions (action rate penalties, CAPS, curriculum)**
- `pufferlib/ocean/dogfight/research/LEAGUE_SYSTEM.md` - **League system design: population training, cross-scheme evaluation, rating anchors**

### Stability Derivatives (validated against JSBSim P-51D)
| Derivative | Our Value | JSBSim P-51D | Status |
|------------|-----------|--------------|--------|
| CM_Q (pitch damping) | -10.0 | -10.0 | ✅ Matched |
| CL_P (roll damping) | -0.4 | -0.4 | ✅ Exact |
| CN_R (yaw damping) | -0.15 | -0.15 | ✅ Exact |
| CL_BETA (dihedral) | -0.08 | -0.10 | ✅ Close |
| CN_BETA (weathervane) | 0.12 | 0.12 | ✅ Exact |

### Aces High III Integration (sim-to-real transfer)
- `pufferlib/ocean/dogfight/aceshigh/DLL_SPEC.md` - **DLL API specification for Hitech Creations**
- `pufferlib/ocean/dogfight/aceshigh/ARCHITECTURE.md` - full system design for AH3 integration
- `pufferlib/ocean/dogfight/aceshigh/CONTROLS.md` - AH3 keyboard control reference
- `pufferlib/ocean/dogfight/aceshigh/INPUT_SYSTEM.md` - joystick/analog input handling in AH3
- `pufferlib/ocean/dogfight/aceshigh/FLIGHT_PHYSICS.md` - AH3 physics effects and transfer analysis

### Running Dogfight Tests
```bash
# C unit tests (vec3/quat math, physics, rewards, combat)
gcc -I raylib-5.5_linux_amd64/include -o pufferlib/ocean/dogfight/dogfight_test pufferlib/ocean/dogfight/dogfight_test.c raylib-5.5_linux_amd64/lib/libraylib.a -lm -lpthread -ldl && ./pufferlib/ocean/dogfight/dogfight_test

# Python flight physics tests (uses force_state() for initial conditions, hijacks actions, runs test flights, fast)
source .venv/bin/activate
python pufferlib/ocean/dogfight/test_flight.py

# Visual flight tests - RENDER at 10 FPS so human can see physics behavior
source .venv/bin/activate
python pufferlib/ocean/dogfight/test_flight.py --render --fps 10

# Run single test with rendering (press ESC to exit)
source .venv/bin/activate
python pufferlib/ocean/dogfight/test_flight.py --render --fps 10 --test pitch_direction
```

**Physics**: Full 6DOF with RK4 integration, aerodynamic moments, stability derivatives.

**Render mode note**: When running with `--render`, let the test run continuously - the human will press ESC when they've seen enough. Don't add artificial time limits or auto-close logic.

**Always run both test suites (C unit tests + Python flight tests) after modifying dogfight.h or dogfight_test.c.**

### Unit & Integration Tests (tests/ directory)

```bash
# Pure logic tests (no GPU, instant)
python tests/test_league.py              # 44 tests: manifest, ELO, promotion
python tests/test_sweep_persistence_and_override.py  # 26 tests: sweep state, overrides
python tests/test_eval.py                # 17 tests: anchor eval, rating injection, elo edge cases

# Full suite (needs GPU + built C extension)
python tests/test_eval.py               # Includes GPU integration tests
```

These test the league/eval/sweep pipeline logic. Run after modifying:
- `anchor_eval.py`, `elo_eval.py`, `league.py`, `league_manifest.py`
- `train_dual_selfplay.py` (anchor_rating injection, sweep metric filtering)
- `collect_from_wandb.py` (checkpoint inference)

### Pre-Sweep Validation Tests (Run Only When Asked)
**NOTE: These tests are slow (~30s). Only run when specifically asked or before launching a sweep.**

Run BEFORE launching sweeps to catch dangerous hyperparameter combinations that could cause NaN crashes mid-sweep:
```bash
# Run all pre-sweep tests
python pufferlib/ocean/dogfight/test_presweep.py

# Verbose output with observation/reward stats
python pufferlib/ocean/dogfight/test_presweep.py --verbose

# Run specific test
python pufferlib/ocean/dogfight/test_presweep.py --test firm_gorge_40
```

Tests extreme hyperparameter combinations that have caused crashes:

**firm-gorge-40 (df5)**: Low `max_grad_norm=0.21` + high `vf_coef=4.71` + low `gae_lambda=0.89` caused gradient explosion.

**toasty-snowflake-4 (df13)**: `adam_eps=1.2e-11` (extremely low, default is 1e-8) likely caused numerical instability. Also had `obs_scheme=5`, `vf_coef=3.855`, `max_grad_norm=1.71`.

### Full Integration Test (Multiprocessing + LSTM + Self-Play)
**NOTE: ~3 min at 1.6M SPS. Run before commits that touch `train_dual_selfplay.py`, `vector.py`, or the training pipeline.**

Pipeline smoke test: multiprocessing vecenv, LSTM policy, dual-perspective self-play, opponent checkpointing. Verifies the full training loop runs without crashes.

Hyperparameters from `rosy-smoke-112` (df33, fastest hs=128 to converge):
```bash
source .venv/bin/activate && python setup.py build_ext --inplace --force && \
python pufferlib/ocean/dogfight/train_dual_selfplay.py train \
  --wandb --wandb-project df_test \
  --policy.hidden-size 128 \
  --env.obs-scheme 0 \
  --train.total-timesteps 300000000 \
  --train.adam-eps 6.74e-08 \
  --train.adam-beta1 0.9315 \
  --train.gae-lambda 0.9955 \
  --train.vtrace-rho-clip 1.776 \
  --env.recovery-trigger-prob 0.01822 \
  --env.reward-aim-scale 0.001483 \
  --env.vertical-spawn-prob 0.01563 \
  --env.energy-gain-scale 0.001 \
  --env.energy-loss-scale 0.0005 \
  --env.energy-advantage-scale 0.004 \
  --selfplay.sp-prob-start 1.0 \
  --selfplay.sp-prob-ramp-steps 1000000 \
  --selfplay.opponent-epoch-length 2000000 \
  2>&1 | tee /tmp/df_integration_test.log
```

Monitor with tail (if run in background with `&`):
```bash
# Check for crashes
strings /tmp/df_integration_test.log | grep -iE 'error|exception|traceback|assert|LSTM|shape|crash'

# Check self-play ratchet is running (should see epoch_done events)
grep 'RATCHET' league/logs/train_local_*.log | tail -10
```

**Pass criteria** (300M steps, ~3 min):
- No crashes, no LSTM assertion errors
- Curriculum reaches stage 20 by ~200M steps
- Self-play activates and ratchet evaluates opponents (look for `[RATCHET] event=epoch_done` in logs)
- Checkpoints saved to `checkpoints/selfplay_*/`
- Strength will be 0 — that's OK, promotions need 800M+. This tests the pipeline, not convergence.

### Regression Benchmarking

Run fixed-hyper training before commits to detect regressions in throughput and quality.
Results stored in `pufferlib/ocean/dogfight/benchmarks.json`.

**Full quality check (~17 min, 800M steps):**
```bash
python pufferlib/ocean/dogfight/run_benchmark.py
python pufferlib/ocean/dogfight/run_benchmark.py --notes "Added new reward"
```

**Quick SPS check (~6 min, 300M steps):**
```bash
python pufferlib/ocean/dogfight/run_benchmark.py --quick
```

**Show history:**
```bash
python pufferlib/ocean/dogfight/run_benchmark.py --history
```

**Rules:**
- Same GPU, no other GPU jobs running
- Commit first (dirty benchmarks get `*` flag)
- Don't pipe through tee
- NEVER change `BENCH_HYPERS` in run_benchmark.py

**Hypers pinned from:** df36 `tuiiz5po` (hs=128, obs=1, str=0.94, AR=1002)

**Tracked metrics:** SPS, wall_seconds, strength, anchor_rating, perf, clean_fights,
sp_player_kills, sp_opp_kills, avg_control_rate, accuracy, shots_fired, episode_length, player_ground

### Training Dogfight (Standard)
```bash
python -m pufferlib.pufferl train puffer_dogfight
```

**Do NOT use command-line args** like `--train.total-timesteps` or `--env.obs-scheme`. Instead, edit the INI file directly at `pufferlib/config/ocean/dogfight.ini`.

### Self-Play Training and Eval
```bash
# Train with self-play
python pufferlib/ocean/dogfight/train_dual_selfplay.py train

# Eval with self-play (USE THIS, not pufferlib.pufferl eval)
python pufferlib/ocean/dogfight/train_dual_selfplay.py eval --load-model-path experiments/puffer_dogfight_XXXXX.pt --opponent-checkpoint checkpoints/selfplay_XXXXX/checkpoint_stageYY_stepZZZZ.pt --render-mode raylib
```

### League System Logs

All league/training operations write structured logs to `league/logs/`. Logging is implemented in `pufferlib/ocean/dogfight/dogfight_log.py` using Python's `logging` module.

**Log format**: `HH:MM:SS [TAG] key=value key=value ...`

**Log file naming**: `{command}_{YYYY-MM-DD_HHMMSS}.log` — e.g. `league_eval_2026-02-12_153247.log`

**Log tags** (searchable with grep):
- `[EVAL]` / `[MATCH]` / `[RATING]` — evaluation rounds, individual matchups, Elo ratings
- `[ANCHOR]` — anchor evaluation results, cross-sweep ratings
- `[TRAIN]` — training start/done/errors
- `[VERIFY]` — verification gauntlet results
- `[VERDICT]` / `[PROMOTE]` — promotion/rejection decisions
- `[SELFPLAY]` — self-play activation, opponent sampling, handicap levels
- `[RATCHET]` — opponent ratchet epochs, gate pass/fail, rank-ups
- `[CHECKPOINT]` — model saves, loads, milestones
- `[ROUND]` / `[PHASE]` — round orchestration, phase transitions
- `[ERROR]` — errors with tracebacks

**Reading logs**:
```bash
# Follow a live league run
tail -f league/logs/league_eval_*.log

# Check recent ratings
grep "\[RATING\]" league/logs/*.log | tail -20

# Check promotion/rejection decisions
grep "\[VERDICT\]\|\[PROMOTE\]" league/logs/*.log

# Track self-play opponent progression
grep "\[RATCHET\].*rank_up" league/logs/train_*.log

# Find errors
grep "\[ERROR\]" league/logs/*.log
```

### Anchor Evaluation (Cross-Sweep Comparable Rating)

The `strength` metric is NOT comparable across sweeps — it measures progress against internal checkpoints. Use anchor evaluation for true cross-sweep comparison.

**Key files:**
- `pufferlib/ocean/dogfight/anchor_eval.py` — standalone evaluation against fixed reference opponents
- `pufferlib/ocean/dogfight/reference_opponents/manifest.json` — anchor definitions (autopilot + neural)
- `pufferlib/ocean/dogfight/reference_opponents/anchors/` — frozen neural anchor .pt files

**Evaluate a model against anchors:**
```bash
python pufferlib/ocean/dogfight/anchor_eval.py eval \
    --model experiments/model.pt --obs-scheme 0 --games 50
```

**Add a neural anchor (permanent reference opponent):**
```bash
python pufferlib/ocean/dogfight/anchor_eval.py add-anchor \
    --model path/to/best.pt --tag df29_best_scheme0 --obs-scheme 0
```

**Cross-sweep tournament (collect + eval):**
```bash
# Collect scheme 0 models from multiple sweeps into one manifest
python pufferlib/ocean/dogfight/collect_from_wandb.py \
    --project df28 --top-n 5 --output-dir league/cross_sweep/ --obs-scheme 0
python pufferlib/ocean/dogfight/collect_from_wandb.py \
    --project df29 --top-n 5 --output-dir league/cross_sweep/ --obs-scheme 0 --merge
python pufferlib/ocean/dogfight/collect_from_wandb.py \
    --project df30 --top-n 5 --output-dir league/cross_sweep/ --obs-scheme 0 --merge

# Run round-robin tournament
python pufferlib/ocean/dogfight/league.py eval \
    --manifest league/cross_sweep/manifest.json --games-per-pair 50
```

**Cross-scheme evaluation (scheme 0 vs scheme 1):**
- Autopilot anchors work with ANY obs_scheme — both scheme 0 and scheme 1 agents can be rated against the same autopilots
- Neural anchors support cross-scheme via `vec_set_opponent_obs_scheme()` in C — the opponent gets its own observation layout
- `anchor_eval.py` handles this automatically: same-scheme uses fast path, cross-scheme uses `run_matches_cross_scheme()`

**Current anchors** (3 total, ~5s eval at 30 games):
- `autopilot_s20` — floor anchor (expected_rating=600), curriculum autoace
- `df31_mid_scheme0` — mid-tier neural anchor (expected_rating=1000), scheme 0, hidden_size=128
- `df31_top_scheme0` — top-tier neural anchor (expected_rating=1300), scheme 0, hidden_size=128

**The metric**: `anchor_rating` (W&B: `environment/anchor_rating`)
- Replaces `strength` as the ground truth for model quality
- Runs at end of each sweep run IF `strength >= strength_gate` (default 0.3)
- Below gate: estimated as `100 + strength * 1400` (monotonic, instant, no eval cost)
- Above gate: real eval against fixed anchors (~5s per run with 3 anchors)
- Optional periodic logging during training: set `enabled = 1` in `[anchor_eval]`
- To make Protein optimize it: change `metric = anchor_rating` in `[sweep]`
- Per-anchor win rates logged: `environment/anchor_wr_autopilot_s20`, `environment/anchor_wr_df31_mid_scheme0`, etc.
- Raw Bradley-Terry rating in ~100-2000 range, comparable across sweeps

### Baseline Benchmarking
Run training and save to log for comparison:
```bash
python -m pufferlib.pufferl train puffer_dogfight 2>&1 | tee pufferlib/ocean/dogfight/baselines/run_name.log
```

View final results:
```bash
tail -50 pufferlib/ocean/dogfight/baselines/*.log
```

## Running Long Tasks in Background (Sweeps, Training)

When running sweeps or long training runs, use background execution to monitor progress in real-time while reasoning through results.

### 1. Launch with Background Flag

Use `run_in_background: true` in the Bash tool:
```bash
python -m pufferlib.pufferl sweep puffer_dogfight --wandb --wandb-project df17 --max-runs 1 2>&1
```

This returns immediately with:
- A **task ID** (e.g., `be75072`)
- An **output file path**: `/tmp/claude/.../tasks/<task_id>.output`

### 2. Always Set an Exit Strategy

**CRITICAL**: Long-running tasks need bounded execution. For sweeps, use `--max-runs`:
```bash
--max-runs 1    # Single run for testing
--max-runs 5    # Short sweep
```

Without `--max-runs`, a Protein sweep runs indefinitely. Always limit runs when testing.

### 3. Monitor with tail

Check latest output (TUI frames with metrics):
```bash
tail -80 /tmp/claude/.../tasks/<task_id>.output
```

Wait and check:
```bash
sleep 30 && tail -50 /tmp/claude/.../tasks/<task_id>.output
```

### 4. Search for Specific Messages

The TUI swallows print statements. Use `strings` + `grep` to find debug output:
```bash
# Find debug messages (adapt pattern to your printf's)
strings /tmp/.../output | grep -E 'DEBUG|INFO|\[.*\]' | tail -20

# Find errors
strings /tmp/.../output | grep -iE 'error|exception|traceback'
```

### 5. Reasoning Loop

The workflow is:
1. **Launch** background task with exit strategy (`--max-runs 1`)
2. **Wait** a bit (`sleep 15-30`)
3. **Check** output with `tail`
4. **Search** for debug printf's with `grep`
5. **Reason** through what the metrics mean
6. **Predict** (optional) what should happen next based on current state
7. **Repeat** steps 2-6 until task completes or you have enough info

The **Predict** step is valuable: "Agent is at stage 9.4 with 34% kill rate - should trigger demotion after 10 stuck intervals." Then verify by checking later output.

### Example Session
```
# Launch with exit strategy
Bash(run_in_background=true): python -m pufferlib.pufferl sweep puffer_dogfight --wandb --wandb-project df17 --max-runs 1 2>&1
→ Task ID: be75072, output: /tmp/.../be75072.output

# Check startup
Bash: sleep 15 && tail -80 /tmp/.../be75072.output
→ See TUI: Steps=36M, Stage=0, Perf=0.88

# Search for debug output
Bash: strings /tmp/.../output | grep '\[CURRICULUM\]' | tail -20
→ See: [CURRICULUM] step=7101440 stage=0.00 ... adv=True

# Later check
Bash: tail -50 /tmp/.../output
→ See TUI: Steps=108M, Stage=7.6, Perf=0.99, Ultimate=0.41

# Reason: Agent advanced from stage 0→7.6, performing well
# Predict: Should reach stage 9+ and possibly trigger demotion if stuck
```

### System Reminders

Claude Code sends `<system-reminder>` notifications when background tasks have new output. These prompt you to check on running tasks.

## Test-Driven Development

1. **Write test first** - Before implementing a feature, write the test that validates it
2. **Implement until test passes** - Write minimal code to make the test pass
3. **Check off and record** - When step passes, check left box and add test reference

Before making changes to code, think about how it will effect tests. Making changes that will change init() args? Think about how that will effect tests.

### PLAN.md Checkbox Convention
```
- [x] [ ] 1.3 Implement c_reset() → test_reset()
  ^   ^
  |   +-- Second checkbox: Full audit verified (later)
  +------ First checkbox: Implemented and test passes
```

## Flight Physics References

### Drag Polar
- [Aircraft Drag Polar Tutorial](https://agodemar.github.io/FlightMechanics4Pilots/mypages/drag-polar/)
- [AeroToolbox Drag Polar](https://aerotoolbox.com/drag-polar/)
- [NASA Induced Drag](https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/induced.html)

### Energy-Maneuverability Theory
- [Wikipedia E-M Theory](https://en.wikipedia.org/wiki/Energy–maneuverability_theory)
- [Boyd's E-M Theory Development](https://acquisitiontalk.com/2019/04/john-boyd-and-the-development-of-em-theory/)

### Flight Dynamics Models
- [Princeton 6DOF Simulation Code](https://stengel.mycpanel.princeton.edu/FDcodeB.html)
- [MITRE Point-Mass Model (PDF)](https://www.mitre.org/sites/default/files/publications/pr_15-1318-derivation-of-point-mass-aircraft-model-used-for-fast-time-simulation.pdf)
- [JSBSim GitHub](https://github.com/JSBSim-Team/jsbsim)

### Game/Simulation
- [Physics for Game Developers - Aircraft](https://www.oreilly.com/library/view/physics-for-game/9781449361037/ch15.html)
- [Gazebo Aerodynamics](https://classic.gazebosim.org/tutorials?tut=aerodynamics)

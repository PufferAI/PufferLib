# SoulsRL Minimal — RL-Focused Boss Fight Environment

## Goal

Build a **minimal** 2D boss fight environment to learn RL concepts with PufferLib.
Focus: **observation design, reward shaping, training experiments and a bit of game dev using Raylib**

The boss has **1 attack** (AOE burst). All hitboxes are circles.

---

## Core Mechanics (Simplified)

### Constants

```
Tick rate: 30 ticks/sec (dt = 1/30)
Arena: 10 x 10 units (centered at origin, so bounds are -5 to +5)

Player:
  - radius: 0.3
  - HP: 100
  - speed: 3.0 units/sec (~0.1 units/tick)

Boss:
  - radius: 0.5
  - HP: 100
  - position: fixed at (0, 0) — does not move
```

### Player Actions (Discrete, 7 total)

```
0: NOOP
1: UP
2: DOWN
3: LEFT
4: RIGHT
5: DODGE
6: ATTACK
```

### Player States

```
FREE     — can move, can act
DODGE    — 6 ticks, i-frames on ticks 1-5, moves at 2.5x speed in last move_dir
ATTACK   — windup(4) + active(3) + recovery(6) = 13 ticks total, no movement
```

**Cooldowns:**

- Dodge: 15 ticks after dodge ends
- Attack: No cooldown (but you're locked for 13 ticks)

**Attack hitbox (during ACTIVE):**

- Circle at `player_pos + facing * 0.7`, radius `0.4`
- `facing` = direction to boss at attack start
- Damage: 10

### Boss Behavior (Single Attack)

Boss cycles: `IDLE → WINDUP → ACTIVE → RECOVERY → IDLE`

```
IDLE:     12 ticks (0.4s) — does nothing
WINDUP:   18 ticks (0.6s) — telegraphing, no damage
ACTIVE:    3 ticks (0.1s) — AOE hits
RECOVERY: 15 ticks (0.5s) — vulnerable, no damage
```

**AOE Attack:**

- Circle centered on boss, radius `1.5`
- Damage: 20
- Player takes damage if: in AOE radius AND not in i-frames

---

## Observation Space (14 floats)

Keep it minimal. You can ablate later.

```
Geometry (3):
  0: rel_boss_x      = boss_x - player_x (normalized by arena half-size)
  1: rel_boss_y      = boss_y - player_y
  2: distance        = clamp(dist / 5.0, 0, 1)

Player (5):
  3: player_hp       = hp / 100
  4: dodge_ready     = 1.0 if can dodge, else 0.0
  5: player_state    = {FREE: 0, DODGE: 0.33, ATTACK: 0.66}  # scalar encoding
  6: state_progress  = ticks_in_state / state_duration
  7: move_dir_x      = -1 to 1

Boss (6):
  8:  boss_hp        = hp / 100
  9:  boss_phase     = {IDLE: 0, WINDUP: 0.33, ACTIVE: 0.66, RECOVERY: 1.0}
  10: phase_progress = ticks_in_phase / phase_duration
  11: time_to_damage = ticks until ACTIVE starts / 18 (1.0 during IDLE/RECOVERY)
  12: in_aoe_range   = 1.0 if distance < 1.5, else 0.0
  13: boss_attacking = 1.0 if in WINDUP/ACTIVE, else 0.0
```

---

## Reward Function (v1 — HP delta)

```python
# Per step
reward = 0
reward += (boss_hp_prev - boss_hp_now) * 0.1      # +1.0 per hit landed
reward += (player_hp_prev - player_hp_now) * -0.1 # -2.0 per AOE hit taken
reward += -0.001                                   # time penalty

# Terminal
if boss_hp <= 0: reward += 1.0   # win bonus
if player_hp <= 0: reward -= 1.0 # lose penalty
```

---

## Episode Termination

- `terminated = True` if player or boss HP <= 0
- `truncated = True` if ticks >= 900 (30 seconds)

---

## Implementation (Single File)

Everything in `soulsrl.py` (~250-300 lines):

```python
class SoulsEnv(pufferlib.PufferEnv):
    # Player state machine
    # Boss state machine
    # Collision detection (circle-circle only)
    # Observation building
    # Reward calculation
```

No separate core.py, no rendering, no curriculum stages.

---

## RL Experiments

Once v1 is working, run these experiments to learn RL concepts:

### Experiment 1: Observation Ablations

| Variant   | Change                                                          | Hypothesis                             |
| --------- | --------------------------------------------------------------- | -------------------------------------- |
| no_timing | Remove `time_to_damage`, `phase_progress`                       | Agent can't learn precise dodge timing |
| no_range  | Remove `in_aoe_range`, `distance`                               | Agent can't learn spacing              |
| minimal   | Only: `distance`, `time_to_damage`, `dodge_ready`, `boss_phase` | Test minimum viable obs                |
| noisy     | Add 5 uniform random floats                                     | Network should ignore noise            |

### Experiment 2: Reward Shaping

| Variant         | Change                           | Hypothesis                 |
| --------------- | -------------------------------- | -------------------------- |
| sparse          | Only win/lose bonus, no HP delta | Much slower learning       |
| no_time_penalty | Remove -0.001/step               | Agent becomes passive      |
| dodge_bonus     | +0.2 for dodging during ACTIVE   | Might create dodge spam    |
| proximity       | +0.01 for being close to boss    | Might discourage safe play |

### Experiment 3: Hyperparameters

| Param         | Values           | What to observe             |
| ------------- | ---------------- | --------------------------- |
| learning_rate | 1e-3, 3e-4, 1e-4 | Learning speed vs stability |
| ent_coef      | 0.0, 0.01, 0.05  | Exploration vs exploitation |
| num_envs      | 8, 32, 128       | Sample efficiency           |
| hidden_size   | 32, 64, 128      | Model capacity              |

---

## Success Criteria

1. **Baseline works**: Random agent wins ~0%, trained agent wins >80%
2. **Learned timing**: Agent dodges during WINDUP, not randomly
3. **Learned punish**: Agent attacks during RECOVERY, not during ACTIVE
4. **Experiments complete**: At least 3 ablations run with plotted comparisons

---

## Optional Extensions (After Experiments)

Only add these if baseline experiments are done:

1. **Sweep attack**: Cone hitbox, tests directional dodging
2. **Boss movement**: Slow drift toward player
3. **Combo attack**: Multi-hit sequence, tests dodge timing
4. **ASCII rendering**: For debugging/demo
5. **Curriculum**: Start with longer windup, tighten over training

---

## Deliverables

1. `soulsrl.py` — Environment (PufferEnv)
2. `train.py` — Training script with logging
3. `experiments/` — Saved runs with different configs
4. `results.md` — Summary of what you learned from experiments

---

## Timeline Estimate

- Day 1: Implement `soulsrl.py`, verify with random agent
- Day 2: Train baseline, confirm learning
- Day 3-4: Run observation ablations
- Day 5-6: Run reward experiments
- Day 7: Document findings, optional extensions

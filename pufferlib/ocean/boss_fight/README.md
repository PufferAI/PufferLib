# SoulsRL Minimal — RL-Focused Boss Fight Environment

## Goal

Build a **minimal** 2D boss fight environment to learn RL concepts with PufferLib.
Focus: **observation design, reward shaping, training experiments and a bit of game dev using Raylib**

The boss has **1 attack** (AOE burst). All hitboxes are circles (collision = circles overlap).

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

### Boss Behavior (Single Attack)

Boss cycles: `IDLE → WINDUP → ACTIVE → RECOVERY → IDLE`

```
IDLE:     12 ticks (0.4s) — does nothing
WINDUP:   18 ticks (0.6s) — telegraphing, no damage
ACTIVE:    3 ticks (0.1s) — AOE hits
RECOVERY: 15 ticks (0.5s) — vulnerable, no damage
```

## RL Experiments

Once v1 is working, design experiments to understand RL concepts:

### Experiment Ideas

**Observation Ablations** — Which observations actually matter?

- What happens if the agent can't see timing information?
- Does it need absolute position, or is relative enough?
- What's the minimum viable observation space?
- Can the network learn to ignore irrelevant/noisy inputs?

**Reward Shaping** — How does reward design affect behavior?

- What if you only reward winning/losing (sparse)?
- What happens without a time penalty?
- Can you incentivize specific behaviors (dodging at the right time)?
- What unintended behaviors might reward bonuses create?

**Hyperparameters** — See `boss_fight.ini` for the sweep config

- Learning rate: stability vs speed
- Entropy coefficient: exploration vs exploitation
- Batch size / num_envs: sample efficiency
- Network size: capacity vs overfitting

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

1. `boss_fight.h` — Core game logic in C
2. `boss_fight.c` — Standalone test binary
3. `boss_fight.py` — PufferLib environment wrapper
4. `experiments/` — Saved runs with different configs
5. `results.md` — Summary of what you learned from experiments

---

## Milestones

1. **Environment works**: `c_step()` implemented, can play manually with keyboard
2. **Random baseline**: Random agent wins ~0%, confirms game is non-trivial
3. **Learning signal**: Trained agent shows improvement over random
4. **Competent agent**: Win rate >80%
5. **Experiments**: At least 3 ablations with documented findings

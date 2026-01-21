# BossFight Reinforcement Learning project

I'm implementing a RL environment using PufferLib in C + Python.

Environment spec file is in `./pufferlib/ocean/boss_fight/README.md`.

You are in PufferLib's (puffer.ai) source repository which contains "Ocean" - a collection of environments.

The environment code I'm working on is located in `./pufferlib/ocean/boss_fight/`. Environment configuration is in `./pufferlib/config/boss_fight.ini`

### Setup

1. Fork pufferlib, create new branch

2. Run these:

```
uv venv
uv pip install -e .
```

3. Setup files using templates, update `environment.py`

4. Not sure what this does yet:

```
python setup.py build_boss_fight --inplace --force
```

### Testing

Make sure shit's running:

```
uv pip install -e .
python -c "
from pufferlib.ocean.boss_fight import BossFight
import numpy as np
env = BossFight(num_envs=2)
env.reset()
for _ in range(100):
    env.step(np.random.randint(0, 7, size=2))
print('ok')
env.close()
"
```

Train and check scores:

```
puffer train puffer_boss_fight --train.total-timesteps 50000
```

## Eval

```
puffer eval puffer_boss_fight --load-model-path $(ls -t experiments/puffer_boss_fight_*/model_*.pt | head -1)
```

## Environment

**Gameplay**: 2D boss fight. Player moves around a 10x10 arena (-5 to +5), dodges boss AOE attacks, and attacks back. Boss is stationary at center (0,0). Tick rate: 30/sec.

**Actions** (7 discrete): NOOP, UP, DOWN, LEFT, RIGHT, DODGE, ATTACK

**Player States**:
- `IDLING` — can move, dodge, or attack
- `DODGING` — 6 ticks, invincible, can't act
- `ATTACKING` — 3 ticks, stationary, can't act

**Boss Behavior** (cycles continuously):
- `IDLE` (7 ticks) — does nothing
- `WINDUP` (5 ticks) — telegraph, player should prepare to dodge
- `ACTIVE` (5 ticks) — AOE damage zone active, dodge or get hit
- `RECOVERY` (5 ticks) — safe window to attack boss

**Rewards**:
- `+10.0` — kill boss
- `+0.5` — hit boss with attack
- `+0.5` — successfully dodge during boss attack
- `+0.05` — approach boss (distance shaping)
- `-0.01` — per-step penalty
- `-0.5` — get hit by boss (10 dmg)
- `-1.0` — hit arena wall
- `-10.0` — die
- `-10.0` — timeout

**Episode termination**:
- Boss HP ≤ 0 (player wins)
- Player HP ≤ 0 (player dies)
- 300 ticks timeout (~10 sec)

**Parameters**:
- Player/Boss HP: 100
- Player attack dmg: 3, Boss AOE dmg: 10
- Player speed: 0.1 units/tick
- Dodge: 6 ticks duration, 15 tick cooldown
- Boss cycle: IDLE(7) → WINDUP(5) → ACTIVE(5) → RECOVERY(5) = 22 ticks/cycle

**Observations** (13 floats):
1. `boss_x - player_x` — relative X to boss
2. `boss_y - player_y` — relative Y to boss
3. `player_x` — absolute X position
4. `player_y` — absolute Y position
5. `boss_x` — boss X (always 0)
6. `boss_y` — boss Y (always 0)
7. `player_hp` — player health
8. `boss_hp` — boss health
9. `player_state` — 0=IDLING, 1=DODGING, 2=ATTACKING
10. `player_dodge_cooldown` — ticks until dodge available
11. `player_state_ticks` — ticks remaining in current state
12. `boss_state` — 0=IDLE, 1=WINDUP, 2=ATTACKING, 3=RECOVERY
13. `boss_phase_ticks` — ticks remaining in current boss phase

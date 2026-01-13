# Baseline Training Results

Date: 2026-01-12
Training: 100M steps each

---

## Pre-Reward Shaping (baseline)
Reward: `-dist/10000` per step (pursuit only)

| Run | Episode Return | Episode Length |
|-----|----------------|----------------|
| 1   | -31.78         | 1111           |
| 2   | -46.42         | 1247           |
| 3   | -73.12         | 1371           |
| **Mean** | **-50.44** | **1243**       |

Observations:
- High variance (-31 to -73)
- All returns negative
- Weak reward signal (~-0.03 per step)

---

## Post-Reward Shaping (Phase 3.5)
Reward: base pursuit + closing velocity + tail position + altitude/speed penalties

| Run | Episode Return | Episode Length |
|-----|----------------|----------------|
| 1   | -66.82         | 1140           |
| 2   | +5.32          | 1063           |
| 3   | +16.13         | 1050           |
| **Mean** | **-15.12** | **1084**       |

Observations:
- **2 of 3 runs achieved positive returns** (significant improvement)
- Shorter episodes (more decisive behavior)
- Still high variance (need more tuning)
- Closing velocity and tail position rewards working

---

## Phase 5: Combat Mechanics
Reward: pursuit shaping + hit (+1.0) + kill (+10.0)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +29.54         | 1047           | 0.24  | 0.24/12.0       |
| 2   | +12.46         | 1081           | 0.16  | 0.16/11.8       |
| 3   | +28.31         | 1061           | 0.18  | 0.18/11.7       |
| **Mean** | **+23.44** | **1063**       | **0.19** | **0.19/11.8** |

Observations:
- **All 3 runs positive** (major improvement from Phase 3.5 mean of -15.12)
- Agent learned to shoot (~12 shots/episode, ~1.6% accuracy)
- ~0.19 kills per episode on average
- Combat rewards providing clear learning signal

---

## Spawn Direction Fix (FAILED)
Date: 2026-01-13
Change: Make respawned opponent fly same direction as player instead of always +X

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | -6.91          | 1107           | 0.133 | 0.133/11.0      |
| 2   | -97.46         | 1118           | 0.06  | 0.06/10.8       |
| 3   | -34.51         | 1075           | 0.061 | 0.06/12.4       |
| **Mean** | **-46.29** | **1100**       | **0.085** | **0.08/11.4** |

Observations:
- **Significantly worse than baseline** (-46.29 vs +23.44)
- Predictable +X direction was actually easier to learn
- **REVERTED** - keeping opponent always flies +X

---

## Aiming Reward (SUCCESS)
Date: 2026-01-13
Change: Add continuous reward for gun cone alignment (tracking bonus + firing solution bonus)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +63.08         | 1067           | 0.51  | 0.51/10.2       |
| 2   | +12.64         | 1127           | 0.21  | 0.21/10.2       |
| 3   | +35.41         | 1113           | 0.37  | 0.37/9.9        |
| **Mean** | **+37.04** | **1102**       | **0.36** | **0.36/10.1** |

Observations:
- **+58% improvement in return** (+23.44 → +37.04)
- **+89% improvement in kills** (0.19 → 0.36)
- **+125% improvement in accuracy** (1.6% → 3.6%)
- Aiming reward provides gradient for learning to aim, not just fire

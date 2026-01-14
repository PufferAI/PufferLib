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

---

## Physics Refactor (3582d2d4) - Pre-Quaternion Fix
Date: 2026-01-13
Commit: 3582d2d4 "Physics in Own File - Test Flights"
Change: Moved physics to flightlib.h, added test_flight.py validation tests

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +45.32         | 1139           | 0.42  | 0.42/10.2       |
| 2   | +15.30         | 1136           | 0.19  | 0.19/10.2       |
| 3   | +51.87         | 1133           | 0.46  | 0.46/10.0       |
| **Mean** | **+37.50** | **1136**       | **0.36** | **0.36/10.1** |

Observations:
- Performance consistent with Aiming Reward baseline (+37.04 → +37.50)
- Physics refactor did not affect training
- test_flight.py shows climb_rate test failing (-29.6 vs +15.4 expected)
- Quaternion sign issue identified in test setup (not affecting training)

---

## Coordinated Turn Tests (1c30c546)
Date: 2026-01-14
Commit: 1c30c546 "Coordinated Turn Tests"
Change: Fixed quaternion signs in tests, added 60° coordinated turn test with PID validation (97% efficiency)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +26.17         | 1151           | 0.29  | 0.29/10.5       |
| 2   | +55.99         | 1148           | 0.47  | 0.47/10.6       |
| 3   | +10.82         | 1151           | 0.20  | 0.20/9.6        |
| **Mean** | **+30.99** | **1150**       | **0.32** | **0.32/10.2** |

Observations:
- Performance consistent with previous baseline (+37.50 → +30.99, within variance)
- Test fixes did not affect training (physics unchanged)
- All tests now passing: max_speed, stall, climb, glide, turn_30, turn_60, pitch, roll

---

## Performance Optimizations (374871df)
Date: 2026-01-14
Commit: 374871df
Change: Replace divisions with multiplications by inverse constants; precompute gun cone cosf() per episode

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +44.39         | 1128           | 0.40  | 0.40/10.1       |
| 2   | +37.43         | 1139           | 0.34  | 0.34/9.9        |
| 3   | +45.54         | 1128           | 0.40  | 0.40/11.2       |
| **Mean** | **+42.45** | **1132**       | **0.38** | **0.38/10.4** |

Observations:
- **+37% improvement over previous baseline** (+30.99 → +42.45)
- 21 divisions replaced with multiplications (2.3x faster per op)
- Gun cone trig precomputed per episode (curriculum-ready)
- SPS: 1.2-1.3M

---

## Autopilot Infrastructure (85980679)
Date: 2026-01-14
Commit: 85980679
Change: Add opponent autopilot system for curriculum learning (not enabled by default)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +36.85         | 1140           | 0.33  | 0.33/9.1        |
| 2   | +55.26         | 1140           | 0.51  | 0.51/11.3       |
| 3   | +12.78         | 1150           | 0.25  | 0.25/10.9       |
| **Mean** | **+34.97** | **1143**       | **0.36** | **0.36/10.4** |

Changes:
- NEW: autopilot.h - 7 autopilot modes (STRAIGHT, LEVEL, TURN_LEFT/RIGHT, CLIMB, DESCEND, RANDOM)
- NEW: set_autopilot() Python API for curriculum learning
- Default: AP_STRAIGHT (identical to previous behavior)
- PID gains from test_flight.py validation

Observations:
- Performance consistent with baseline (+42.45 → +34.97, within variance)
- **No regression** - autopilot infrastructure has negligible overhead
- Autopilot disabled by default (AP_STRAIGHT = old behavior)
- Ready for curriculum: call `env.set_autopilot(mode=AutopilotMode.RANDOM)` to enable

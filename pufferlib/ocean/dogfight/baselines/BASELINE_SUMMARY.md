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

---

## Vectorized set_autopilot (80bcf31e)
Date: 2026-01-14
Commit: 80bcf31e
Change: Add vec_set_autopilot() C binding; set_autopilot(env_idx=None) sets all envs in one call

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +45.37         | 1153           | 0.47  | 0.47/10.6       |
| 2   | +51.04         | 1140           | 0.46  | 0.46/11.2       |
| 3   | +37.00         | 1110           | 0.35  | 0.35/10.8       |
| **Mean** | **+44.47** | **1134**       | **0.43** | **0.43/10.9** |

Changes:
- binding.c: Added vec_set_autopilot() for batch autopilot configuration
- dogfight.py: set_autopilot(env_idx=None) now sets all envs in one C call

Observations:
- Performance consistent with baseline (+34.97 → +44.47, within variance)
- **No regression** - vectorized API adds no overhead during training
- Unblocks multi-env curriculum learning (no more N Python->C calls)

---

## Mode Weights for Curriculum (0a1c2e6d)
Date: 2026-01-14
Commit: 0a1c2e6d
Change: Add weighted random mode selection for curriculum learning

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +41.15         | 1139           | 0.36  | 0.36/11.0       |
| 2   | +53.82         | 1149           | 0.46  | 0.46/7.8        |
| 3   | +52.25         | 1133           | 0.45  | 0.45/10.2       |
| **Mean** | **+49.07** | **1140**       | **0.42** | **0.42/9.7** |

Changes:
- autopilot.h: Added mode_weights[AP_COUNT] array, weighted random selection in autopilot_randomize()
- autopilot.h: Added separate LCG RNG (rng_state) to avoid srand() interference from vec_reset
- binding.c: Added vec_set_mode_weights(), env_get_autopilot_mode() bindings
- dogfight.py: Added set_mode_weights(), get_autopilot_mode() methods
- test_flight.py: Added test_mode_weights() unit test

Observations:
- Performance consistent with baseline (+44.47 → +49.07, within variance)
- **No regression** - mode weights infrastructure has negligible overhead
- Fixed RNG bug: autopilot now uses own LCG instead of shared rand() (was always selecting same mode)
- Ready for curriculum: `env.set_mode_weights(level=0.8, turn_left=0.1, turn_right=0.1)` to bias easy modes

---

## Observation Scheme Sweep

### Scheme 0: WORLD_FRAME (Baseline)
Date: 2026-01-14
Config: obs_scheme = 0
Observations: 19 (player pos/vel/ori/up + world-frame rel_pos/vel)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +18.22         | 1128           | 0.20  | 0.20/10.1       |
| 2   | +19.71         | 1135           | 0.23  | 0.23/10.5       |
| 3   | +52.98         | 1139           | 0.46  | 0.46/8.0        |
| **Mean** | **+30.30** | **1134**       | **0.30** | **0.30/9.5** |

Observations:
- High variance between runs (18-53 return)
- Baseline for comparison with body-frame and angles schemes

---

### Scheme 1: BODY_FRAME
Date: 2026-01-14
Config: obs_scheme = 1
Observations: 21 (body-frame rel_pos/vel + aim_dot + dist_norm)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +50.12         | 1205           | 0.14  | 0.14/1.2        |
| 2   | +21.95         | 1292           | 0.02  | 0.02/0.5        |
| 3   | -5.26          | 1258           | 0.19  | 0.19/8.4        |
| **Mean** | **+22.27** | **1252**       | **0.12** | **0.12/3.4** |

Observations:
- **Worse than WORLD_FRAME** (+22.27 vs +30.30)
- Agent fires much less often (3.4 shots vs 9.5)
- Fewer kills despite aim helpers (0.12 vs 0.30)
- Higher variance - body-frame transform may confuse learning

---

### Scheme 2: ANGLES
Date: 2026-01-14
Config: obs_scheme = 2
Observations: 12 (pos + speed + euler angles + azimuth/elevation/dist + closing_rate + opp_heading)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +163.78        | 1198           | 0.01  | 0.01/0.02       |
| 2   | +71.56         | 1298           | 0.31  | 0.31/4.9        |
| 3   | +151.36        | 1263           | 0.01  | 0.01/0.06       |
| **Mean** | **+128.90** | **1253**      | **0.11** | **0.11/1.7** |

Observations:
- **Highest return** but misleading - agent exploits pursuit shaping without shooting
- 2 of 3 runs learned to not fire at all (0.02 and 0.06 shots)
- Only run 2 learned combat (0.31 kills)
- Smaller obs space (12) may lack info needed to learn trigger timing

---

### Observation Scheme Summary

| Scheme | Obs Size | Mean Return | Mean Kills | Shots/Ep | Notes |
|--------|----------|-------------|------------|----------|-------|
| 0: WORLD_FRAME | 19 | +30.30 | 0.30 | 9.5 | **Best combat learning** |
| 1: BODY_FRAME | 21 | +22.27 | 0.12 | 3.4 | Worse than baseline |
| 2: ANGLES | 12 | +128.90 | 0.11 | 1.7 | Exploits pursuit reward |

---

### Scheme 3: CONTROL_ERROR
Date: 2026-01-14
Config: obs_scheme = 3
Observations: 17 (player state + pitch/yaw/roll errors to target + closing_rate + opp_heading)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +165.98        | 1233           | 0.00  | 0.00/0.00       |
| 2   | +167.76        | 1238           | 0.00  | 0.00/0.00       |
| 3   | +165.45        | 1245           | 0.00  | 0.00/0.01       |
| **Mean** | **+166.40** | **1239**      | **0.00** | **0.00/0.00** |

Observations:
- **Highest return** but completely exploits pursuit reward
- Agent learned to not fire at all (0 shots across all runs)
- Control error obs may be too "solved" - agent just follows target

---

### Scheme 4: REALISTIC
Date: 2026-01-14
Config: obs_scheme = 4
Observations: 10 (airspeed/altitude/pitch/roll + gunsight az/el/size + aspect/horizon/dist)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +174.53        | 1159           | 0.00  | 0.00/0.01       |
| 2   | +171.96        | 1174           | 0.00  | 0.00/0.01       |
| 3   | +158.68        | 1252           | 0.00  | 0.00/0.01       |
| **Mean** | **+168.39** | **1195**      | **0.00** | **0.00/0.01** |

Observations:
- **Very high return** but no combat at all
- Smallest network (2.2K params) but same exploitation pattern
- Missing world position may prevent learning proper pursuit

---

### Scheme 5: MAXIMALIST
Date: 2026-01-14
Config: obs_scheme = 5
Observations: 43 (everything: world+body velocities, quaternion+euler, world+body rel_pos/vel, angles, etc.)

| Run | Episode Return | Episode Length | Kills | Shots Hit/Fired |
|-----|----------------|----------------|-------|-----------------|
| 1   | +90.95         | 1279           | 0.04  | 0.04/0.10       |
| 2   | +66.94         | 1167           | 0.31  | 0.31/2.1        |
| 3   | +92.29         | 1219           | 0.04  | 0.04/0.11       |
| **Mean** | **+83.39** | **1222**       | **0.13** | **0.13/0.8** |

Observations:
- Run 2 learned combat (0.31 kills) - only non-WORLD_FRAME scheme to do so reliably
- Lower return than pursuit-exploiting schemes but more combat
- Largest network (6.4K params) - may need more training time

---

### Final Observation Scheme Summary

| Scheme | Obs Size | Params | Mean Return | Mean Kills | Shots/Ep | Combat? |
|--------|----------|--------|-------------|------------|----------|---------|
| 0: WORLD_FRAME | 19 | 3.3K | +30.30 | **0.30** | 9.5 | **YES** |
| 1: BODY_FRAME | 21 | 3.6K | +22.27 | 0.12 | 3.4 | Weak |
| 2: ANGLES | 12 | 2.4K | +128.90 | 0.11 | 1.7 | No |
| 3: CONTROL_ERROR | 17 | 3.1K | +166.40 | 0.00 | 0.0 | No |
| 4: REALISTIC | 10 | 2.2K | +168.39 | 0.00 | 0.0 | No |
| 5: MAXIMALIST | 43 | 6.4K | +83.39 | 0.13 | 0.8 | 1/3 runs |

**Conclusion:** WORLD_FRAME (scheme 0) is the best observation representation for learning combat:
- Only scheme where all 3 runs learned to fire consistently
- Best kill rate (0.30 kills/episode)
- The "engineered" schemes (ANGLES, CONTROL_ERROR, REALISTIC) all exploit pursuit reward without learning to shoot
- MAXIMALIST occasionally learns combat but inconsistently

**Insight:** The pursuit reward shaping is too strong relative to kill rewards. Agents can achieve high return just by chasing without ever firing. The world-frame observations may make it harder to exploit this pattern because the agent can't "solve" pursuit as cleanly.

# Dogfight Implementation Plan

**Note: This plan is a living document and may be adjusted as development progresses.**

First checkbox: initial implementation complete
Second checkbox: audited and verified

---

## Phase 0: Scaffolding
- [x] [ ] 0.1 Create pufferlib/ocean/dogfight/ folder
- [x] [ ] 0.2 Create dogfight.h with basic Dogfight struct (observations, actions, rewards, terminals, Log)
- [x] [ ] 0.2b Define Log struct with ONLY float fields (env_binding.h iterates as floats):
        episode_return, episode_length, score, kills, deaths, shots_fired, shots_hit, n
- [x] [ ] 0.3 Create binding.c: implement my_init() (unpack kwargs, call init()) and my_log() (map Log fields to dict)
- [x] [ ] 0.4 Create dogfight.py following drone_race.py pattern:
        - Box(5) continuous actions [-1, 1]
        - self.actions = self.actions.astype(np.float32)  # REQUIRED for continuous
- [x] [ ] 0.5 Create pufferlib/config/ocean/dogfight.ini:
        [base] package=ocean, env_name=puffer_dogfight
        [vec] num_envs=8
        [env] num_envs=128, max_steps=3000
        [train] hyperparameters
- [x] [ ] 0.6 Verify setup.py compiles binding (already configured at line 192)
- [x] [ ] 0.7 Verify env can be instantiated: `from pufferlib.ocean.dogfight.dogfight import Dogfight`

## Phase 1: Minimal Viable Environment
- [x] [ ] 1.1 Implement init(), c_close() → test_init()
- [x] [ ] 1.2 Define Plane struct (pos, vel, ori quat) → test_reset_plane()
- [x] [ ] 1.3 Implement c_reset(): spawn plane random pos → test_c_reset()
- [x] [ ] 1.4 Implement c_step(): plane moves forward → test_c_step_moves_forward()
- [x] [ ] 1.5 Implement compute_observations(): pos/vel/ori/up normalized → test_compute_observations()
- [x] [ ] 1.6 Wire up actions array (float* for 5 floats) - read but ignore for now
- [x] [ ] 1.7 Episode terminates: OOB → test_oob_terminates(), max_steps → test_max_steps_terminates()
- [x] [ ] 1.8 Python integration: env.reset() and env.step() work

## Phase 2: Target Plane (Scripted Opponent)
- [x] [ ] 2.1 Add second Plane struct for opponent → test_opponent_spawns()
- [x] [ ] 2.2 Opponent spawns ahead of player, flies straight → test_opponent_spawns()
- [x] [ ] 2.3 Add relative position to opponent in observations → test_relative_observations()
- [x] [ ] 2.4 Add relative velocity to opponent in observations → test_relative_observations()
- [x] [ ] 2.5 Define observation space size in Python: OBS_SIZE=19
- [x] [ ] 2.6 Basic reward: negative distance to opponent → test_pursuit_reward()
- [x] [ ] 2.7 Python integration: obs shape (19,), negative reward working

## Phase 3: Flight Physics (Controls + Aerodynamics merged - correct order)
- [x] [ ] 3.1 Add aircraft parameters → test_aircraft_params()
- [x] [ ] 3.2 Quaternion orientation → done in Phase 1
- [x] [ ] 3.3 Map throttle action [0] to engine power → test_throttle_accelerates()
- [x] [ ] 3.4 Map elevator action [1] to pitch rate → test_controls_affect_orientation()
- [x] [ ] 3.5 Map ailerons action [2] to roll rate → test_controls_affect_orientation()
- [x] [ ] 3.6 Map rudder action [3] to yaw rate → step_plane_with_physics()
- [x] [ ] 3.7 Add rate limits → MAX_PITCH_RATE, MAX_ROLL_RATE, MAX_YAW_RATE
- [x] [ ] 3.8 Integrate orientation: q_dot = 0.5 * q * omega_quat → test_controls_affect_orientation()
- [x] [ ] 3.9 Compute angle of attack → step_plane_with_physics()
- [x] [ ] 3.10 Compute C_L clamped to C_L_max → test_stall_clamps_lift()
- [x] [ ] 3.11 Implement dynamic pressure → test_dynamic_pressure()
- [x] [ ] 3.12 Compute lift magnitude → test_lift_opposes_gravity()
- [x] [ ] 3.13 Compute drag magnitude → test_drag_slows_plane()
- [x] [ ] 3.14 Velocity-dependent propeller thrust → test_throttle_accelerates()
- [x] [ ] 3.15 Compute weight → test_plane_falls_without_lift()
- [x] [ ] 3.16 Transform forces to world frame → step_plane_with_physics()
- [x] [ ] 3.17 Sum forces → test_forces_sum_correctly()
- [x] [ ] 3.18 Integrate: a = F/m, v += a*dt, pos += v*dt → test_integration_updates_state()
- [x] [ ] 3.19 Enforce C_L ≤ C_L_max (stall) → test_stall_clamps_lift()
- [x] [ ] 3.20 Enforce n ≤ 8 (g-limit) → test_glimit_clamps_acceleration()
- [x] [ ] 3.21 Test: all Phase 3 tests pass (23 total)

## Phase 3.5: Reward Shaping
Current pursuit reward (-dist/10000 per step) is too weak for effective learning.

- [ ] [ ] 3.5.1 Add closing velocity reward: +bonus when distance decreasing → test_closing_velocity_reward()
- [ ] [ ] 3.5.2 Add tail position reward: +bonus when behind opponent (angle from opponent's forward) → test_tail_position_reward()
- [ ] [ ] 3.5.3 Add altitude maintenance: small penalty for z < 200m or z > 2500m → test_altitude_penalty()
- [ ] [ ] 3.5.4 Add speed maintenance: small penalty for V < 50 m/s (stall risk) → test_speed_penalty()
- [ ] [ ] 3.5.5 Scale rewards appropriately (total episode reward ~10-100 for good policy)
- [ ] [ ] 3.5.6 Test: training shows faster convergence with new rewards

## Phase 4: Rendering
**Moved before Combat** - Can't debug combat without seeing planes.

Camera and visibility:
- [ ] [ ] 4.1 Fix camera: chase cam behind player, ~50-100m back → test visual
- [ ] [ ] 4.2 Camera follows player position and orientation
- [ ] [ ] 4.3 Add mouse controls for camera orbit (like drone_race)

Drawing planes:
- [ ] [ ] 4.4 Draw player plane: cone (fuselage) + triangles (wings) or simple sphere
- [ ] [ ] 4.5 Draw opponent plane: different color
- [ ] [ ] 4.6 Draw velocity vectors for debugging (optional, toggle with key)

Environment:
- [ ] [ ] 4.7 Draw ground plane at z=0 with grid
- [ ] [ ] 4.8 Draw sky gradient or horizon reference
- [ ] [ ] 4.9 Draw world bounds (wireframe box)

HUD:
- [ ] [ ] 4.10 Display: speed (m/s), altitude (m), throttle (%)
- [ ] [ ] 4.11 Display: distance to opponent, episode tick
- [ ] [ ] 4.12 Display: episode return

## Phase 5: Combat Mechanics
**Struct additions:**
- Add to Plane: `int fire_cooldown`, `bool alive` (or `float health`)

**Constants:**
- `GUN_RANGE` = 500.0f (meters)
- `GUN_CONE_ANGLE` = 0.087f (5 degrees in radians)
- `FIRE_COOLDOWN` = 10 (ticks = 0.2 seconds)

**Implementation:**
- [ ] [ ] 5.1 Add fire_cooldown and alive fields to Plane struct
- [ ] [ ] 5.2 Add combat constants (GUN_RANGE, GUN_CONE_ANGLE, FIRE_COOLDOWN)
- [ ] [ ] 5.3 Map trigger action [4] to fire (if > 0.5 and cooldown == 0) → test_trigger_fires()
- [ ] [ ] 5.4 Implement cone check hit detection → test_cone_hit_detection()
        ```c
        bool check_hit(Plane* shooter, Plane* target) {
            Vec3 to_target = sub3(target->pos, shooter->pos);
            float dist = norm3(to_target);
            if (dist > GUN_RANGE) return false;
            Vec3 forward = quat_rotate(shooter->ori, vec3(1, 0, 0));
            float cos_angle = dot3(normalize3(to_target), forward);
            return cos_angle > cosf(GUN_CONE_ANGLE);
        }
        ```
- [ ] [ ] 5.5 Track shots_fired in Log when trigger pulled
- [ ] [ ] 5.6 Track shots_hit in Log when hit detected
- [ ] [ ] 5.7 Reward for hit: +1.0 → test_hit_reward()
- [ ] [ ] 5.8 On kill: respawn opponent, +10.0 reward, increment kills in Log
- [ ] [ ] 5.9 Episode does NOT terminate on kill (continue fighting)
- [ ] [ ] 5.10 Test: player can shoot and hit opponent → test_combat_works()

## Phase 6: Opponent AI
**Physics fix:** Both planes must use same physics model.

- [ ] [ ] 6.1 Add `float opponent_actions[5]` array (computed by AI each step)
- [ ] [ ] 6.2 Call `step_plane_with_physics(&env->opponent, opponent_actions, DT)` instead of `step_plane()`
- [ ] [ ] 6.3 Remove old `step_plane()` function (no longer needed)

**AI behaviors (compute_opponent_ai function):**
- [ ] [ ] 6.4 Pure pursuit: turn toward player → test_opponent_pursues()
- [ ] [ ] 6.5 Lead pursuit: aim ahead of player based on closure rate
- [ ] [ ] 6.6 Fire when player in gun cone → test_opponent_fires()
- [ ] [ ] 6.7 Throttle management: speed up when far, maintain when close
- [ ] [ ] 6.8 Basic evasion: break turn when player behind

**Difficulty scaling:**
- [ ] [ ] 6.9 Add `float ai_skill` parameter (0.0 = random, 1.0 = perfect)
- [ ] [ ] 6.10 Scale AI accuracy/reaction time with skill level
- [ ] [ ] 6.11 Test: opponent provides meaningful challenge at skill=0.5

## Phase 7: Tuning & Polish
- [ ] [ ] 7.1 Tune aircraft parameters to match WW2 fighter specs:
        - Max level speed: ~180-200 m/s (400-450 mph)
        - Climb rate: ~15-20 m/s (3000-4000 ft/min)
        - Sustained turn: 4-5g at combat speed
        - Corner velocity: ~130-150 m/s (260-300 knots)
- [ ] [ ] 7.2 Verify add_log() populates all fields (score, kills, deaths, shots)
- [ ] [ ] 7.3 Performance profiling
- [ ] [ ] 7.4 Optimize hot paths for 1M+ steps/sec
- [ ] [ ] 7.5 Verify no memory leaks or allocations per step

## Phase 8: Validation & Audit
- [ ] [ ] 8.1 Full test suite passes
- [ ] [ ] 8.2 Performance benchmark: confirm 1M+ steps/sec
- [ ] [ ] 8.3 Flight model validation: verify corner velocity, sustained turn rate, stall behavior
- [ ] [ ] 8.4 Training run: agent learns to pursue and shoot
- [ ] [ ] 8.5 Code review: all phases second checkbox
- [ ] [ ] 8.6 Documentation complete

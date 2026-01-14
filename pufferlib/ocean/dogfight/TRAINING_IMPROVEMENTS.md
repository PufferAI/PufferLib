# Dogfight Training Improvements

Analysis and recommendations for improving agent training before implementing opponent AI.

**Date**: 2026-01-13
**Current Performance**: Phase 5 baseline - +23.44 mean return, 0.19 kills/episode, ~1.6% accuracy

---

## Problem Analysis

### Current Target Behavior

From `step_plane()` (dogfight.h:317-324):
```c
void step_plane(Plane *p, float dt) {
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    float speed = norm3(p->vel);
    if (speed < 1.0f) speed = 80.0f;
    p->vel = mul3(forward, speed);
    p->pos = add3(p->pos, mul3(p->vel, dt));
}
```

From `respawn_opponent()` (dogfight.h:340-352):
```c
Vec3 vel = vec3(80, 0, 0);  // Always flies +X direction
reset_plane(&env->opponent, opp_pos, vel);
```

**Result**: Target flies straight at 80 m/s, always in +X direction, forever. Never turns, never changes altitude. Orientation quaternion stays at identity.

### Why Training Is Hard

| Issue | Impact |
|-------|--------|
| Target always flies +X | If player spawns heading different direction, target flies away |
| No orientation variation | Target always faces +X regardless of spawn position |
| Constant speed = easy overshoot | Player accelerates to catch up, overshoots, target keeps going |
| 5° gun cone at 200m = ~17m radius | Very precise aiming required |
| No aiming reward | Agent gets no feedback until actual hit |

### Current Results Breakdown

- **0.19 kills/episode** - less than 1 kill per 5 episodes
- **~12 shots/episode** - agent learned to fire
- **1.6% accuracy** - agent did NOT learn to aim

The agent learned pursuit and firing, but not aiming.

---

## Improvement Recommendations

### Priority 1: Fix Target Spawn Direction

**Problem**: Target always flies +X regardless of where player is facing.

**Solution A - Match player direction**:
```c
void respawn_opponent(Dogfight *env) {
    Plane *p = &env->player;
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));

    Vec3 opp_pos = vec3(
        p->pos.x + fwd.x * rndf(300, 600) + rndf(-100, 100),
        p->pos.y + fwd.y * rndf(300, 600) + rndf(-100, 100),
        clampf(p->pos.z + rndf(-100, 100), 200, 2500)
    );

    // KEY CHANGE: Opponent flies same direction as player
    Vec3 vel = mul3(fwd, 80.0f);
    reset_plane(&env->opponent, opp_pos, vel);
    env->opponent.ori = p->ori;  // Match orientation too
}
```

### Priority 2: Add Aiming Reward

**Problem**: No feedback on aim quality until actual hit.

**Solution**: Reward for having target near gun cone:
```c
// In c_step(), after pursuit rewards:
Vec3 to_opp = sub3(o->pos, p->pos);
float dist = norm3(to_opp);
Vec3 to_opp_norm = normalize3(to_opp);
Vec3 player_fwd = quat_rotate(p->ori, vec3(1, 0, 0));
float aim_dot = dot3(to_opp_norm, player_fwd);  // 1.0 = perfect aim

// Reward for tracking (within 2x gun cone and in range)
if (aim_dot > cosf(GUN_CONE_ANGLE * 2) && dist < GUN_RANGE) {
    reward += 0.05f;  // Small continuous reward for good tracking
}

// Bonus for very close aim (within gun cone but didn't fire)
if (aim_dot > cosf(GUN_CONE_ANGLE) && dist < GUN_RANGE) {
    reward += 0.1f;  // Stronger reward for firing solution
}
```

### Priority 3: Wider Gun Cone (Training Wheels)

**Current**: 5° (0.087 rad) - realistic but hard
**Proposed**: Start with 10-15° for initial training

```c
// Option: Make gun cone a parameter instead of constant
// In Dogfight struct:
float gun_cone_angle;  // Set via config

// Or just widen temporarily:
#define GUN_CONE_ANGLE 0.175f  // ~10 degrees
```

### Priority 4: Target Behavior Modes

Add different target behaviors for curriculum:

```c
typedef enum {
    TARGET_STRAIGHT = 0,    // Current: flies straight
    TARGET_CIRCLE = 1,      // Constant gentle turn
    TARGET_WEAVE = 2,       // Sinusoidal lateral movement
    TARGET_RANDOM = 3       // Occasional random turns
} TargetMode;

void step_plane(Plane *p, float dt, TargetMode mode) {
    switch (mode) {
        case TARGET_STRAIGHT:
            // Current behavior
            break;

        case TARGET_CIRCLE:
            // Constant turn rate
            float turn_rate = 0.3f;  // rad/s, ~17 deg/s
            Quat turn = quat_from_axis_angle(vec3(0, 0, 1), turn_rate * dt);
            p->ori = quat_mul(turn, p->ori);
            quat_normalize(&p->ori);
            break;

        case TARGET_WEAVE:
            // Sinusoidal yaw
            static float phase = 0;
            phase += dt;
            float yaw_rate = 0.5f * sinf(phase * 0.5f);
            Quat weave = quat_from_axis_angle(vec3(0, 0, 1), yaw_rate * dt);
            p->ori = quat_mul(weave, p->ori);
            quat_normalize(&p->ori);
            break;
    }

    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    p->vel = mul3(forward, 80.0f);
    p->pos = add3(p->pos, mul3(p->vel, dt));
}
```

### Priority 5: Better Observations for Aiming

Current observations (19 total):
- Player state: pos(3), vel(3), ori(4), up(3) = 13
- Relative: pos(3), vel(3) = 6

**Missing**: Direct aim information

**Add**:
```c
// After existing observations:

// Aim dot product (1.0 = perfect aim, -1.0 = facing away)
Vec3 to_opp_norm = normalize3(sub3(o->pos, p->pos));
Vec3 player_fwd = quat_rotate(p->ori, vec3(1, 0, 0));
float aim_dot = dot3(to_opp_norm, player_fwd);
env->observations[i++] = aim_dot;

// Distance normalized by gun range (0-1 = in range, >1 = out of range)
float dist = norm3(sub3(o->pos, p->pos));
env->observations[i++] = dist / GUN_RANGE;

// Update OBS_SIZE from 19 to 21
```

---

## Curriculum Learning Plan

| Level | Target Behavior | Speed | Gun Cone | Spawn Distance |
|-------|----------------|-------|----------|----------------|
| 1 | Stationary | 0 m/s | 15° | 200-300m |
| 2 | Slow straight | 40 m/s | 12° | 200-400m |
| 3 | Medium straight | 80 m/s | 10° | 300-500m |
| 4 | Gentle circles | 80 m/s | 7° | 300-600m |
| 5 | Variable | 80 m/s | 5° | 300-600m |

---

## Implementation Order

1. **Quick wins** (do first):
   - [x] Fix opponent spawn to match player direction → **FAILED, REVERTED** (made things worse)
   - [x] Add aiming reward → **SUCCESS** (+58% return, +89% kills, +125% accuracy)
   - [x] Run benchmark to compare

2. **If still struggling**:
   - [ ] Widen gun cone temporarily
   - [ ] Add aim_dot and distance observations
   - [ ] Run benchmark

3. **For polish**:
   - [ ] Add target behavior modes
   - [ ] Implement curriculum

---

## Debug Tools

### DEBUG flag in dogfight.h
Set `#define DEBUG 1` at the top of dogfight.h to enable verbose per-step logging:
- Actions (throttle, elevator, ailerons, rudder, trigger)
- Physics (speed, AoA, lift, drag, thrust, g-force)
- Target state (speed, position, direction)
- Reward breakdown (each component)
- Combat (aim angle, distance, in_cone, in_range)

### Python sanity tests
Run `python pufferlib/ocean/dogfight/test_flight.py` to verify physics:
- Full throttle straight flight → ~150 m/s max
- Stall speed → ~50 m/s
- Climb rate → ~16 m/s
- Glide L/D → ~14.7
- Turn tests → 30° and 60° bank with PID control

---

## Test Ideas

```c
void test_opponent_spawns_same_direction() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Player and opponent should be flying roughly same direction
    Vec3 player_fwd = quat_rotate(env.player.ori, vec3(1, 0, 0));
    Vec3 opp_fwd = quat_rotate(env.opponent.ori, vec3(1, 0, 0));
    float alignment = dot3(player_fwd, opp_fwd);

    assert(alignment > 0.9f);  // Should be nearly parallel
}

void test_aiming_reward() {
    Dogfight env = make_env(1000);
    c_reset(&env);

    // Place player aimed directly at opponent
    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat(1, 0, 0, 0);
    env.opponent.pos = vec3(200, 0, 500);  // Directly ahead

    c_step(&env);
    float reward_aimed = env.rewards[0];

    // Place player aimed away
    c_reset(&env);
    env.player.pos = vec3(0, 0, 500);
    env.player.ori = quat_from_axis_angle(vec3(0, 0, 1), PI / 2);  // 90° off
    env.opponent.pos = vec3(200, 0, 500);

    c_step(&env);
    float reward_not_aimed = env.rewards[0];

    assert(reward_aimed > reward_not_aimed);
}
```

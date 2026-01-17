# Observation Space Experiments

Design doc for systematic observation scheme comparison.

**Goal**: Empirically determine the best observation representation for dogfight learning.

**Method**: PufferLib sweep across observation schemes, all else constant.

---

## Observation Scheme Candidates

### Scheme 0: CURRENT (Baseline)
World-frame Cartesian.

```
Player: pos(3), vel(3), quat(4), up(3) = 13
Relative: pos(3), vel(3) = 6
Total: 19
```

Issues:
- Rel pos/vel in world frame - agent must learn quaternion rotation
- No direct aiming info

---

### Scheme 1: BODY_FRAME
Transform relative quantities to player body frame.

```
Player: pos(3), vel_body(3), quat(4), up(3) = 13
Relative: pos_body(3), vel_body(3) = 6
Aiming: aim_dot(1), dist_norm(1) = 2
Total: 21
```

Computation:
```c
Quat q_inv = quat_inverse(p->ori);
Vec3 rel_pos_body = quat_rotate(q_inv, sub3(o->pos, p->pos));
Vec3 rel_vel_body = quat_rotate(q_inv, sub3(o->vel, p->vel));
Vec3 vel_body = quat_rotate(q_inv, p->vel);
```

Benefit: `rel_pos_body.x > 0` always means "ahead of me"

---

### Scheme 2: ANGLES
Spherical coordinates - azimuth, elevation, distance.

```
Player: pos(3), speed(1), pitch(1), roll(1), yaw(1) = 7
Target: azimuth(1), elevation(1), distance(1), closing_rate(1) = 4
Opponent: heading_rel(1) = 1
Total: 13
```

Computation:
```c
// Target angles in body frame
Vec3 to_target_body = quat_rotate(q_inv, rel_pos);
float azimuth = atan2f(to_target_body.y, to_target_body.x);      // -pi to pi
float elevation = asinf(to_target_body.z / norm3(to_target_body)); // -pi/2 to pi/2
float distance = norm3(rel_pos);

// Player attitude (Euler from quaternion)
// pitch = asin(2*(qw*qy - qz*qx))
// roll = atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
// yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
```

Benefit: Directly answers "how far off am I pointing?"
Risk: Discontinuities at ±180° azimuth, ±90° elevation

---

### Scheme 3: CONTROL_ERROR
What control inputs would point at target?

```
Player: pos(3), speed(1), quat(4), up(3) = 11
Aiming: pitch_error(1), yaw_error(1), roll_to_turn(1), dist(1) = 4
Target: closing_rate(1), opp_heading_rel(1) = 2
Total: 17
```

Computation:
```c
// Error = angle from nose to target
Vec3 to_target_body = quat_rotate(q_inv, rel_pos);
Vec3 to_target_norm = normalize3(to_target_body);

// Pitch error: positive = need to pitch up
float pitch_error = asinf(to_target_norm.z);

// Yaw error: positive = need to yaw right
float yaw_error = atan2f(to_target_norm.y, to_target_norm.x);

// Roll to align turn: if target is right, roll right helps
float roll_to_turn = ... // bank angle that would help turn toward target
```

Benefit: Minimal transformation from observation to action
Risk: May lose situational awareness info

---

### Scheme 4: REALISTIC (Cockpit Instruments)
Only what a WW2 pilot could actually see/read.

```
Instruments: airspeed(1), altitude(1), pitch(1), roll(1) = 4
Gunsight: target_az(1), target_el(1), target_size(1) = 3
Visual: target_aspect(1), horizon_visible(1) = 2
Total: 10
```

Benefit: Most constrained - if this works, simpler is better
Risk: May lack critical information

---

### Scheme 5: MAXIMALIST
Everything potentially useful, let network sort it out.

```
Player: pos(3), vel_world(3), vel_body(3), quat(4), up(3),
        speed(1), pitch(1), roll(1), yaw(1) = 20
Target: rel_pos_world(3), rel_pos_body(3), rel_vel_world(3), rel_vel_body(3),
        azimuth(1), elevation(1), distance(1), aim_dot(1), closing_rate(1) = 17
Opponent: opp_fwd_world(3), opp_fwd_body(3) = 6
Total: 43
```

Benefit: Network has all possible information
Risk: Larger network needed, slower training, redundancy

---

## Implementation Strategy

### Option A: Compile-Time Schemes
```c
#define OBS_SCHEME 1  // 0=CURRENT, 1=BODY_FRAME, 2=ANGLES, etc.

#if OBS_SCHEME == 0
#define OBS_SIZE 19
#elif OBS_SCHEME == 1
#define OBS_SIZE 21
// ...
#endif

void compute_observations(Dogfight *env) {
#if OBS_SCHEME == 0
    // Current implementation
#elif OBS_SCHEME == 1
    // Body frame implementation
#endif
}
```

Pro: No runtime overhead
Con: Requires recompilation for each scheme

### Option B: Runtime Config
```c
typedef enum {
    OBS_CURRENT = 0,
    OBS_BODY_FRAME,
    OBS_ANGLES,
    OBS_CONTROL_ERROR,
    OBS_REALISTIC,
    OBS_MAXIMALIST
} ObsScheme;

// In Dogfight struct
ObsScheme obs_scheme;
int obs_size;  // Set at init based on scheme
```

Pro: Single binary, config-driven sweeps
Con: Slight runtime overhead, more complex code

### Recommendation: Option B (Runtime Config)

Enables PufferLib sweep integration without recompilation.

---

## PufferLib Sweep Integration

### INI Config Addition
```ini
[dogfight]
obs_scheme = 1  # 0-5, maps to enum
```

### Sweep Config
```yaml
# sweeps/dogfight_obs.yaml
base_config: puffer_dogfight
sweep:
  obs_scheme: [0, 1, 2, 3, 4, 5]

# Or focused comparison:
sweep:
  obs_scheme: [0, 1, 2]  # Current vs Body Frame vs Angles
```

### Python Integration
```python
# dogfight.py
def __init__(self, ..., obs_scheme=0):
    self.obs_scheme = obs_scheme
    self.obs_size = OBS_SIZES[obs_scheme]  # Lookup table

    self.observation_space = gymnasium.spaces.Box(
        low=-1, high=1,
        shape=(self.obs_size,),
        dtype=np.float32
    )
```

---

## Experimental Protocol

### Phase 1: Focused Comparison (Recommended First)
Compare 3 most promising schemes:
- Scheme 0 (CURRENT) - baseline
- Scheme 1 (BODY_FRAME) - expected best
- Scheme 2 (ANGLES) - alternative representation

Run: 3 schemes × 3 seeds × 100M steps = 9 runs

### Phase 2: Extended Comparison
If Phase 1 inconclusive, add:
- Scheme 3 (CONTROL_ERROR)
- Scheme 5 (MAXIMALIST)

### Phase 3: Ablation Studies
On best scheme, ablate individual observations:
- Remove aim_dot: does it hurt?
- Remove opponent heading: does it matter?
- etc.

---

## Metrics

Primary:
- Episode return (mean, std over 3 seeds)
- Kills per episode
- Accuracy (hits/shots)

Secondary:
- Learning speed (steps to reach threshold performance)
- Final policy variance
- Behavioral analysis (does agent pursue? aim? fire appropriately?)

---

## Expected Outcomes (Pre-Experiment Hypotheses)

| Scheme | Hypothesis |
|--------|------------|
| 0 CURRENT | Baseline - agent struggles with aiming |
| 1 BODY_FRAME | **Best** - easiest for network to learn spatial relationships |
| 2 ANGLES | Good for aiming, may have discontinuity issues |
| 3 CONTROL_ERROR | Fast initial learning, may plateau |
| 4 REALISTIC | Insufficient information |
| 5 MAXIMALIST | Works but slower, needs bigger network |

---

## Actual Results (2026-01-14)

| Scheme | Obs | Return | Kills | Combat? | Notes |
|--------|-----|--------|-------|---------|-------|
| 0 WORLD_FRAME | 19 | +30.30 | **0.30** | **YES** | Best for combat |
| 1 BODY_FRAME | 21 | +22.27 | 0.12 | Weak | Worse than baseline |
| 2 ANGLES | 12 | +128.90 | 0.11 | No | Exploits pursuit reward |
| 3 CONTROL_ERROR | 17 | +166.40 | 0.00 | No | Exploits pursuit reward |
| 4 REALISTIC | 10 | +168.39 | 0.00 | No | Exploits pursuit reward |
| 5 MAXIMALIST | 43 | +83.39 | 0.13 | 1/3 | High variance |

**Key Finding:** WORLD_FRAME (scheme 0) is the only scheme where all 3 runs consistently learned combat. The "engineered" observation schemes (2-4) achieved very high return by exploiting pursuit reward shaping without learning to fire. The hypothesis that BODY_FRAME would be best was **wrong** - the network actually learns combat better with world-frame observations.

**Insight:** The pursuit reward shaping is too strong. Agents can maximize return by chasing without ever shooting. World-frame observations may make this exploitation harder because the spatial relationships aren't pre-computed for the agent.

---

## Implementation Checklist

- [x] Add `obs_scheme` to Dogfight struct
- [x] Add `obs_scheme` to INI config parsing
- [x] Implement OBS_SIZE lookup table
- [x] Implement compute_observations_scheme_X() for each scheme (all 6)
- [x] Update Python observation_space based on scheme
- [x] Create sweep config file
- [x] Run Phase 1 experiments (all 6 schemes, 3 runs each)
- [x] Analyze results
- [x] Document findings in BASELINE_SUMMARY.md

---

## Code Sketch: Body Frame Implementation

```c
void compute_observations_body_frame(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    // Inverse quaternion for world->body transforms
    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player state
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));

    // Relative state in body frame
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_vel = sub3(o->vel, p->vel);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    Vec3 rel_vel_body = quat_rotate(q_inv, rel_vel);

    // Aiming info
    float dist = norm3(rel_pos);
    Vec3 fwd = vec3(1, 0, 0);  // In body frame, forward is always +X
    Vec3 to_target = normalize3(rel_pos_body);
    float aim_dot = dot3(to_target, fwd);  // 1.0 = perfect aim

    int i = 0;
    // Player position (world frame - needed for bounds awareness)
    env->observations[i++] = p->pos.x * INV_WORLD_HALF_X;
    env->observations[i++] = p->pos.y * INV_WORLD_HALF_Y;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;

    // Player velocity (body frame)
    env->observations[i++] = vel_body.x * INV_MAX_SPEED;  // Forward speed
    env->observations[i++] = vel_body.y * INV_MAX_SPEED;  // Sideslip
    env->observations[i++] = vel_body.z * INV_MAX_SPEED;  // Climb rate

    // Player orientation
    env->observations[i++] = p->ori.w;
    env->observations[i++] = p->ori.x;
    env->observations[i++] = p->ori.y;
    env->observations[i++] = p->ori.z;

    // Player up vector (world frame)
    env->observations[i++] = up.x;
    env->observations[i++] = up.y;
    env->observations[i++] = up.z;

    // Relative position (body frame) - THE KEY CHANGE
    env->observations[i++] = rel_pos_body.x * INV_WORLD_HALF_X;
    env->observations[i++] = rel_pos_body.y * INV_WORLD_HALF_Y;
    env->observations[i++] = rel_pos_body.z * INV_WORLD_MAX_Z;

    // Relative velocity (body frame)
    env->observations[i++] = rel_vel_body.x * INV_MAX_SPEED;
    env->observations[i++] = rel_vel_body.y * INV_MAX_SPEED;
    env->observations[i++] = rel_vel_body.z * INV_MAX_SPEED;

    // Aiming helpers
    env->observations[i++] = aim_dot;  // -1 to 1, 1 = perfect aim
    env->observations[i++] = dist / GUN_RANGE;  // ~0-4, 1 = at gun range

    // OBS_SIZE = 21 for this scheme
}
```

---

## Code Sketch: Angles Implementation

```c
void compute_observations_angles(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles from quaternion
    float pitch = asinf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));
    float yaw = atan2f(2.0f * (p->ori.w * p->ori.z + p->ori.x * p->ori.y),
                       1.0f - 2.0f * (p->ori.y * p->ori.y + p->ori.z * p->ori.z));

    // Target in body frame -> spherical
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float azimuth = atan2f(rel_pos_body.y, rel_pos_body.x);  // -pi to pi
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float elevation = atan2f(rel_pos_body.z, r_horiz);  // -pi/2 to pi/2

    // Closing rate
    Vec3 rel_vel = sub3(p->vel, o->vel);  // Note: p - o for closing
    float closing_rate = dot3(rel_vel, normalize3(rel_pos));

    // Opponent heading relative to player
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 opp_fwd_body = quat_rotate(q_inv, opp_fwd);
    float opp_heading_rel = atan2f(opp_fwd_body.y, opp_fwd_body.x);

    int i = 0;
    // Player state
    env->observations[i++] = p->pos.x * INV_WORLD_HALF_X;
    env->observations[i++] = p->pos.y * INV_WORLD_HALF_Y;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = norm3(p->vel) * INV_MAX_SPEED;  // Speed scalar
    env->observations[i++] = pitch / PI;      // -0.5 to 0.5
    env->observations[i++] = roll / PI;       // -1 to 1
    env->observations[i++] = yaw / PI;        // -1 to 1 (or omit - world symmetric)

    // Target angles
    env->observations[i++] = azimuth / PI;    // -1 to 1
    env->observations[i++] = elevation / (PI * 0.5f);  // -1 to 1
    env->observations[i++] = dist / (GUN_RANGE * 2.0f);  // ~0-2
    env->observations[i++] = closing_rate * INV_MAX_SPEED;

    // Opponent info
    env->observations[i++] = opp_heading_rel / PI;  // -1 to 1

    // OBS_SIZE = 12 for this scheme
}
```

---

## Notes

- Angle normalization: divide by PI to get [-1, 1] range
- Discontinuity handling: atan2 handles ±180° gracefully, but crossing still causes jump
- Alternative: use sin/cos of angles instead of raw angles (no discontinuity)
  - `sin(azimuth), cos(azimuth)` instead of `azimuth`
  - Doubles the observation count but removes discontinuities

---

## References

- drone_race.h: Body-frame transform pattern (lines 79-85)
- TRAINING_IMPROVEMENTS.md: Original observation improvement ideas
- **REALISTIC_SCHEMES.md**: Proposed schemes 6-9 (MINIMAL, GUNSIGHT, ENERGY, PLUS) for Aces High III transfer - tests minimal observations needed for dogfighting
- aceshigh/DLL_SPEC.md: How observation choice affects DLL data requirements
- PufferLib sweep docs: (link to docs if available)

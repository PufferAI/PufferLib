# Elevator Inversion Bug Investigation

**Date:** 2026-01-16
**Status:** Suspected bug, needs verification

## Summary

Empirical testing suggests the elevator control may be **inverted** from what the code comments claim.

## Evidence

### Code Comment (flightlib.h:220)
```c
float pitch_rate = actions[1] * MAX_PITCH_RATE;  // rad/s, + = nose up
```

### Empirical Test Results

**Test 1: Wings level (identity quaternion), flying East**
```
BEFORE: nose = (1.00, 0.00, 0.00) pointing East
AFTER positive elevator (+1.0) for 0.5s:
  nose = (0.32, 0.00, -0.95)  ← fwd_z NEGATIVE = nose DOWN!
```

**Expected:** Positive elevator = pull back = nose UP
**Actual:** Positive elevator = nose DOWN

### Test 2: Knife-edge (rolled 90° right, canopy pointing South)
```
Canopy (body +Z) = South (-Y world)
Positive elevator: nose moved toward NORTH (+Y)
Negative elevator: nose moved toward SOUTH (-Y)
```

Nose should move toward canopy direction when "pulling back". But positive elevator moves nose AWAY from canopy (toward belly).

## Possible Explanations

1. **Bug in quaternion kinematics** - The formula `q_dot = q * omega` might need to be `q_dot = omega * q` or have a sign flip somewhere

2. **Body frame convention mismatch** - The omega_body vector might use a different axis convention than expected

3. **Comment is simply wrong** - The code works as intended but the comment is backwards

4. **Right-hand rule interpretation** - Positive rotation about body Y might be defined opposite to standard aerospace convention

## Impact

If the elevator is inverted:
- The `test_pitch_direction` test in `test_flight.py` may be wrong
- RL agents trained on this might have learned inverted controls
- The "penalty_neg_g" (penalizing negative elevator) might be penalizing the WRONG action

## Quaternion Kinematics Analysis

The code uses:
```c
Vec3 omega_body = vec3(roll_rate, pitch_rate, yaw_rate);
Quat omega_quat = quat(0, omega_body.x, omega_body.y, omega_body.z);
Quat q_dot = quat_mul(p->ori, omega_quat);
```

Standard formula: `q_dot = 0.5 * q ⊗ ω_body` (body frame)

At identity orientation:
- Body Y axis = +Y world (North)
- Positive pitch = rotation about +Y
- Right-hand rule: thumb North, fingers curl +X→+Z
- So nose (+X) should go toward +Z (UP)

But empirically nose goes DOWN. This suggests either:
1. The multiplication order is wrong
2. There's a sign error in the quaternion multiplication
3. The omega_quat construction has wrong signs

## Recommended Actions

1. **Verify with rendered visualization** - Watch the plane pitch with render mode on
2. **Check quaternion multiplication** - Compare against reference implementation
3. **Test all control axes** - Roll and yaw might also be affected
4. **Review training results** - See if agents have learned compensating behaviors

## Related Files

- `flightlib.h` - Physics implementation (lines 201-236)
- `test_flight.py` - `test_pitch_direction()` may need updating
- `dogfight.h` - Reward calculations that depend on elevator sign

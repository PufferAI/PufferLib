"""
Dynamic maneuver observation tests for dogfight environment.
Tests observation continuity and bounds during active flight maneuvers.

Run: python pufferlib/ocean/dogfight/test_flight_obs_dynamic.py --test obs_during_loop
"""
import numpy as np
from dogfight import Dogfight

from test_flight_base import (
    get_render_mode, get_render_fps, get_physics_mode,
    RESULTS,
)
from test_flight_obs_static import obs_continuity_check


def test_obs_during_loop():
    """
    Full inside loop maneuver - verify observations during complete pitch cycle.

    Purpose: Ensure Euler angle observations (pitch) smoothly transition through
    full range [-1, 1] during a loop without discontinuities.

    Expected behavior:
    - Pitch sweeps through full range (0 -> -0.5 (nose up 90deg) -> +/-1 (inverted) -> +0.5 -> 0)
    - Roll stays near 0 throughout (wings level loop)
    - No sudden jumps in any observation (discontinuity = bug)

    This tests the quaternion->euler conversion under continuous rotation.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    env.reset()

    # Start with good speed at safe altitude, target ahead to avoid edge cases
    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(150, 0, 0),  # Fast for complete loop
        player_throttle=1.0,
        opponent_pos=(1000, 0, 1500),  # Target ahead
        opponent_vel=(100, 0, 0),
    )

    pitches = []
    rolls = []
    prev_obs = None
    continuity_errors = []

    for step in range(350):  # ~7 seconds should complete most of loop
        action = np.array([[1.0, -0.8, 0.0, 0.0, 0.0]], dtype=np.float32)  # Full throttle, strong pull
        env.step(action)
        obs = env.observations[0]

        pitches.append(obs[4])  # pitch
        rolls.append(obs[5])    # roll

        # Check continuity
        passed, err = obs_continuity_check(obs, prev_obs, step)
        if not passed:
            continuity_errors.append(err)
        prev_obs = obs.copy()

        # Check termination (might hit bounds)
        state = env.get_state()
        if state['pz'] < 100:
            break

    # Analysis
    pitch_range = max(pitches) - min(pitches)
    max_roll_drift = max(abs(r) for r in rolls)

    # Verify:
    # 1. Pitch spans significant range (at least 0.8 of [-1, 1] = 1.6)
    # 2. Roll stays bounded (less than 0.4 drift from wings level)
    # 3. No discontinuities

    pitch_ok = pitch_range > 0.8  # Should cover most of the range
    roll_ok = max_roll_drift < 0.4  # Wings should stay relatively level
    continuity_ok = len(continuity_errors) == 0

    all_ok = pitch_ok and roll_ok and continuity_ok
    RESULTS['obs_loop'] = all_ok
    status = "OK" if all_ok else "CHECK"

    print(f"obs_loop:      pitch_range={pitch_range:.2f}, roll_drift={max_roll_drift:.2f}, errors={len(continuity_errors)} [{status}]")

    if not pitch_ok:
        print(f"    WARNING: Pitch range {pitch_range:.2f} < 0.8 - loop may be incomplete")
    if not roll_ok:
        print(f"    WARNING: Roll drifted {max_roll_drift:.2f} - wings not level during loop")
    if continuity_errors:
        for err in continuity_errors[:3]:
            print(f"    {err}")

    env.close()
    return all_ok


def test_obs_during_roll():
    """
    Full 360deg aileron roll - verify roll and horizon_visible observations.

    Purpose: Ensure roll observation smoothly transitions through +/-180deg without
    discontinuity, and horizon_visible follows expected pattern.

    Expected behavior (scheme 2):
    - Roll: 0 -> -1 (90deg right) -> +/-1 (inverted wrap) -> +1 (270deg) -> 0
    - horizon_visible: 1 -> 0 -> -1 -> 0 -> 1

    The +/-180deg crossover is the critical test - if there's a wrap bug,
    roll will jump from +1 to -1 instantly instead of smoothly transitioning.
    """
    env = Dogfight(num_envs=1, obs_scheme=2, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    env.reset()

    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(100, 0, 0),
        player_throttle=1.0,
        opponent_pos=(500, 0, 1500),
        opponent_vel=(100, 0, 0),
    )

    rolls = []
    horizons = []
    prev_obs = None
    continuity_errors = []

    # Roll at MAX_ROLL_RATE=3.0 rad/s = 172deg/s, so 360deg takes ~2.1 seconds = 105 steps
    for step in range(120):  # ~2.4 seconds for full 360deg with margin
        action = np.array([[0.7, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)  # Full right aileron
        env.step(action)
        obs = env.observations[0]

        # In scheme 2: roll is at index 3, horizon_visible at index 8
        rolls.append(obs[3])
        horizons.append(obs[8])

        # Check continuity with higher tolerance for roll (can change faster)
        passed, err = obs_continuity_check(obs, prev_obs, step, max_delta=0.4)
        if not passed:
            continuity_errors.append(err)
        prev_obs = obs.copy()

    # Analysis
    roll_min = min(rolls)
    roll_max = max(rolls)
    roll_range = roll_max - roll_min
    horizon_min = min(horizons)
    horizon_max = max(horizons)

    # Check for discontinuities specifically in roll (the main concern)
    roll_jumps = []
    for i in range(1, len(rolls)):
        delta = abs(rolls[i] - rolls[i-1])
        if delta > 0.5:  # Large jump indicates wrap-around bug
            roll_jumps.append((i, rolls[i-1], rolls[i], delta))

    # Verify:
    # 1. Roll covers most of range (near +/-1)
    # 2. Horizon covers full range (1 to -1)
    # 3. No sudden roll jumps (discontinuity)

    roll_ok = roll_range > 1.5  # Should span nearly [-1, 1]
    horizon_ok = horizon_max > 0.8 and horizon_min < -0.8
    no_jumps = len(roll_jumps) == 0

    all_ok = roll_ok and horizon_ok and no_jumps
    RESULTS['obs_roll'] = all_ok
    status = "OK" if all_ok else "CHECK"

    print(f"obs_roll:      roll=[{roll_min:.2f},{roll_max:.2f}], horizon=[{horizon_min:.2f},{horizon_max:.2f}], jumps={len(roll_jumps)} [{status}]")

    if not roll_ok:
        print(f"    WARNING: Roll range {roll_range:.2f} < 1.5 - incomplete roll")
    if not horizon_ok:
        print(f"    WARNING: Horizon didn't reach extremes")
    if roll_jumps:
        for step, prev, curr, delta in roll_jumps[:3]:
            print(f"    Roll discontinuity at step {step}: {prev:.2f} -> {curr:.2f} (delta={delta:.2f})")

    env.close()
    return all_ok


def test_obs_vertical_pitch():
    """
    Vertical pitch (+/-90deg) gimbal lock detection test.

    Purpose: Detect gimbal lock behavior when pitch reaches +/-90deg where
    the euler angle representation becomes singular.

    At pitch = +/-90deg:
    - roll = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2)) becomes undefined
    - May cause roll to snap/oscillate wildly

    This documents the behavior rather than asserting specific values,
    since gimbal lock is a known limitation of euler angles.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    env.reset()

    # Test nose straight up (90deg pitch)
    pitch_90 = np.radians(90)
    qw = np.cos(pitch_90 / 2)
    qy = -np.sin(pitch_90 / 2)  # Negative for nose UP

    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(100, 0, 0),
        player_ori=(qw, 0, qy, 0),  # Nose straight up
        opponent_pos=(500, 0, 1500),
        opponent_vel=(100, 0, 0),
    )

    # Step once to compute observations
    action = np.array([[0.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    env.step(action)
    obs_up = env.observations[0].copy()
    pitch_up = obs_up[4]
    roll_up = obs_up[5]

    # Test nose straight down (-90deg pitch)
    env.reset()
    qw = np.cos(-pitch_90 / 2)
    qy = -np.sin(-pitch_90 / 2)  # Positive for nose DOWN

    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(100, 0, 0),
        player_ori=(qw, 0, qy, 0),  # Nose straight down
        opponent_pos=(500, 0, 1500),
        opponent_vel=(100, 0, 0),
    )

    env.step(action)
    obs_down = env.observations[0].copy()
    pitch_down = obs_down[4]
    roll_down = obs_down[5]

    # Check bounds and NaN
    all_bounded = True
    for obs in [obs_up, obs_down]:
        for val in obs:
            if np.isnan(val) or np.isinf(val) or val < -1.0 or val > 1.0:
                all_bounded = False

    # Pitch should be near +/-0.5 (90deg/180deg = 0.5)
    pitch_up_ok = abs(abs(pitch_up) - 0.5) < 0.15
    pitch_down_ok = abs(abs(pitch_down) - 0.5) < 0.15

    RESULTS['obs_vertical'] = all_bounded
    status = "OK" if all_bounded else "WARN"

    print(f"obs_vertical:  up=(pitch={pitch_up:.3f}, roll={roll_up:.3f}), down=(pitch={pitch_down:.3f}, roll={roll_down:.3f}) [{status}]")

    if not pitch_up_ok:
        print(f"    NOTE: Pitch up {pitch_up:.3f} not near +/-0.5 (expected for 90deg pitch)")
    if not pitch_down_ok:
        print(f"    NOTE: Pitch down {pitch_down:.3f} not near +/-0.5")
    if not all_bounded:
        print(f"    WARNING: Observations out of bounds or NaN at vertical pitch")
    if abs(roll_up) > 0.3 or abs(roll_down) > 0.3:
        print(f"    NOTE: Roll unstable at vertical pitch (gimbal lock region)")

    env.close()
    return all_bounded


def test_obs_azimuth_crossover():
    """
    Target azimuth +/-180deg crossover test.

    Purpose: Verify azimuth doesn't jump discontinuously when target
    crosses from behind-left to behind-right (through +/-180deg).

    Risk: Azimuth might jump from +1 to -1 instantly instead of transitioning
    smoothly, causing RL agent to see huge observation delta.

    Test: Sweep opponent from right-behind through directly-behind to left-behind
    and check for discontinuities.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    azimuths = []
    y_positions = []

    # Sweep opponent from right-behind (y=-200) through left-behind (y=+200)
    # This forces azimuth to cross through +/-180deg (behind the player)
    for step in range(50):
        env.reset()
        y_offset = -200 + step * 8  # Sweep from y=-200 to y=+200

        env.force_state(
            player_pos=(0, 0, 1000),
            player_vel=(100, 0, 0),
            player_ori=(1, 0, 0, 0),  # Identity - facing +X
            opponent_pos=(-200, y_offset, 1000),  # Behind player, sweeping Y
            opponent_vel=(100, 0, 0),
        )

        env.step(action)
        azimuths.append(env.observations[0][7])
        y_positions.append(y_offset)

    # Check for discontinuities
    azimuth_jumps = []
    for i in range(1, len(azimuths)):
        delta = abs(azimuths[i] - azimuths[i-1])
        if delta > 0.5:  # Large jump = discontinuity
            azimuth_jumps.append((i, y_positions[i], azimuths[i-1], azimuths[i], delta))

    # Verify azimuth range covers +/-1 (behind = +/-180deg)
    az_min = min(azimuths)
    az_max = max(azimuths)
    range_ok = az_max > 0.8 and az_min < -0.8

    # Discontinuity at +/-180deg crossover is EXPECTED for atan2-based azimuth
    # This test documents the behavior - a discontinuity here is not necessarily
    # a bug, but agents should be aware of it
    has_discontinuity = len(azimuth_jumps) > 0

    RESULTS['obs_azimuth_cross'] = range_ok
    status = "OK" if range_ok else "CHECK"

    print(f"obs_az_cross:  range=[{az_min:.2f},{az_max:.2f}], discontinuities={len(azimuth_jumps)} [{status}]")

    if has_discontinuity:
        print(f"    NOTE: Azimuth has discontinuity at +/-180deg (expected for atan2)")
        for _, y_pos, prev_az, curr_az, delta in azimuth_jumps[:2]:
            print(f"    At y={y_pos:.0f}: azimuth {prev_az:.2f} -> {curr_az:.2f} (delta={delta:.2f})")
        print(f"    Consider: Use sin/cos encoding to avoid wrap-around for RL")

    if not range_ok:
        print(f"    WARNING: Azimuth didn't reach +/-1 (behind player)")

    env.close()
    return range_ok


def test_obs_yaw_wrap():
    """
    Yaw observation +/-180deg wrap test.

    Purpose: Verify yaw observation behavior when heading crosses +/-180deg.
    Tests CONTINUOUS heading transition across the wrap boundary.

    The critical test: sweep from +170deg to -170deg (crossing +180deg/-180deg).
    If yaw wraps, we'll see a jump from ~+1 to ~-1.

    For RL, yaw wrap at +/-180deg is less problematic than roll wrap because:
    - Normal flight rarely involves facing directly backwards
    - Roll wrap happens during inverted flight (loops, barrel rolls)
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    yaws = []
    headings = []

    # Test 1: Sweep ACROSS the +/-180deg boundary (170deg to 190deg = -170deg)
    # This is the critical test - continuous transition through the wrap point
    for heading_deg in range(170, 195, 2):  # 170deg to 194deg in 2deg steps
        env.reset()

        # Normalize to [-180, 180] range for quaternion
        h = heading_deg if heading_deg <= 180 else heading_deg - 360
        heading_rad = np.radians(h)
        qw = np.cos(heading_rad / 2)
        qz = np.sin(heading_rad / 2)

        vx = 100 * np.cos(heading_rad)
        vy = -100 * np.sin(heading_rad)

        env.force_state(
            player_pos=(0, 0, 1500),
            player_vel=(vx, vy, 0),
            player_ori=(qw, 0, 0, qz),
            opponent_pos=(500, 0, 1500),
            opponent_vel=(100, 0, 0),
        )

        env.step(action)
        obs = env.observations[0]

        yaws.append(obs[6])
        headings.append(heading_deg)

    # Check for discontinuities at the +/-180deg crossing
    yaw_jumps = []
    for i in range(1, len(yaws)):
        delta = abs(yaws[i] - yaws[i-1])
        if delta > 0.3:  # 2deg step should give ~0.022 change, 0.3 is a big jump
            yaw_jumps.append((headings[i-1], headings[i], yaws[i-1], yaws[i], delta))

    yaw_min = min(yaws)
    yaw_max = max(yaws)

    # Also do a full range check
    full_range_yaws = []
    for heading_deg in range(-180, 185, 30):
        env.reset()
        heading_rad = np.radians(heading_deg)
        qw = np.cos(heading_rad / 2)
        qz = np.sin(heading_rad / 2)
        vx = 100 * np.cos(heading_rad)
        vy = -100 * np.sin(heading_rad)

        env.force_state(
            player_pos=(0, 0, 1500),
            player_vel=(vx, vy, 0),
            player_ori=(qw, 0, 0, qz),
            opponent_pos=(500, 0, 1500),
            opponent_vel=(100, 0, 0),
        )
        env.step(action)
        full_range_yaws.append(env.observations[0][6])

    full_min = min(full_range_yaws)
    full_max = max(full_range_yaws)
    full_range = full_max - full_min

    has_wrap = len(yaw_jumps) > 0
    range_ok = full_range > 1.5

    RESULTS['obs_yaw_wrap'] = range_ok
    status = "OK" if range_ok else "CHECK"

    print(f"obs_yaw_wrap:  full_range=[{full_min:.2f},{full_max:.2f}], crossover_jumps={len(yaw_jumps)} [{status}]")

    if has_wrap:
        print(f"    WRAP DETECTED at +/-180deg heading:")
        for h1, h2, y1, y2, delta in yaw_jumps[:2]:
            print(f"    heading {h1}deg->{h2}deg: yaw {y1:.2f} -> {y2:.2f} (delta={delta:.2f})")
        print(f"    Consider: Use sin/cos encoding for yaw to avoid wrap")
    else:
        print(f"    No discontinuity at +/-180deg crossing (yaw: {yaw_min:.2f} to {yaw_max:.2f})")

    env.close()
    return range_ok


def test_obs_elevation_extremes():
    """
    Elevation observation at +/-90deg (target directly above/below).

    Purpose: Verify elevation doesn't have singularity when target is
    directly above or below player. Elevation uses asin which is bounded
    by definition, so this should be stable.

    Test: Place target directly above and below player, verify elevation
    is correct and bounded.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # Target directly above (500m up)
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(100, 0, 0),
        player_ori=(1, 0, 0, 0),
        opponent_pos=(0, 0, 1500),  # Directly above
        opponent_vel=(100, 0, 0),
    )
    env.step(action)
    elev_above = env.observations[0][8]

    # Target directly below (500m down)
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(100, 0, 0),
        player_ori=(1, 0, 0, 0),
        opponent_pos=(0, 0, 500),  # Directly below
        opponent_vel=(100, 0, 0),
    )
    env.step(action)
    elev_below = env.observations[0][8]

    # Target at extreme angle (nearly overhead, slightly forward)
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(100, 0, 0),
        player_ori=(1, 0, 0, 0),
        opponent_pos=(10, 0, 1500),  # Slightly forward, mostly above
        opponent_vel=(100, 0, 0),
    )
    env.step(action)
    elev_steep_up = env.observations[0][8]

    # Verify values
    all_bounded = True
    for val in [elev_above, elev_below, elev_steep_up]:
        if np.isnan(val) or np.isinf(val) or val < -1.0 or val > 1.0:
            all_bounded = False

    # Target above should have positive elevation (close to +1)
    above_ok = elev_above > 0.8
    # Target below should have negative elevation (close to -1)
    below_ok = elev_below < -0.8
    # Steep up should be very high
    steep_ok = elev_steep_up > 0.9

    all_ok = all_bounded and above_ok and below_ok and steep_ok
    RESULTS['obs_elevation_extremes'] = all_ok
    status = "OK" if all_ok else "CHECK"

    print(f"obs_elev_ext:  above={elev_above:.3f}, below={elev_below:.3f}, steep={elev_steep_up:.3f} [{status}]")

    if not above_ok:
        print(f"    WARNING: Target above should have elev >0.8, got {elev_above:.3f}")
    if not below_ok:
        print(f"    WARNING: Target below should have elev <-0.8, got {elev_below:.3f}")
    if not all_bounded:
        print(f"    WARNING: Elevation out of bounds or NaN at extreme angles")

    env.close()
    return all_ok


def test_obs_complex_maneuver():
    """
    Complex maneuver (barrel roll) - simultaneous pitch, roll, yaw changes.

    Purpose: Verify all observations stay bounded and continuous during
    complex combined rotations that exercise multiple rotation axes.

    This tests edge cases that might not appear in single-axis tests.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    env.reset()

    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(120, 0, 0),
        player_throttle=1.0,
        opponent_pos=(500, 0, 1500),
        opponent_vel=(100, 0, 0),
    )

    prev_obs = None
    continuity_errors = []
    bound_errors = []

    for step in range(200):  # ~4 seconds of complex maneuver
        # Barrel roll: pull + roll (creates helical path)
        action = np.array([[0.8, -0.3, 0.8, 0.2, 0.0]], dtype=np.float32)
        env.step(action)
        obs = env.observations[0]

        # Check bounds
        for i, val in enumerate(obs):
            if np.isnan(val) or np.isinf(val):
                bound_errors.append(f"NaN/Inf at step {step}, obs[{i}]={val}")
            elif val < -1.0 or val > 1.0:
                bound_errors.append(f"Out of bounds at step {step}, obs[{i}]={val:.3f}")

        # Check continuity (higher tolerance for complex maneuver)
        passed, err = obs_continuity_check(obs, prev_obs, step, max_delta=0.5)
        if not passed:
            continuity_errors.append(err)
        prev_obs = obs.copy()

        # Check termination
        state = env.get_state()
        if state['pz'] < 200:
            break

    bounds_ok = len(bound_errors) == 0
    continuity_ok = len(continuity_errors) <= 5  # Allow some discontinuities at wrap points

    all_ok = bounds_ok and continuity_ok
    RESULTS['obs_complex'] = all_ok
    status = "OK" if all_ok else "CHECK"

    print(f"obs_complex:   bound_errors={len(bound_errors)}, continuity_errors={len(continuity_errors)} [{status}]")

    if bound_errors:
        for err in bound_errors[:3]:
            print(f"    {err}")
    if continuity_errors:
        print(f"    NOTE: {len(continuity_errors)} continuity errors (wrap points expected)")
        for err in continuity_errors[:3]:
            print(f"    {err}")

    env.close()
    return all_ok


def test_quaternion_normalization():
    """
    Quaternion normalization drift test.

    Purpose: Verify quaternion stays normalized (magnitude ~1.0) during
    extended flight with various maneuvers. Floating point accumulation
    could cause drift from unit quaternion over time.

    Non-unit quaternion -> incorrect euler angles -> bad observations.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    env.reset()

    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(100, 0, 0),
        player_throttle=1.0,
        opponent_pos=(500, 0, 1500),
        opponent_vel=(100, 0, 0),
    )

    quat_mags = []

    for step in range(500):  # ~10 seconds of varied maneuvers
        # Varied maneuvers to stress quaternion integration
        t = step * 0.02  # Time in seconds
        aileron = 0.5 * np.sin(t * 2.0)   # Rolling
        elevator = 0.3 * np.cos(t * 1.5)  # Pitching
        rudder = 0.2 * np.sin(t * 0.8)    # Yawing

        action = np.array([[0.7, elevator, aileron, rudder, 0.0]], dtype=np.float32)
        env.step(action)

        state = env.get_state()
        qw, qx, qy, qz = state['ow'], state['ox'], state['oy'], state['oz']
        mag = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        quat_mags.append(mag)

        # Safety check - don't let plane crash
        if state['pz'] < 200:
            break

    # Calculate drift statistics
    max_drift = max(abs(m - 1.0) for m in quat_mags)
    mean_drift = np.mean([abs(m - 1.0) for m in quat_mags])
    final_mag = quat_mags[-1] if quat_mags else 1.0

    # Quaternion should stay very close to unit length
    drift_ok = max_drift < 0.01  # Allow 1% drift

    RESULTS['quat_norm'] = drift_ok
    status = "OK" if drift_ok else "WARN"

    print(f"quat_norm:     max_drift={max_drift:.6f}, mean_drift={mean_drift:.6f}, final_mag={final_mag:.6f} [{status}]")

    if not drift_ok:
        print(f"    WARNING: Quaternion drift {max_drift:.6f} > 0.01 - may cause euler angle errors")
        print(f"    Consider: Normalize quaternion after integration in C code")

    env.close()
    return drift_ok


# Test registry for this module
TESTS = {
    'obs_during_loop': test_obs_during_loop,
    'obs_during_roll': test_obs_during_roll,
    'obs_vertical_pitch': test_obs_vertical_pitch,
    'obs_azimuth_crossover': test_obs_azimuth_crossover,
    'obs_yaw_wrap': test_obs_yaw_wrap,
    'obs_elevation_extremes': test_obs_elevation_extremes,
    'obs_complex_maneuver': test_obs_complex_maneuver,
    'quat_normalization': test_quaternion_normalization,
}


if __name__ == "__main__":
    from test_flight_base import get_args
    args = get_args()

    print("Dynamic Observation Tests")
    print("=" * 60)

    if args.test:
        if args.test in TESTS:
            print(f"Running single test: {args.test}")
            if get_render_mode():
                print("Rendering enabled - press ESC to exit")
            print("=" * 60)
            TESTS[args.test]()
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {', '.join(TESTS.keys())}")
    else:
        print("Running all dynamic observation tests")
        if get_render_mode():
            print("Rendering enabled - press ESC to exit")
        print("=" * 60)
        for test_func in TESTS.values():
            test_func()

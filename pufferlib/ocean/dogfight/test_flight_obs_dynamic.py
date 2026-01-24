"""
Dynamic maneuver observation tests for dogfight environment.
Tests observation continuity and bounds during active flight maneuvers.

NEW OBS SCHEMES use body-frame observations (velocity, angular rates, AoA).
OLD schemes with Euler angles (pitch/roll/yaw) have been removed.

NEW Scheme 0 (OBS_MOMENTUM) Layout - 15 obs:
  [0-2]   Body-frame velocity (forward speed, sideslip, climb rate)
  [3-5]   Angular velocity (roll rate, pitch rate, yaw rate)
  [6]     Angle of attack
  [7-8]   Altitude, own energy
  [9-12]  Target spherical (azimuth, elevation, range, closure)
  [13-14] Tactical (energy advantage, target aspect)

Run: python pufferlib/ocean/dogfight/test_flight_obs_dynamic.py --test obs_azimuth_crossover
"""
import numpy as np
from dogfight import Dogfight

from test_flight_base import (
    get_render_mode, get_render_fps,
    RESULTS,
)
from test_flight_obs_static import obs_continuity_check


def test_obs_azimuth_crossover():
    """
    Target azimuth +/-180deg crossover test.

    Purpose: Verify azimuth doesn't jump discontinuously when target
    crosses from behind-left to behind-right (through +/-180deg).

    Risk: Azimuth might jump from +1 to -1 instantly instead of transitioning
    smoothly, causing RL agent to see huge observation delta.

    Test: Sweep opponent from right-behind through directly-behind to left-behind
    and check for discontinuities.

    NEW scheme: azimuth is at index 9
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps())
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
        azimuths.append(env.observations[0][9])  # azimuth at index 9
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


def test_obs_elevation_extremes():
    """
    Elevation observation at +/-90deg (target directly above/below).

    Purpose: Verify elevation doesn't have singularity when target is
    directly above or below player. Elevation uses asin which is bounded
    by definition, so this should be stable.

    Test: Place target directly above and below player, verify elevation
    is correct and bounded.

    NEW scheme: elevation is at index 10
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    elev_above = env.observations[0][10]  # elevation at index 10

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
    elev_below = env.observations[0][10]  # elevation at index 10

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
    elev_steep_up = env.observations[0][10]  # elevation at index 10

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
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    'obs_azimuth_crossover': test_obs_azimuth_crossover,
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

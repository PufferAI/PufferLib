"""
Static observation scheme tests for dogfight environment.
Tests observation bounds, dimensions, and values at specific orientations.

Run: python pufferlib/ocean/dogfight/test_flight_obs_static.py --test obs_bounds
"""
import numpy as np
from dogfight import Dogfight, OBS_SIZES

from test_flight_base import (
    get_render_mode, get_render_fps,
    RESULTS, OBS_ATOL, OBS_RTOL,
)


def obs_assert_close(actual, expected, name, atol=OBS_ATOL, rtol=OBS_RTOL):
    """Assert two values are close, with descriptive error."""
    if np.isclose(actual, expected, atol=atol, rtol=rtol):
        return True
    else:
        print(f"    {name}: {actual:.4f} != {expected:.4f} [FAIL]")
        return False


def obs_continuity_check(obs, prev_obs, step, max_delta=0.3):
    """
    Check observation continuity and bounds during dynamic flight.

    Returns tuple: (passed, error_msg)
    - All obs should be in [-1, 1] (proper bounds for NN input)
    - No NaN/Inf values
    - No sudden jumps > max_delta between timesteps (discontinuity detection)

    Args:
        obs: Current observation array
        prev_obs: Previous observation array (or None for first step)
        step: Current timestep (for error messages)
        max_delta: Maximum allowed change per timestep (default 0.3)

    Returns:
        (passed: bool, error_msg: str or None)
    """
    # Check for NaN/Inf
    if np.any(np.isnan(obs)):
        nan_indices = np.where(np.isnan(obs))[0]
        return False, f"NaN at step {step}, indices: {nan_indices}"

    if np.any(np.isinf(obs)):
        inf_indices = np.where(np.isinf(obs))[0]
        return False, f"Inf at step {step}, indices: {inf_indices}"

    # Check bounds [-1, 1]
    for i, val in enumerate(obs):
        if val < -1.0 or val > 1.0:
            return False, f"Obs[{i}]={val:.3f} out of bounds [-1,1] at step {step}"

    # Check continuity (no sudden jumps)
    if prev_obs is not None:
        for i in range(len(obs)):
            delta = abs(obs[i] - prev_obs[i])
            if delta > max_delta:
                return False, f"Discontinuity at step {step}: obs[{i}] jumped {prev_obs[i]:.3f} -> {obs[i]:.3f} (delta={delta:.3f})"

    return True, None


def test_obs_scheme_dimensions():
    """Verify all obs schemes have correct dimensions."""
    all_passed = True
    for scheme, expected_size in OBS_SIZES.items():
        env = Dogfight(num_envs=1, obs_scheme=scheme, render_mode=get_render_mode(), render_fps=get_render_fps())
        env.reset()
        obs = env.observations[0]
        actual = len(obs)
        passed = actual == expected_size
        all_passed &= passed
        status = "OK" if passed else "FAIL"
        print(f"obs_dim_{scheme}:     {actual} obs (expected {expected_size}) [{status}]")
        env.close()
    RESULTS['obs_dimensions'] = all_passed
    return all_passed


def test_obs_identity_orientation():
    """
    Test identity orientation: player at origin, target ahead.
    Expect: pitch=0, roll=0, yaw=0, azimuth=0, elevation=0
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(100, 0, 0),
        player_ori=(1, 0, 0, 0),  # Identity quaternion
        opponent_pos=(400, 0, 1000),
        opponent_vel=(100, 0, 0),
    )

    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    env.step(action)
    obs = env.observations[0]

    passed = True
    passed &= obs_assert_close(obs[4], 0.0, "pitch")
    passed &= obs_assert_close(obs[5], 0.0, "roll")
    passed &= obs_assert_close(obs[6], 0.0, "yaw")
    passed &= obs_assert_close(obs[7], 0.0, "azimuth")
    passed &= obs_assert_close(obs[8], 0.0, "elevation")

    RESULTS['obs_identity'] = passed
    status = "OK" if passed else "FAIL"
    print(f"obs_identity:  identity orientation [{status}]")
    env.close()
    return passed


def test_obs_pitched_up():
    """
    Pitched up 30 degrees.
    Expect: pitch = -30/180 = -0.167 (negative = nose UP)
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    pitch_rad = np.radians(30)
    qw = np.cos(-pitch_rad / 2)
    qy = np.sin(-pitch_rad / 2)

    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(100, 0, 0),
        player_ori=(qw, 0, qy, 0),
        opponent_pos=(400, 0, 1000),
        opponent_vel=(100, 0, 0),
    )

    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    env.step(action)
    obs = env.observations[0]

    expected_pitch = -30.0 / 180.0
    passed = obs_assert_close(obs[4], expected_pitch, "pitch")

    RESULTS['obs_pitched'] = passed
    status = "OK" if passed else "FAIL"
    print(f"obs_pitched:   pitch={obs[4]:.3f} (expect {expected_pitch:.3f}) [{status}]")
    env.close()
    return passed


def test_obs_target_angles():
    """Test target azimuth/elevation computation."""
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps())

    # Target to the right
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(100, 0, 0),
        player_ori=(1, 0, 0, 0),
        opponent_pos=(0, -400, 1000),  # Right (negative Y)
        opponent_vel=(100, 0, 0),
    )
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    env.step(action)
    azimuth_right = env.observations[0][7]

    # Target above
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(100, 0, 0),
        player_ori=(1, 0, 0, 0),
        opponent_pos=(0, 0, 1400),
        opponent_vel=(100, 0, 0),
    )
    env.step(action)
    elev_above = env.observations[0][8]

    passed = True
    passed &= obs_assert_close(azimuth_right, -0.5, "azimuth_right")
    passed &= obs_assert_close(elev_above, 1.0, "elev_above", atol=0.1)

    RESULTS['obs_target_angles'] = passed
    status = "OK" if passed else "FAIL"
    print(f"obs_target:    az_right={azimuth_right:.3f}, elev_up={elev_above:.3f} [{status}]")
    env.close()
    return passed


def test_obs_horizon_visible():
    """Test horizon_visible in scheme 2 (level=1, knife=0, inverted=-1)."""
    env = Dogfight(num_envs=1, obs_scheme=2, render_mode=get_render_mode(), render_fps=get_render_fps())
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # Level
    env.reset()
    env.force_state(player_pos=(0, 0, 1000), player_vel=(100, 0, 0), player_ori=(1, 0, 0, 0),
                    opponent_pos=(400, 0, 1000), opponent_vel=(100, 0, 0))
    env.step(action)
    h_level = env.observations[0][8]

    # Knife-edge (90 deg roll)
    env.reset()
    roll_90 = np.radians(90)
    env.force_state(player_pos=(0, 0, 1000), player_vel=(100, 0, 0),
                    player_ori=(np.cos(-roll_90/2), np.sin(-roll_90/2), 0, 0),
                    opponent_pos=(400, 0, 1000), opponent_vel=(100, 0, 0))
    env.step(action)
    h_knife = env.observations[0][8]

    # Inverted (180 deg roll)
    env.reset()
    roll_180 = np.radians(180)
    env.force_state(player_pos=(0, 0, 1000), player_vel=(100, 0, 0),
                    player_ori=(np.cos(-roll_180/2), np.sin(-roll_180/2), 0, 0),
                    opponent_pos=(400, 0, 1000), opponent_vel=(100, 0, 0))
    env.step(action)
    h_inv = env.observations[0][8]

    passed = True
    passed &= obs_assert_close(h_level, 1.0, "level")
    passed &= obs_assert_close(h_knife, 0.0, "knife", atol=0.1)
    passed &= obs_assert_close(h_inv, -1.0, "inverted")

    RESULTS['obs_horizon'] = passed
    status = "OK" if passed else "FAIL"
    print(f"obs_horizon:   level={h_level:.2f}, knife={h_knife:.2f}, inv={h_inv:.2f} [{status}]")
    env.close()
    return passed


def test_obs_edge_cases():
    """Test edge cases: azimuth at 180°, zero speed, extreme distance."""
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps())
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    passed = True

    # Target behind-left (near +180°)
    env.reset()
    env.force_state(player_pos=(0, 0, 1000), player_vel=(100, 0, 0), player_ori=(1, 0, 0, 0),
                    opponent_pos=(-400, 10, 1000), opponent_vel=(100, 0, 0))
    env.step(action)
    az_left = env.observations[0][7]

    # Target behind-right (near -180°)
    env.reset()
    env.force_state(player_pos=(0, 0, 1000), player_vel=(100, 0, 0), player_ori=(1, 0, 0, 0),
                    opponent_pos=(-400, -10, 1000), opponent_vel=(100, 0, 0))
    env.step(action)
    az_right = env.observations[0][7]

    # Extreme distance (5km)
    env.reset()
    env.force_state(player_pos=(0, 0, 1000), player_vel=(100, 0, 0), player_ori=(1, 0, 0, 0),
                    opponent_pos=(5000, 0, 1000), opponent_vel=(100, 0, 0))
    env.step(action)
    dist_obs = env.observations[0][9]

    passed &= az_left > 0.9  # Should be near +1
    passed &= az_right < -0.9  # Should be near -1
    passed &= -1.0 <= dist_obs <= 1.0  # Should be clamped

    RESULTS['obs_edge_cases'] = passed
    status = "OK" if passed else "FAIL"
    print(f"obs_edges:     az_180={az_left:.2f}/{az_right:.2f}, dist_clamp={dist_obs:.2f} [{status}]")
    env.close()
    return passed


def test_obs_bounds():
    """Test that random states produce bounded observations in [-1, 1] for NN input."""
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=get_render_mode(), render_fps=get_render_fps())
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    passed = True
    out_of_bounds = []

    for trial in range(30):
        env.reset()
        pos = (np.random.uniform(-4000, 4000), np.random.uniform(-4000, 4000), np.random.uniform(100, 2900))
        vel = tuple(np.random.randn(3) * 100)
        ori = np.random.randn(4)
        ori /= np.linalg.norm(ori)
        if ori[0] < 0: ori = -ori
        opp_pos = (pos[0] + np.random.uniform(-500, 500), pos[1] + np.random.uniform(-500, 500), pos[2] + np.random.uniform(-500, 500))

        env.force_state(player_pos=pos, player_vel=vel, player_ori=tuple(ori),
                        opponent_pos=opp_pos, opponent_vel=(100, 0, 0))
        env.step(action)

        for i, val in enumerate(env.observations[0]):
            if val < -1.0 or val > 1.0:
                passed = False
                out_of_bounds.append((trial, i, val))

    RESULTS['obs_bounds'] = passed
    status = "OK" if passed else "FAIL"
    print(f"obs_bounds:    30 random states, all in [-1.0, 1.0] [{status}]")
    if out_of_bounds:
        for trial, idx, val in out_of_bounds[:5]:  # Show first 5 violations
            print(f"    trial {trial}: obs[{idx}]={val:.3f} out of bounds")
    env.close()
    return passed


# Test registry for this module
TESTS = {
    'obs_dimensions': test_obs_scheme_dimensions,
    'obs_identity': test_obs_identity_orientation,
    'obs_pitched': test_obs_pitched_up,
    'obs_target_angles': test_obs_target_angles,
    'obs_horizon': test_obs_horizon_visible,
    'obs_edge_cases': test_obs_edge_cases,
    'obs_bounds': test_obs_bounds,
}


if __name__ == "__main__":
    from test_flight_base import get_args
    args = get_args()

    print("Static Observation Tests")
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
        print("Running all static observation tests")
        if get_render_mode():
            print("Rendering enabled - press ESC to exit")
        print("=" * 60)
        for test_func in TESTS.values():
            test_func()

"""
OBS_PURSUIT (scheme 1) specific tests for dogfight environment.
Tests energy observations, target aspect, closure rate, and wrap behavior.

Observation layout for OBS_PURSUIT (13 observations):
  0: speed        - clamp(speed/250, 0, 1)           [0, 1]
  1: potential    - alt/3000                         [0, 1]
  2: pitch        - pitch / (PI/2)                   [-1, 1]
  3: roll         - roll / PI                        [-1, 1]  **WRAPS**
  4: own_energy   - (potential + kinetic) / 2        [0, 1]
  5: target_az    - target_az / PI                   [-1, 1]  **WRAPS**
  6: target_el    - target_el / (PI/2)               [-1, 1]
  7: dist         - clamp(dist/500, 0, 2) - 1        [-1, 1]
  8: closure      - clamp(closure/250, -1, 1)        [-1, 1]
  9: target_roll  - target_roll / PI                 [-1, 1]  **WRAPS**
 10: target_pitch - target_pitch / (PI/2)            [-1, 1]
 11: target_aspect- dot(opp_fwd, to_player)          [-1, 1]
 12: energy_adv   - clamp(own_E - opp_E, -1, 1)      [-1, 1]

Run: python pufferlib/ocean/dogfight/test_flight_obs_pursuit.py --test obs_pursuit_bounds
"""
import numpy as np
from dogfight import Dogfight

from test_flight_base import (
    get_render_mode, get_render_fps, get_physics_mode,
    RESULTS,
)


def test_obs_pursuit_bounds():
    """
    Run random maneuvers in OBS_PURSUIT (scheme 1) and verify all observations
    stay in valid ranges. This catches NaN/Inf/out-of-bounds issues.

    OBS_PURSUIT has 13 observations with specific bounds:
    - Indices 0, 1, 4: [0, 1] (speed, potential, own_energy)
    - All others: [-1, 1]
    """
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    env.reset()

    violations = []
    np.random.seed(42)  # Reproducible

    for step in range(500):
        # Random maneuvers
        throttle = np.random.uniform(0.3, 1.0)
        elevator = np.random.uniform(-0.5, 0.5)
        aileron = np.random.uniform(-0.8, 0.8)
        rudder = np.random.uniform(-0.3, 0.3)
        action = np.array([[throttle, elevator, aileron, rudder, 0.0]], dtype=np.float32)

        _, _, term, _, _ = env.step(action)
        obs = env.observations[0]

        for i, val in enumerate(obs):
            if np.isnan(val) or np.isinf(val):
                violations.append(f"NaN/Inf at step {step}, obs[{i}]")
            # Indices 0, 1, 4 are [0, 1], rest are [-1, 1]
            if i in [0, 1, 4]:  # speed, potential, energy are [0, 1]
                if val < -0.01 or val > 1.01:
                    violations.append(f"obs[{i}]={val:.3f} out of [0,1] at step {step}")
            else:
                if val < -1.01 or val > 1.01:
                    violations.append(f"obs[{i}]={val:.3f} out of [-1,1] at step {step}")

        if term[0]:
            env.reset()

    passed = len(violations) == 0
    RESULTS['obs_pursuit_bounds'] = passed
    status = "OK" if passed else "FAIL"
    print(f"obs_pursuit_bounds: 500 steps, violations={len(violations)} [{status}]")
    if violations:
        for v in violations[:5]:
            print(f"    {v}")
    env.close()
    return passed


def test_obs_pursuit_energy_conservation():
    """
    Vertical climb: watch kinetic -> potential energy conversion.

    Physics: In ideal climb (no drag): E = mgh + 0.5mv^2 = constant
    At v=100 m/s, h_max = v^2/(2g) = 509.7m (drag-free)
    With drag, actual h_max < 509.7m

    Energy observation (obs[4]) should decrease slightly due to drag,
    but not increase significantly (conservation violation).
    """
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    env.reset()

    # 90deg pitch, 100 m/s, low throttle
    pitch_90 = np.radians(90)
    qw = np.cos(pitch_90 / 2)
    qy = -np.sin(pitch_90 / 2)  # Negative for nose UP

    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(0, 0, 100),      # 100 m/s vertical velocity
        player_ori=(qw, 0, qy, 0),   # Nose straight up
        player_throttle=0.1,          # Minimal throttle
        opponent_pos=(500, 0, 1000),
        opponent_vel=(100, 0, 0),
    )

    data = []
    for step in range(200):  # ~4 seconds
        action = np.array([[0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # Minimal throttle
        env.step(action)
        obs = env.observations[0]
        state = env.get_state()

        data.append({
            'step': step,
            'vz': state['vz'],
            'alt': state['pz'],
            'speed_obs': obs[0],
            'potential_obs': obs[1],
            'own_energy': obs[4],
        })

        # Stop when vertical velocity near zero (apex)
        if state['vz'] < 5:
            break

    # Analysis
    initial_energy = data[0]['own_energy']
    final_energy = data[-1]['own_energy']
    alt_gained = data[-1]['alt'] - data[0]['alt']

    # Energy should not INCREASE significantly (conservation violation)
    # Allow 5% tolerance for thrust contribution at low throttle
    energy_increase = final_energy > initial_energy + 0.05

    # Altitude gain should be reasonable (with drag losses)
    # Ideal: 509.7m, expect ~300-550m with drag
    alt_reasonable = 200 < alt_gained < 600

    passed = not energy_increase and alt_reasonable
    RESULTS['obs_pursuit_energy_climb'] = passed
    status = "OK" if passed else "CHECK"

    print(f"obs_pursuit_energy_climb: E: {initial_energy:.3f}->{final_energy:.3f}, alt_gain={alt_gained:.0f}m [{status}]")
    if energy_increase:
        print(f"    WARNING: Energy increased {final_energy - initial_energy:.3f} (conservation violation?)")
    if not alt_reasonable:
        print(f"    WARNING: Alt gain {alt_gained:.0f}m outside expected 200-600m")

    env.close()
    return passed


def test_obs_pursuit_energy_dive():
    """
    Dive: watch potential -> kinetic energy conversion.

    Start high (2500m), pitch down, let gravity accelerate.
    Energy should be relatively stable (gravity -> speed, drag -> loss).
    """
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    env.reset()

    # Start high, pitch down 45deg
    pitch_down = np.radians(-45)
    qw = np.cos(pitch_down / 2)
    qy = -np.sin(pitch_down / 2)

    env.force_state(
        player_pos=(0, 0, 2500),
        player_vel=(50, 0, 0),
        player_ori=(qw, 0, qy, 0),
        player_throttle=0.0,  # Idle
        opponent_pos=(500, 0, 2500),
        opponent_vel=(100, 0, 0),
    )

    data = []
    for step in range(200):
        action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # Idle, let gravity work
        _, _, term, _, _ = env.step(action)
        obs = env.observations[0]
        state = env.get_state()

        speed = np.sqrt(state['vx']**2 + state['vy']**2 + state['vz']**2)
        data.append({
            'step': step,
            'speed': speed,
            'alt': state['pz'],
            'speed_obs': obs[0],
            'potential_obs': obs[1],
            'own_energy': obs[4],
        })

        if state['pz'] < 800 or term[0]:  # Stop at 800m or termination
            break

    initial_energy = data[0]['own_energy']
    final_energy = data[-1]['own_energy']
    speed_gained = data[-1]['speed'] - data[0]['speed']
    alt_lost = data[0]['alt'] - data[-1]['alt']

    # Energy should decrease slightly (drag) but not increase
    energy_increase = final_energy > initial_energy + 0.05
    # Speed should increase (gravity)
    speed_gain_ok = speed_gained > 20

    passed = not energy_increase and speed_gain_ok
    RESULTS['obs_pursuit_energy_dive'] = passed
    status = "OK" if passed else "CHECK"

    print(f"obs_pursuit_energy_dive: E: {initial_energy:.3f}->{final_energy:.3f}, speed_gain={speed_gained:.0f}m/s, alt_loss={alt_lost:.0f}m [{status}]")
    if energy_increase:
        print(f"    WARNING: Energy increased during unpowered dive")

    env.close()
    return passed


def test_obs_pursuit_energy_advantage():
    """
    Test energy advantage observation (obs[12]) with different altitude/speed configs.

    Energy advantage = own_energy - opponent_energy, clamped to [-1, 1]
    - Higher/faster player should have positive advantage
    - Lower/slower player should have negative advantage
    - Equal state should have ~0 advantage
    """
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # Case 1: Player higher, same speed -> positive advantage
    env.reset()
    env.force_state(
        player_pos=(0, 0, 2000), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 1000), opponent_vel=(100, 0, 0),
    )
    env.step(action)
    adv_high = env.observations[0][12]

    # Case 2: Player lower, same speed -> negative advantage
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1000), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 2000), opponent_vel=(100, 0, 0),
    )
    env.step(action)
    adv_low = env.observations[0][12]

    # Case 3: Same altitude, player faster -> positive advantage
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(150, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(80, 0, 0),
    )
    env.step(action)
    adv_fast = env.observations[0][12]

    # Case 4: Equal state -> zero advantage
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(100, 0, 0),
    )
    env.step(action)
    adv_equal = env.observations[0][12]

    # Verify
    high_ok = adv_high > 0.1
    low_ok = adv_low < -0.1
    fast_ok = adv_fast > 0.0
    equal_ok = abs(adv_equal) < 0.05

    passed = high_ok and low_ok and fast_ok and equal_ok
    RESULTS['obs_pursuit_energy_adv'] = passed
    status = "OK" if passed else "FAIL"

    print(f"obs_pursuit_energy_adv: high={adv_high:.3f}, low={adv_low:.3f}, fast={adv_fast:.3f}, equal={adv_equal:.3f} [{status}]")
    if not high_ok:
        print(f"    FAIL: Higher player should have positive advantage, got {adv_high:.3f}")
    if not low_ok:
        print(f"    FAIL: Lower player should have negative advantage, got {adv_low:.3f}")
    if not equal_ok:
        print(f"    FAIL: Equal state should have ~0 advantage, got {adv_equal:.3f}")

    env.close()
    return passed


def test_obs_pursuit_target_aspect():
    """
    Test target aspect observation (obs[11]).

    target_aspect = dot(opponent_forward, to_player)
    - Head-on (opponent facing us): ~+1.0
    - Tail (opponent facing away): ~-1.0
    - Beam (perpendicular): ~0.0

    IMPORTANT: Must set opponent_ori to match opponent_vel, otherwise
    physics step will severely alter velocity (flying "backward" is not stable).
    """
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    action = np.array([[0.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # Some throttle

    # Head-on: opponent facing toward player (yaw=180deg = facing -X)
    # Quaternion for yaw=180deg: qw=0, qz=1
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(-100, 0, 0),
        opponent_ori=(0, 0, 0, 1),  # Yaw=180deg = facing -X (toward player)
    )
    env.step(action)
    aspect_head_on = env.observations[0][11]

    # Tail: opponent facing away from player (identity = facing +X)
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(100, 0, 0),
        opponent_ori=(1, 0, 0, 0),  # Identity = facing +X (away from player)
    )
    env.step(action)
    aspect_tail = env.observations[0][11]

    # Beam: opponent perpendicular (yaw=-90deg = facing +Y)
    # Quaternion for yaw=-90deg: qw=cos(-45deg)~0.707, qz=sin(-45deg)~-0.707
    cos45 = np.cos(np.radians(-45))
    sin45 = np.sin(np.radians(-45))
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(0, 100, 0),
        opponent_ori=(cos45, 0, 0, sin45),  # Yaw=-90deg = facing +Y
    )
    env.step(action)
    aspect_beam = env.observations[0][11]

    # Verify
    head_on_ok = aspect_head_on > 0.85  # Near +1
    tail_ok = aspect_tail < -0.85       # Near -1
    beam_ok = abs(aspect_beam) < 0.3    # Near 0

    passed = head_on_ok and tail_ok and beam_ok
    RESULTS['obs_pursuit_aspect'] = passed
    status = "OK" if passed else "FAIL"

    print(f"obs_pursuit_aspect: head_on={aspect_head_on:.3f}, tail={aspect_tail:.3f}, beam={aspect_beam:.3f} [{status}]")
    if not head_on_ok:
        print(f"    FAIL: Head-on should be >0.85, got {aspect_head_on:.3f}")
    if not tail_ok:
        print(f"    FAIL: Tail should be <-0.85, got {aspect_tail:.3f}")
    if not beam_ok:
        print(f"    FAIL: Beam should be near 0, got {aspect_beam:.3f}")

    env.close()
    return passed


def test_obs_pursuit_closure_rate():
    """
    Test closure rate observation (obs[8]).

    closure = dot(relative_vel, normalized_to_target)
    - Closing (getting closer): positive
    - Separating (getting farther): negative
    - Head-on (both approaching): high positive

    IMPORTANT: Must set opponent_ori to match opponent_vel to avoid
    physics instability (flying backward causes extreme drag).
    """
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    action = np.array([[0.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # Some throttle

    # Closing: player faster toward target (chasing)
    # Both facing +X (default orientation)
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(150, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(50, 0, 0),
        opponent_ori=(1, 0, 0, 0),  # Facing +X (same as velocity)
    )
    env.step(action)
    closure_closing = env.observations[0][8]

    # Separating: target running away faster
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(80, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(150, 0, 0),
        opponent_ori=(1, 0, 0, 0),  # Facing +X
    )
    env.step(action)
    closure_separating = env.observations[0][8]

    # Head-on: both approaching each other
    # Opponent facing -X (toward player): yaw=180deg -> qw=0, qz=1
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(-100, 0, 0),
        opponent_ori=(0, 0, 0, 1),  # Yaw=180deg = facing -X
    )
    env.step(action)
    closure_head_on = env.observations[0][8]

    # Verify
    closing_ok = closure_closing > 0.3
    separating_ok = closure_separating < -0.2
    head_on_ok = closure_head_on > 0.7

    passed = closing_ok and separating_ok and head_on_ok
    RESULTS['obs_pursuit_closure'] = passed
    status = "OK" if passed else "FAIL"

    print(f"obs_pursuit_closure: closing={closure_closing:.3f}, separating={closure_separating:.3f}, head_on={closure_head_on:.3f} [{status}]")
    if not closing_ok:
        print(f"    FAIL: Closing rate should be >0.3, got {closure_closing:.3f}")
    if not separating_ok:
        print(f"    FAIL: Separating rate should be <-0.2, got {closure_separating:.3f}")
    if not head_on_ok:
        print(f"    FAIL: Head-on closure should be >0.7, got {closure_head_on:.3f}")

    env.close()
    return passed


def test_obs_pursuit_target_angles_wrap():
    """
    Check target_az (obs[5]) and target_roll (obs[9]) for wrap discontinuities.

    Sweep target position around player (behind the player through +/-180deg)
    and check for large discontinuities in target_az.
    """
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=get_render_mode(), render_fps=get_render_fps(), physics_mode=get_physics_mode())
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    target_azs = []
    y_positions = []

    # Sweep opponent from right-behind (y=-200) through left-behind (y=+200)
    for step in range(50):
        env.reset()
        y_offset = -200 + step * 8  # Sweep from y=-200 to y=+200

        env.force_state(
            player_pos=(0, 0, 1500),
            player_vel=(100, 0, 0),
            player_ori=(1, 0, 0, 0),  # Identity - facing +X
            opponent_pos=(-200, y_offset, 1500),  # Behind player, sweeping Y
            opponent_vel=(100, 0, 0),
        )

        env.step(action)
        target_azs.append(env.observations[0][5])
        y_positions.append(y_offset)

    # Check for discontinuities
    az_jumps = []
    for i in range(1, len(target_azs)):
        delta = abs(target_azs[i] - target_azs[i-1])
        if delta > 0.5:  # Large jump = discontinuity
            az_jumps.append((i, y_positions[i], target_azs[i-1], target_azs[i], delta))

    # Verify azimuth range covers near +/-1 (behind = +/-180deg)
    az_min = min(target_azs)
    az_max = max(target_azs)
    range_ok = az_max > 0.8 and az_min < -0.8

    # Discontinuity at +/-180deg crossover is EXPECTED for atan2-based azimuth
    has_discontinuity = len(az_jumps) > 0

    RESULTS['obs_pursuit_az_wrap'] = range_ok
    status = "OK" if range_ok else "CHECK"

    print(f"obs_pursuit_az_wrap: range=[{az_min:.2f},{az_max:.2f}], discontinuities={len(az_jumps)} [{status}]")

    if has_discontinuity:
        print(f"    NOTE: target_az has discontinuity at +/-180deg (expected for atan2)")
        for _, y_pos, prev_az, curr_az, delta in az_jumps[:2]:
            print(f"    At y={y_pos:.0f}: az {prev_az:.2f} -> {curr_az:.2f} (delta={delta:.2f})")
        print(f"    Consider: Use sin/cos encoding for RL training")

    if not range_ok:
        print(f"    WARNING: target_az didn't reach +/-1 (behind player)")

    env.close()
    return range_ok


# Test registry for this module
TESTS = {
    'obs_pursuit_bounds': test_obs_pursuit_bounds,
    'obs_pursuit_energy_climb': test_obs_pursuit_energy_conservation,
    'obs_pursuit_energy_dive': test_obs_pursuit_energy_dive,
    'obs_pursuit_energy_adv': test_obs_pursuit_energy_advantage,
    'obs_pursuit_aspect': test_obs_pursuit_target_aspect,
    'obs_pursuit_closure': test_obs_pursuit_closure_rate,
    'obs_pursuit_az_wrap': test_obs_pursuit_target_angles_wrap,
}


if __name__ == "__main__":
    from test_flight_base import get_args
    args = get_args()

    print("OBS_PURSUIT (Scheme 1) Tests")
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
        print("Running all OBS_PURSUIT tests")
        if get_render_mode():
            print("Rendering enabled - press ESC to exit")
        print("=" * 60)
        for test_func in TESTS.values():
            test_func()

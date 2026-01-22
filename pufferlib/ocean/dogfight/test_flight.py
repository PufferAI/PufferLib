"""
Physics validation tests for dogfight environment.
Uses force_state() to set exact initial conditions for accurate measurements.

Run: python pufferlib/ocean/dogfight/test_flight.py
     python pufferlib/ocean/dogfight/test_flight.py --render  # with visualization
     python pufferlib/ocean/dogfight/test_flight.py --render --test pitch_direction  # single test

TODO - FLIGHT PHYSICS TESTS NEEDED:
=====================================
1. RUDDER-ONLY TURN TEST (HIGH PRIORITY)
   - Current MAX_YAW_RATE = 1.5 rad/s (86 deg/s) is WAY too high
   - P-51D rudder should give ~5-15 deg/s yaw rate max, with significant sideslip
   - Test: wings level, full rudder, measure actual yaw rate and heading change
   - Compare against P-51D flight test data (see P51d_REFERENCE_DATA.md)
   - Expected: rudder alone should NOT be effective for turning - need bank

2. COORDINATED TURN TEST
   - Bank to 30°, 45°, 60° and measure sustained turn rate
   - P-51D should get ~17.5 deg/s at max sustained (corner velocity)
   - Verify turn rate vs bank angle relationship

3. ROLL RATE TEST
   - Full aileron deflection, measure time to roll 90° and 360°
   - P-51D: ~90-100 deg/s roll rate at 300 mph

4. PITCH AUTHORITY TEST
   - Full elevator, measure pitch rate and G-loading
   - Should be speed-dependent (less authority at low speed)
"""
import argparse
import numpy as np
from dogfight import Dogfight, AutopilotMode, OBS_SIZES


def parse_args():
    parser = argparse.ArgumentParser(description='P-51D Physics Validation Tests')
    parser.add_argument('--render', action='store_true', help='Enable visual rendering')
    parser.add_argument('--fps', type=int, default=50, help='Target FPS when rendering (default 50 = real-time, try 5-10 for slow-mo)')
    parser.add_argument('--test', type=str, default=None, help='Run specific test only')
    return parser.parse_args()


ARGS = parse_args()
RENDER_MODE = 'human' if ARGS.render else None
RENDER_FPS = ARGS.fps if ARGS.render else None

# Constants (must match dogfight.h)
MAX_SPEED = 250.0
WORLD_MAX_Z = 3000.0
WORLD_HALF_X = 5000.0
WORLD_HALF_Y = 5000.0
GUN_RANGE = 1000.0

# Tolerance for observation tests
OBS_ATOL = 0.05  # Absolute tolerance
OBS_RTOL = 0.1   # Relative tolerance

# P-51D reference values (from P51d_REFERENCE_DATA.md)
P51D_MAX_SPEED = 159.0      # m/s (355 mph, Military power, SL)
P51D_STALL_SPEED = 45.0     # m/s (100 mph, 9000 lb, clean)
P51D_CLIMB_RATE = 15.4      # m/s (3030 ft/min, Military power)
P51D_TURN_RATE = 17.5       # deg/s at max sustained turn (DCS testing data)

# PID values for level flight autopilot (found via pid_sweep.py)
# These give stable level flight with vz_std < 0.3 m/s
LEVEL_FLIGHT_KP = 0.001     # Proportional gain on vz error
LEVEL_FLIGHT_KD = 0.001     # Derivative gain (damping)

RESULTS = {}

# Observation indices to highlight for each test (scheme 0 - ANGLES)
# These are the key observations to watch during visual inspection
# Scheme 0: px(0), py(1), pz(2), speed(3), pitch(4), roll(5), yaw(6), tgt_az(7), tgt_el(8), dist(9), closure(10), opp_hdg(11)
TEST_HIGHLIGHTS = {
    'knife_edge_pull': [4, 5, 6],     # pitch, roll, yaw - watch yaw change, roll should stay ~90°
    'knife_edge_flight': [4, 5, 6],   # pitch, roll, yaw - watch altitude loss and yaw authority
    'sustained_turn': [4, 5],         # pitch, roll - watch bank angle
    'turn_60': [4, 5],                # pitch, roll - 60° bank turn
    'pitch_direction': [4],           # pitch - confirm direction matches input
    'roll_direction': [5],            # roll - confirm direction matches input
    'rudder_only_turn': [6],          # yaw - watch yaw rate
    'g_level_flight': [4],            # pitch - should stay near 0
    'g_push_forward': [4],            # pitch - pushing forward
    'g_pull_back': [4],               # pitch - pulling back
    'g_limit_negative': [4, 5],       # pitch, roll - negative G limit
    'g_limit_positive': [4, 5],       # pitch, roll - positive G limit
    'climb_rate': [2, 4],             # pz (altitude), pitch
    'glide_ratio': [2, 3],            # pz (altitude), speed
    'stall_speed': [3],               # speed - watch it decrease
}


def setup_highlights(env, test_name):
    """Set observation highlights if this test has them defined and rendering is enabled."""
    if RENDER_MODE and test_name in TEST_HIGHLIGHTS:
        env.set_obs_highlight(TEST_HIGHLIGHTS[test_name])


# =============================================================================
# State accessor functions using get_state() (independent of obs_scheme)
# =============================================================================

def get_speed_from_state(env):
    """Get total speed from raw state."""
    s = env.get_state()
    return np.sqrt(s['vx']**2 + s['vy']**2 + s['vz']**2)


def get_vz_from_state(env):
    """Get vertical velocity from raw state."""
    return env.get_state()['vz']


def get_alt_from_state(env):
    """Get altitude from raw state."""
    return env.get_state()['pz']


def get_up_vector_from_state(env):
    """Get up vector from raw state."""
    s = env.get_state()
    return s['up_x'], s['up_y'], s['up_z']


def get_velocity_from_state(env):
    """Get velocity vector from raw state."""
    s = env.get_state()
    return s['vx'], s['vy'], s['vz']


def level_flight_pitch_from_state(env, kp=LEVEL_FLIGHT_KP, kd=LEVEL_FLIGHT_KD):
    """
    PD autopilot for level flight (vz = 0).
    Uses tuned PID values from pid_sweep.py for stable flight.
    """
    vz = get_vz_from_state(env)
    # Negative because: if climbing (vz>0), need nose down (negative elevator)
    elevator = -kp * vz - kd * vz
    return np.clip(elevator, -0.2, 0.2)


# =============================================================================
# Legacy functions (use observations - for obs_scheme testing only)
# =============================================================================

def get_speed(obs):
    """Get total speed from observation (LEGACY - assumes WORLD_FRAME)."""
    vx = obs[0, 3] * MAX_SPEED
    vy = obs[0, 4] * MAX_SPEED
    vz = obs[0, 5] * MAX_SPEED
    return np.sqrt(vx**2 + vy**2 + vz**2)


def get_vz(obs):
    """Get vertical velocity from observation (LEGACY - assumes WORLD_FRAME)."""
    return obs[0, 5] * MAX_SPEED


def get_alt(obs):
    """Get altitude from observation (LEGACY - assumes WORLD_FRAME)."""
    return obs[0, 2] * WORLD_MAX_Z


def level_flight_pitch(obs, kp=LEVEL_FLIGHT_KP, kd=LEVEL_FLIGHT_KD):
    """
    PD autopilot for level flight (vz = 0). LEGACY - assumes WORLD_FRAME.
    Uses tuned PID values from pid_sweep.py for stable flight.
    """
    vz = get_vz(obs)
    # Negative because: if climbing (vz>0), need nose down (negative elevator)
    elevator = -kp * vz - kd * vz
    return np.clip(elevator, -0.2, 0.2)


def test_max_speed():
    """
    Full throttle level flight starting near max speed.
    Should stabilize around 159 m/s (P-51D Military power).
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Start at 150 m/s (near expected max), center of world, flying +X
    env.force_state(
        player_pos=(-1000, 0, 1000),
        player_vel=(150, 0, 0),
        player_throttle=1.0,
    )

    prev_speed = get_speed_from_state(env)
    stable_count = 0

    for step in range(1500):  # 30 seconds
        elevator = level_flight_pitch_from_state(env)
        action = np.array([[1.0, elevator, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)

        if term[0]:
            print("  (terminated - hit bounds)")
            break

        speed = get_speed_from_state(env)
        if abs(speed - prev_speed) < 0.05:
            stable_count += 1
            if stable_count > 100:
                break
        else:
            stable_count = 0
        prev_speed = speed

    final_speed = get_speed_from_state(env)
    RESULTS['max_speed'] = final_speed
    diff = final_speed - P51D_MAX_SPEED
    status = "OK" if abs(diff) < 15 else "CHECK"
    print(f"max_speed:     {final_speed:6.1f} m/s  (P-51D: {P51D_MAX_SPEED:.0f}, diff: {diff:+.1f}) [{status}]")


def test_acceleration():
    """
    Full throttle starting at 100 m/s - verify plane accelerates.
    Should see speed increase toward max speed (~150 m/s).
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Start at 100 m/s (well below max speed)
    env.force_state(
        player_pos=(-1000, 0, 1000),
        player_vel=(100, 0, 0),
        player_throttle=1.0,
    )

    initial_speed = get_speed_from_state(env)
    speeds = [initial_speed]

    for step in range(500):  # 10 seconds
        elevator = level_flight_pitch_from_state(env)
        action = np.array([[1.0, elevator, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)

        if term[0]:
            print("  (terminated - hit bounds)")
            break

        speed = get_speed_from_state(env)
        speeds.append(speed)

    final_speed = speeds[-1]
    speed_gain = final_speed - initial_speed
    RESULTS['acceleration'] = speed_gain

    # Should gain at least 20 m/s in 10 seconds
    status = "OK" if speed_gain > 20 else "CHECK"
    print(f"acceleration:  {initial_speed:.0f} -> {final_speed:.0f} m/s  (gained {speed_gain:+.1f} m/s) [{status}]")


def test_deceleration():
    """
    Zero throttle starting at 150 m/s - verify plane decelerates due to drag.
    Should see speed decrease as drag slows the plane.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Start at 150 m/s with zero throttle
    env.force_state(
        player_pos=(-1000, 0, 1000),
        player_vel=(150, 0, 0),
        player_throttle=0.0,
    )

    initial_speed = get_speed_from_state(env)
    speeds = [initial_speed]

    for step in range(500):  # 10 seconds
        elevator = level_flight_pitch_from_state(env)
        # Zero throttle (action[0] = -1 maps to 0% throttle)
        action = np.array([[-1.0, elevator, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)

        if term[0]:
            print("  (terminated - hit bounds)")
            break

        speed = get_speed_from_state(env)
        speeds.append(speed)

    final_speed = speeds[-1]
    speed_loss = initial_speed - final_speed
    RESULTS['deceleration'] = speed_loss

    # Should lose at least 20 m/s in 10 seconds due to drag
    status = "OK" if speed_loss > 20 else "CHECK"
    print(f"deceleration:  {initial_speed:.0f} -> {final_speed:.0f} m/s  (lost {speed_loss:+.1f} m/s) [{status}]")


def test_cruise_speed():
    """50% throttle level flight - cruise speed."""
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Start at moderate speed
    env.force_state(
        player_pos=(-1000, 0, 1000),
        player_vel=(120, 0, 0),
        player_throttle=0.5,
    )

    prev_speed = get_speed_from_state(env)
    stable_count = 0

    for step in range(1500):
        elevator = level_flight_pitch_from_state(env)
        action = np.array([[0.0, elevator, 0.0, 0.0, 0.0]], dtype=np.float32)  # 50% throttle
        _, _, term, _, _ = env.step(action)

        if term[0]:
            break

        speed = get_speed_from_state(env)
        if abs(speed - prev_speed) < 0.05:
            stable_count += 1
            if stable_count > 100:
                break
        else:
            stable_count = 0
        prev_speed = speed

    final_speed = get_speed_from_state(env)
    RESULTS['cruise_speed'] = final_speed
    print(f"cruise_speed:  {final_speed:6.1f} m/s  (50% throttle)")


def test_stall_speed():
    """
    Find stall speed by testing level flight at decreasing speeds.

    At each speed, set the exact pitch angle needed for level flight,
    then verify the physics can maintain altitude. Stall occurs when
    required C_L exceeds C_L_max.

    This bypasses autopilot limitations by setting pitch directly.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)

    # Physics constants (must match flightlib.h)
    W = 4082 * 9.81      # Weight (N)
    rho = 1.225          # Air density
    S = 21.65            # Wing area
    C_L_max = 1.48       # Max lift coefficient
    C_L_alpha = 5.56     # Lift curve slope
    alpha_zero = -0.021  # Zero-lift angle (rad)
    wing_inc = 0.026     # Wing incidence (rad)

    # Theoretical stall speed
    V_stall_theory = np.sqrt(2 * W / (rho * S * C_L_max))

    # Test speeds from high to low
    stall_speed = None
    last_flyable = None

    for V in range(70, 35, -5):
        env.reset()

        # C_L needed for level flight at this speed
        q_dyn = 0.5 * rho * V * V
        C_L_needed = W / (q_dyn * S)

        # Check if within aerodynamic limits
        if C_L_needed > C_L_max:
            # Can't fly level - this is stall
            stall_speed = V
            break

        # Calculate pitch angle needed for this C_L
        # C_L = C_L_alpha * (alpha + wing_inc - alpha_zero)
        alpha_needed = C_L_needed / C_L_alpha - wing_inc + alpha_zero

        # Create pitch-up quaternion (rotation about Y axis)
        # Negative angle because positive Y rotation = nose DOWN (right-hand rule)
        pitch_rad = alpha_needed
        ori_w = np.cos(-pitch_rad / 2)
        ori_y = np.sin(-pitch_rad / 2)

        # Set up plane at exact pitch for level flight
        env.force_state(
            player_pos=(0, 0, 1000),
            player_vel=(V, 0, 0),
            player_ori=(ori_w, 0, ori_y, 0),
            player_throttle=0.0,  # Zero throttle - just testing lift
        )

        # Run for 2 seconds with zero controls, measure vz
        vzs = []
        for _ in range(100):  # 2 seconds
            vz = get_vz_from_state(env)
            vzs.append(vz)
            action = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            _, _, term, _, _ = env.step(action)
            if term[0]:
                break

        avg_vz = np.mean(vzs[-50:]) if len(vzs) >= 50 else np.mean(vzs)

        # If maintaining altitude (vz near 0 or positive), plane can fly
        if avg_vz >= -5:  # Allow small sink rate
            last_flyable = V

    # Stall speed is between last_flyable and the speed where C_L > C_L_max
    if stall_speed is None:
        stall_speed = 35  # Below our test range
    elif last_flyable is not None:
        # Interpolate: stall is where we transition from flyable to not
        stall_speed = last_flyable

    RESULTS['stall_speed'] = stall_speed
    diff = stall_speed - P51D_STALL_SPEED
    status = "OK" if abs(diff) < 10 else "CHECK"
    print(f"stall_speed:   {stall_speed:6.1f} m/s  (P-51D: {P51D_STALL_SPEED:.0f}, diff: {diff:+.1f}, theory: {V_stall_theory:.0f}) [{status}]")


def test_climb_rate():
    """
    Measure climb rate at Vy (best climb speed) with optimal pitch.

    Sets up plane at Vy with the pitch angle calculated for steady climb,
    then measures actual climb rate. This tests that physics produces
    correct excess thrust at climb speed.

    Approach: Calculate pitch for expected P-51D climb (15.4 m/s at 74 m/s),
    set that state with force_state(), run with zero elevator (pitch holds),
    and verify physics produces the expected climb rate.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)

    # Physics constants (must match flightlib.h)
    W = 4082 * 9.81      # Weight (N)
    rho = 1.225          # Air density
    S = 21.65            # Wing area
    C_L_alpha = 5.56     # Lift curve slope
    alpha_zero = -0.021  # Zero-lift angle (rad)
    wing_inc = 0.026     # Wing incidence (rad)

    Vy = 74.0  # Best climb speed (m/s)

    # Calculate climb geometry for P-51D expected performance
    expected_ROC = P51D_CLIMB_RATE  # 15.4 m/s
    gamma = np.arcsin(expected_ROC / Vy)  # Climb angle ~12°

    # In steady climb: L = W * cos(gamma)
    L_needed = W * np.cos(gamma)
    q_dyn = 0.5 * rho * Vy * Vy
    C_L = L_needed / (q_dyn * S)

    # Calculate AOA needed for this lift
    alpha = C_L / C_L_alpha - wing_inc + alpha_zero

    # Body pitch = AOA + climb angle (nose above horizon)
    pitch = alpha + gamma

    # Create pitch-up quaternion (negative angle because positive Y rotation = nose DOWN)
    ori_w = np.cos(-pitch / 2)
    ori_y = np.sin(-pitch / 2)

    # Set up plane in steady climb: velocity vector along climb path
    vx = Vy * np.cos(gamma)
    vz = Vy * np.sin(gamma)  # This IS the expected climb rate

    env.reset()
    setup_highlights(env, 'climb_rate')
    env.force_state(
        player_pos=(0, 0, 500),
        player_vel=(vx, 0, vz),  # Velocity along climb path
        player_ori=(ori_w, 0, ori_y, 0),  # Pitch for steady climb
        player_throttle=1.0,
    )

    # Run with zero elevator (pitch holds constant) and measure vz
    vzs = []
    speeds = []

    for step in range(1000):  # 20 seconds
        # Use state-based accessors (independent of obs_scheme)
        vz_now = get_vz_from_state(env)
        speed = get_speed_from_state(env)

        # Skip first 5 seconds for settling, then collect data
        if step >= 250:
            vzs.append(vz_now)
            speeds.append(speed)

        # Zero elevator - pitch angle holds due to rate-based controls
        action = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)
        if term[0]:
            break

    avg_vz = np.mean(vzs) if vzs else 0
    avg_speed = np.mean(speeds) if speeds else 0

    RESULTS['climb_rate'] = avg_vz
    diff = avg_vz - P51D_CLIMB_RATE
    status = "OK" if abs(diff) < 5 else "CHECK"
    print(f"climb_rate:    {avg_vz:6.1f} m/s  (P-51D: {P51D_CLIMB_RATE:.0f}, diff: {diff:+.1f}, speed: {avg_speed:.0f}/{Vy:.0f}) [{status}]")


def test_glide_ratio():
    """
    Power-off glide test - validates drag polar (Cd = Cd0 + K*Cl^2).

    At best glide speed, L/D is maximized. This occurs when induced drag
    equals parasitic drag (Cd0 = K*Cl^2).

    From our drag polar:
      Cl_opt = sqrt(Cd0/K) = sqrt(0.0163/0.072) = 0.476
      Cd_opt = 2*Cd0 = 0.0326
      L/D_max = Cl_opt/Cd_opt = 14.6

    Best glide speed: V = sqrt(2W/(rho*S*Cl)) = 80 m/s
    Glide angle: γ = arctan(1/L/D) = 3.9°
    Expected sink rate: V * sin(γ) = V/(L/D) = 5.5 m/s
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)

    # Calculate theoretical values from drag polar
    Cd0 = 0.0163
    K = 0.072
    W = 4082 * 9.81
    rho = 1.225
    S = 21.65
    C_L_alpha = 5.56
    alpha_zero = -0.021
    wing_inc = 0.026

    Cl_opt = np.sqrt(Cd0 / K)  # 0.476
    Cd_opt = 2 * Cd0           # 0.0326
    LD_max = Cl_opt / Cd_opt   # 14.6

    # Best glide speed
    V_glide = np.sqrt(2 * W / (rho * S * Cl_opt))  # ~80 m/s

    # Glide angle (nose below horizon for descent)
    gamma = np.arctan(1 / LD_max)  # ~3.9° = 0.068 rad

    # Expected sink rate
    sink_expected = V_glide * np.sin(gamma)  # ~5.5 m/s

    # AOA needed for Cl_opt
    alpha = Cl_opt / C_L_alpha - wing_inc + alpha_zero  # ~0.04 rad

    # In steady glide: body pitch = alpha - gamma (nose below velocity)
    # But our velocity is along glide path, so body pitch relative to horizontal = alpha - gamma
    # For quaternion: we want nose tilted down from horizontal
    pitch = alpha - gamma  # Negative = nose down

    # Create quaternion for glide attitude (negative because positive Y rotation = nose down)
    ori_w = np.cos(-pitch / 2)
    ori_y = np.sin(-pitch / 2)

    # Velocity along glide path (descending)
    vx = V_glide * np.cos(gamma)
    vz = -V_glide * np.sin(gamma)  # Negative = descending

    env.reset()
    setup_highlights(env, 'glide_ratio')
    env.force_state(
        player_pos=(0, 0, 2000),  # High altitude for long glide
        player_vel=(vx, 0, vz),
        player_ori=(ori_w, 0, ori_y, 0),
        player_throttle=0.0,
    )

    # Run with zero controls - let physics maintain steady glide
    vzs = []
    speeds = []

    for step in range(500):  # 10 seconds
        # Use state-based accessors (independent of obs_scheme)
        vz_now = get_vz_from_state(env)
        speed = get_speed_from_state(env)

        # Collect data after 2 seconds of settling
        if step >= 100:
            vzs.append(vz_now)
            speeds.append(speed)

        # Zero controls - pitch angle holds due to rate-based system
        action = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)
        if term[0]:
            break

    avg_vz = np.mean(vzs) if vzs else 0  # Should be negative (descending)
    avg_sink = -avg_vz  # Convert to positive sink rate
    avg_speed = np.mean(speeds) if speeds else 0
    measured_LD = avg_speed / avg_sink if avg_sink > 0.1 else 0

    RESULTS['glide_sink'] = avg_sink
    RESULTS['glide_LD'] = measured_LD

    diff = avg_sink - sink_expected
    status = "OK" if abs(diff) < 2 else "CHECK"
    print(f"glide_ratio:   L/D={measured_LD:4.1f}   (theory: {LD_max:.1f}, sink: {avg_sink:.1f} m/s, expected: {sink_expected:.1f}) [{status}]")


def test_sustained_turn():
    """
    Sustained turn test - verifies banked flight produces a turn.

    Tests that at 30° bank, 100 m/s:
      - Plane turns (heading changes)
      - Turn rate is positive and consistent
      - Altitude loss is bounded

    Note: The physics model produces ~2-3°/s at 30° bank (ideal theory: 3.2°/s).
    This is acceptable for RL training - the physics is consistent.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)

    # Test parameters - 30° bank is gentle and stable
    V = 100.0           # m/s
    bank_deg = 30.0     # degrees
    bank = np.radians(bank_deg)

    # Build quaternion: small pitch up, then bank right
    alpha = np.radians(3)  # Small fixed pitch for lift

    # Pitch (negative = nose up)
    qp_w = np.cos(-alpha / 2)
    qp_y = np.sin(-alpha / 2)

    # Roll (negative = bank right due to quaternion convention)
    qr_w = np.cos(-bank / 2)
    qr_x = np.sin(-bank / 2)

    # Combined: q = qr * qp
    ori_w = qr_w * qp_w
    ori_x = qr_x * qp_w
    ori_y = qr_w * qp_y
    ori_z = qr_x * qp_y

    env.reset()
    setup_highlights(env, 'sustained_turn')
    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(V, 0, 0),
        player_ori=(ori_w, ori_x, ori_y, ori_z),
        player_throttle=1.0,
    )

    # Run with zero controls
    headings = []
    speeds = []
    alts = []

    for step in range(250):  # 5 seconds
        state = env.get_state()
        vx, vy = state['vx'], state['vy']
        heading = np.arctan2(vy, vx)
        speed = np.sqrt(vx**2 + vy**2 + state['vz']**2)
        alt = state['pz']

        if step >= 50:  # After 1 second settling
            headings.append(heading)
            speeds.append(speed)
            alts.append(alt)

        action = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)
        if term[0]:
            break

    # Calculate turn rate
    if len(headings) > 50:
        headings = np.unwrap(headings)
        heading_change = headings[-1] - headings[0]
        time_elapsed = len(headings) * 0.02
        turn_rate_actual = np.degrees(heading_change / time_elapsed)
    else:
        turn_rate_actual = 0

    avg_speed = np.mean(speeds) if speeds else 0
    alt_change = alts[-1] - alts[0] if len(alts) > 1 else 0

    RESULTS['turn_rate'] = abs(turn_rate_actual)

    # Check: positive turn rate (plane is turning), not diving catastrophically
    is_turning = abs(turn_rate_actual) > 1.0
    alt_ok = alt_change > -200  # Less than 200m loss in 5 seconds
    status = "OK" if (is_turning and alt_ok) else "CHECK"

    print(f"turn_rate:     {abs(turn_rate_actual):5.1f}°/s ({bank_deg:.0f}° bank, speed: {avg_speed:.0f}, Δalt: {alt_change:+.0f}m) [{status}]")


def test_turn_60():
    """
    Coordinated turn at 60° bank with PID control.

    P-51D reference: 60° bank (2.0g) at 350 mph gives 5°/s
    At 100 m/s: theory = g*tan(60°)/V = 9.81*1.732/100 = 9.7°/s
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)

    bank_deg = 60.0
    bank_target = np.radians(bank_deg)
    V = 100.0

    # Right bank quaternion
    ori_w = np.cos(bank_target / 2)
    ori_x = -np.sin(bank_target / 2)

    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(V, 0, 0),
        player_ori=(ori_w, ori_x, 0.0, 0.0),
        player_throttle=1.0,
    )

    # PID gains (found via sweep in debug_turn.py)
    elev_kp, elev_kd = -0.05, 0.005
    roll_kp, roll_kd = -2.0, -0.1

    prev_vz = 0.0
    prev_bank_error = 0.0

    headings, alts, banks = [], [], []

    for step in range(250):  # 5 seconds
        # Get state from raw state (independent of obs_scheme)
        state = env.get_state()
        vz = state['vz']
        alt = state['pz']
        vx, vy = state['vx'], state['vy']
        heading = np.arctan2(vy, vx)
        up_y, up_z = state['up_y'], state['up_z']
        bank_actual = np.arccos(np.clip(up_z, -1, 1))
        if up_y < 0:
            bank_actual = -bank_actual

        # Elevator PID
        vz_error = -vz
        vz_deriv = (vz - prev_vz) / 0.02
        elevator = elev_kp * vz_error + elev_kd * vz_deriv
        elevator = np.clip(elevator, -1.0, 1.0)
        prev_vz = vz

        # Aileron PID
        bank_error = bank_target - bank_actual
        bank_deriv = (bank_error - prev_bank_error) / 0.02
        aileron = roll_kp * bank_error + roll_kd * bank_deriv
        aileron = np.clip(aileron, -1.0, 1.0)
        prev_bank_error = bank_error

        if step >= 25:
            headings.append(heading)
            alts.append(alt)
            banks.append(np.degrees(bank_actual))

        action = np.array([[1.0, elevator, aileron, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)
        if term[0]:
            break

    # Calculate results
    headings = np.unwrap(headings)
    turn_rate = np.degrees((headings[-1] - headings[0]) / (len(headings) * 0.02))
    alt_change = alts[-1] - alts[0]
    bank_mean = np.mean(banks)
    theory_rate = np.degrees(9.81 * np.tan(bank_target) / V)
    eff = 100 * turn_rate / theory_rate

    RESULTS['turn_rate_60'] = turn_rate

    status = "OK" if (85 < eff < 105 and abs(alt_change) < 50) else "CHECK"
    print(f"turn_60:       {turn_rate:5.1f}°/s (theory: {theory_rate:.1f}, eff: {eff:.0f}%, bank: {bank_mean:.0f}°, Δalt: {alt_change:+.0f}m) [{status}]")


def test_pitch_direction():
    """Verify positive elevator = nose DOWN (standard joystick: push forward)."""
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    env.force_state(player_vel=(80, 0, 0))

    # Get initial forward vector Z component (nose pointing direction)
    initial_fwd_z = env.get_state()['fwd_z']

    # Apply positive elevator (+1.0 = push forward)
    action = np.array([[0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    for step in range(50):
        env.step(action)

    # Check if nose went DOWN (fwd_z should decrease)
    final_fwd_z = env.get_state()['fwd_z']
    nose_down = final_fwd_z < initial_fwd_z  # fwd_z decreases when nose pitches down

    RESULTS['pitch_direction'] = 'DOWN' if nose_down else 'UP'
    status = 'OK' if nose_down else 'WRONG'
    print(f"pitch_dir:     {RESULTS['pitch_direction']:>6}      (+elev = nose DOWN) [{status}]")


def test_roll_direction():
    """Verify positive ailerons = roll right."""
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    env.force_state(player_vel=(80, 0, 0))

    action = np.array([[0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    for _ in range(50):
        env.step(action)
    state = env.get_state()
    up_y_changed = abs(state['up_y']) > 0.1
    RESULTS['roll_works'] = 'YES' if up_y_changed else 'NO'
    status = 'OK' if up_y_changed else 'WRONG'
    print(f"roll_works:    {RESULTS['roll_works']:>6}      (should be YES) [{status}]")


def test_rudder_only_turn():
    """
    Test: Wings level, nose on horizon, full rudder - verify limited heading change.

    Real rudder physics: deflection creates sideslip angle (not sustained yaw rate).
    Vertical tail creates restoring moment, limiting sideslip to ~10 degrees.
    Once equilibrium sideslip is reached, yaw rate approaches zero.

    Expected behavior:
    - Initial yaw rate is high (MAX_YAW_RATE ~29 deg/s)
    - Yaw rate decays as sideslip builds
    - Total heading change is LIMITED to ~10-15 degrees
    - Cannot turn around with just rudder

    This test uses PID control to:
    - Hold wings level (ailerons fight any roll)
    - Hold nose on horizon (elevator maintains level flight)
    - Apply full rudder and measure total heading change
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()
    setup_highlights(env, 'rudder_only_turn')

    # Start at cruise speed, wings level
    V = 120.0  # m/s cruise
    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(V, 0, 0),
        player_ori=(1.0, 0.0, 0.0, 0.0),  # Identity = wings level, heading +X
        player_throttle=1.0,
    )

    # PID gains for wings level (tuned to stay stable with full rudder)
    roll_kp = 1.0   # Proportional - lower prevents oscillation
    roll_kd = 0.05  # Derivative damping

    # PID gains for level flight (from existing tests)
    elev_kp = 0.001
    elev_kd = 0.001

    prev_roll = 0.0
    prev_vz = 0.0

    headings = []

    for step in range(300):  # 6 seconds at 50Hz
        # Extract state from raw state (independent of obs_scheme)
        state = env.get_state()
        vx, vy, vz = state['vx'], state['vy'], state['vz']
        up_y, up_z = state['up_y'], state['up_z']

        # Calculate heading from velocity
        heading = np.arctan2(vy, vx)
        headings.append(heading)

        # Calculate roll angle from up vector
        roll = np.arctan2(up_y, up_z)

        # Wings level PID: drive roll to zero
        roll_error = 0.0 - roll
        roll_deriv = (roll - prev_roll) / 0.02
        aileron = roll_kp * roll_error - roll_kd * roll_deriv
        aileron = np.clip(aileron, -1.0, 1.0)
        prev_roll = roll

        # Level flight PID: drive vz to zero
        vz_error = 0.0 - vz
        vz_deriv = (vz - prev_vz) / 0.02
        elevator = -elev_kp * vz_error - elev_kd * vz_deriv
        elevator = np.clip(elevator, -0.3, 0.3)
        prev_vz = vz

        # FULL RUDDER
        rudder = 1.0

        # Action: [throttle, elevator, aileron, rudder, trigger]
        action = np.array([[1.0, elevator, aileron, rudder, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)

        if term[0]:
            break

    # Analyze heading change
    headings = np.unwrap(headings)  # Handle wraparound
    total_heading_change_deg = np.degrees(headings[-1] - headings[0]) if len(headings) > 1 else 0

    # Calculate initial yaw rate (first 0.5 seconds = 25 steps)
    if len(headings) > 25:
        initial_change = headings[25] - headings[0]
        initial_yaw_rate_deg_s = np.degrees(initial_change / 0.5)
    else:
        initial_yaw_rate_deg_s = 0

    # Calculate final yaw rate (last 2 seconds)
    if len(headings) > 200:
        final_change = headings[-1] - headings[-100]
        final_yaw_rate_deg_s = np.degrees(final_change / 2.0)
    else:
        final_yaw_rate_deg_s = 0

    RESULTS['rudder_total_heading'] = total_heading_change_deg
    RESULTS['rudder_initial_rate'] = initial_yaw_rate_deg_s
    RESULTS['rudder_final_rate'] = final_yaw_rate_deg_s

    # Verify damping behavior:
    # Real rudder physics: heading changes slowly because rudder creates sideslip,
    # NOT a direct heading rate. The sideforce from sideslip is what turns the velocity.
    #
    # Expected behavior:
    # 1. Total heading change should be limited and small (~3-15 degrees)
    #    - Rudder can't spin the plane around, it's a small control
    # 2. Heading changes at all (rudder has SOME effect)
    # 3. Final rate should be similar to initial (slow, steady turn from sideslip)
    #
    # Note: In a P-51D, full rudder at cruise gives ~5-10° sideslip and very slow turn
    heading_changed = abs(total_heading_change_deg) > 2.0  # Rudder does something
    heading_limited = abs(total_heading_change_deg) < 20.0  # Can't do unlimited turns

    is_realistic = heading_changed and heading_limited
    status = "OK" if is_realistic else "FAIL"

    print(f"rudder_only:   heading={total_heading_change_deg:5.1f}° (2-20° OK), "
          f"initial={initial_yaw_rate_deg_s:5.1f}°/s, final={final_yaw_rate_deg_s:4.1f}°/s [{status}]")

    if not is_realistic:
        if not heading_changed:
            print(f"  ISSUE: Rudder should change heading (got only {total_heading_change_deg:.1f}°)")
        if not heading_limited:
            print(f"  ISSUE: Heading change should be <20°, got {total_heading_change_deg:.1f}°")


def test_knife_edge_pull():
    """
    Knife-edge pull test - validates that elevator becomes YAW when rolled 90°.

    Physics explanation:
    - Plane rolled 90° right: right wing DOWN, canopy facing RIGHT
    - Body axes after roll:
      - Body X (nose): +X world (forward)
      - Body Y (right wing): -Z world (DOWN)
      - Body Z (canopy): +Y world (RIGHT)
    - Negative elevator (pull back) = pitch up in BODY frame = rotation about body Y
    - Body Y is now -Z world, so this is rotation about world -Z
    - Right-hand rule: thumb on -Z, fingers curl +X toward -Y
    - Result: Nose yaws LEFT in world frame (since we pull back = negative elevator)

    Expected behavior:
    1. Heading changes significantly (plane turns left with pull back)
    2. Altitude drops (lift is horizontal, not vertical)
    3. Up vector stays roughly horizontal (still in knife-edge)
    4. This is essentially a "flat turn" using elevator

    This tests that the quaternion kinematics correctly transform body-frame
    rotations to world-frame effects.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()
    setup_highlights(env, 'knife_edge_pull')

    # Start at high speed to avoid stall during the pull
    V = 150.0  # m/s - well above stall speed even at high AoA

    # Use EXACT 90° right roll via force_state for precise test
    # Roll -90° about X axis: q = (cos(45°), -sin(45°), 0, 0)
    roll_90 = np.radians(90)
    qw = np.cos(roll_90 / 2)
    qx = -np.sin(roll_90 / 2)  # Negative for right roll

    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(V, 0, 0),  # Flying +X
        player_ori=(qw, qx, 0.0, 0.0),  # EXACT 90° right roll
        player_throttle=1.0,
    )

    # Verify knife-edge achieved
    state = env.get_state()
    up_x, up_y, up_z = state['up_x'], state['up_y'], state['up_z']

    # Record initial state
    alt_start = state['pz']
    vx_start, vy_start = state['vx'], state['vy']
    heading_start = np.arctan2(vy_start, vx_start)

    # --- Phase 2: Full elevator pull in knife-edge ---
    headings = []
    alts = []
    up_zs = []

    for step in range(100):  # 2 seconds
        state = env.get_state()
        vx, vy, vz = state['vx'], state['vy'], state['vz']
        heading = np.arctan2(vy, vx)
        alt = state['pz']
        up_z_now = state['up_z']

        headings.append(heading)
        alts.append(alt)
        up_zs.append(up_z_now)

        # Full throttle, FULL ELEVATOR PULL, no aileron, no rudder
        # Convention: -elevator = pull back = nose up
        action = np.array([[1.0, -1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)
        if term[0]:
            break

    # --- Analysis ---
    headings = np.unwrap(headings)
    heading_change = np.degrees(headings[-1] - headings[0])
    alt_loss = alt_start - alts[-1]
    avg_up_z = np.mean(up_zs)
    time_elapsed = len(headings) * 0.02

    # Calculate turn rate
    turn_rate = heading_change / time_elapsed if time_elapsed > 0 else 0

    RESULTS['knife_pull_turn'] = turn_rate
    RESULTS['knife_pull_alt_loss'] = alt_loss

    # Expected:
    # 1. Significant heading change (turns left with pull back, so negative)
    # 2. Altitude loss (no vertical lift)
    # 3. Up vector stays near horizontal (|up_z| small)

    # In our coordinate system: X forward, Y left, Z up
    # atan2(vy, vx) increases when turning left (positive vy)
    heading_ok = heading_change > 20  # Should turn at least 20° left in 2 seconds
    alt_ok = alt_loss > 5  # Should lose altitude
    roll_maintained = abs(avg_up_z) < 0.3  # Up vector stays roughly horizontal

    all_ok = heading_ok and alt_ok and roll_maintained
    status = "OK" if all_ok else "CHECK"

    # Positive heading change = LEFT turn (Y is left in our coords)
    direction = "LEFT" if heading_change > 0 else "RIGHT"
    print(f"knife_pull:    turn={turn_rate:+.1f}°/s ({direction}), alt_lost={alt_loss:.0f}m, |up_z|={abs(avg_up_z):.2f} [{status}]")

    if not heading_ok:
        print(f"  WARNING: Expected significant left turn, got {heading_change:.1f}° heading change")
    if not alt_ok:
        print(f"  WARNING: Expected altitude loss, got {alt_loss:.1f}m")
    if not roll_maintained:
        print(f"  WARNING: Roll not maintained, up_z={avg_up_z:.2f} (should be near 0)")


def test_knife_edge_flight():
    """
    Knife-edge flight test - validates that the plane CANNOT maintain altitude.

    In knife-edge flight (90° roll), the wings are vertical and generate
    NO vertical lift. The plane must rely on:
    1. Fuselage side area (very inefficient, NOT modeled)
    2. Rudder sideforce (NOT modeled - rudder only creates yaw rate)
    3. Thrust vector (only if nosed up significantly)

    A P-51D is NOT designed for knife-edge - streamlined fuselage = poor side area.
    Even purpose-built aerobatic planes struggle to maintain altitude in true knife-edge.

    Expected behavior: Plane should lose altitude rapidly (~9 m/s sink or more).
    The nose may yaw from rudder input, but vertical force is insufficient.

    Sources:
    - https://www.thenakedscientists.com/articles/questions/what-produces-lift-during-knife-edge-pass
    - https://www.aopa.org/news-and-media/all-news/1998/august/flight-training-magazine/form-and-function
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()
    setup_highlights(env, 'knife_edge_flight')

    # Start at cruise speed, wings level, flying +X
    V = 120.0  # m/s - fast enough for good control authority
    env.force_state(
        player_pos=(0, 0, 1500),  # High altitude for test duration
        player_vel=(V, 0, 0),      # Flying +X direction
        player_ori=(1.0, 0.0, 0.0, 0.0),  # Wings level
        player_throttle=1.0,
    )

    # --- Phase 1: Roll to knife-edge (90° right) ---
    # Takes about 30 steps at MAX_ROLL_RATE=3.0 rad/s (0.5s to roll 90°)
    for step in range(30):
        # Full right aileron to roll 90°
        action = np.array([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        env.step(action)

    # Verify we're in knife-edge (up vector should be pointing +Y or -Y)
    state = env.get_state()
    up_y, up_z = state['up_y'], state['up_z']
    roll_deg = np.degrees(np.arccos(np.clip(up_z, -1, 1)))

    # Record altitude at start of knife-edge
    alt_start = state['pz']

    if abs(roll_deg - 90) > 15:
        print(f"knife_edge: [SKIP] Failed to roll to 90° (got {roll_deg:.0f}°)")
        return

    # --- Phase 2: Knife-edge with top rudder ---
    # Right wing is down (up_y < 0 means rolled right)
    # Negative rudder = yaw LEFT in body frame
    # In knife-edge, body-left is world-up, so this tries to pitch nose up

    alts = []
    vzs = []

    for step in range(150):  # 3 seconds at 50Hz
        state = env.get_state()
        alt = state['pz']
        vz = state['vz']
        alts.append(alt)
        vzs.append(vz)

        # Full throttle, no elevator, no aileron (hold knife-edge), TOP RUDDER
        # Negative rudder = yaw LEFT in body frame
        # In knife-edge (rolled 90° right), body-left is world-up
        # So this SHOULD help keep nose up... if rudder created sideforce
        action = np.array([[1.0, 0.0, 0.0, -1.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)
        if term[0]:
            break

    alt_end = alts[-1] if alts else alt_start
    alt_loss = alt_start - alt_end
    avg_vz = np.mean(vzs) if vzs else 0
    time_elapsed = len(alts) * 0.02  # seconds

    # Calculate sink rate
    sink_rate = alt_loss / time_elapsed if time_elapsed > 0 else 0

    RESULTS['knife_edge_sink'] = sink_rate
    RESULTS['knife_edge_alt_loss'] = alt_loss

    # Expected: significant altitude loss
    # At 1g downward acceleration: v = g*t = 9.81 * 3 = 29 m/s after 3s
    # Distance = 0.5 * g * t^2 = 0.5 * 9.81 * 9 = 44 m (free fall)
    # With some lift from thrust vector angle, maybe 20-30m loss
    # If plane CAN maintain altitude (loss < 5m), physics is WRONG

    is_realistic = alt_loss > 10  # Should lose at least 10m in 3 seconds
    status = "OK" if is_realistic else "FAIL - physics allows impossible knife-edge!"

    print(f"knife_edge:    sink={sink_rate:5.1f} m/s, alt_lost={alt_loss:.0f}m in {time_elapsed:.1f}s [{status}]")

    if not is_realistic:
        print(f"  WARNING: P-51D should NOT maintain altitude in knife-edge!")
        print(f"  Wings are vertical = no lift. Rudder only creates yaw, not sideforce.")
        print(f"  Consider: Is thrust somehow pointing upward? Is there phantom lift?")


def test_mode_weights():
    """
    Test that mode_weights actually biases autopilot randomization.

    Sets 100% weight on AP_LEVEL, triggers multiple resets,
    verifies that selected mode is always AP_LEVEL.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Set AP_RANDOM mode and bias 100% toward LEVEL
    env.set_autopilot(env_idx=0, mode=AutopilotMode.RANDOM)
    env.set_mode_weights(level=1.0, turn_left=0.0, turn_right=0.0, climb=0.0, descend=0.0)

    # Trigger multiple resets and check mode each time
    level_count = 0
    num_trials = 50

    for _ in range(num_trials):
        env.reset()
        mode = env.get_autopilot_mode(env_idx=0)
        if mode == AutopilotMode.LEVEL:
            level_count += 1

    pct = 100 * level_count / num_trials
    RESULTS['mode_weights'] = pct

    # With 100% weight on LEVEL, should always get LEVEL
    status = "OK" if pct == 100 else "CHECK"
    print(f"mode_weights:  {pct:5.1f}%   (should be 100% AP_LEVEL) [{status}]")

    # Also test distribution with mixed weights
    env.set_autopilot(env_idx=0, mode=AutopilotMode.RANDOM)  # Re-enable randomization
    env.set_mode_weights(level=0.5, turn_left=0.25, turn_right=0.25, climb=0.0, descend=0.0)

    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # LEVEL, TURN_L, TURN_R, CLIMB, DESCEND
    num_trials = 200

    for _ in range(num_trials):
        env.reset()
        mode = env.get_autopilot_mode(env_idx=0)
        if mode in counts:
            counts[mode] += 1

    # Check that LEVEL is most common (~50%) and CLIMB/DESCEND are rare (~0%)
    level_pct = 100 * counts[1] / num_trials
    climb_pct = 100 * counts[4] / num_trials
    distribution_ok = level_pct > 35 and climb_pct < 10
    status2 = "OK" if distribution_ok else "CHECK"
    print(f"  distribution: LEVEL={level_pct:.0f}%, TURN_L={100*counts[2]/num_trials:.0f}%, TURN_R={100*counts[3]/num_trials:.0f}%, CLIMB={climb_pct:.0f}% [{status2}]")


# =============================================================================
# G-FORCE TESTS - Validate G-loading physics
# =============================================================================

def test_g_level_flight():
    """
    Level flight at cruise speed - verify G ≈ 1.0.
    In steady level flight, lift equals weight, so G-loading should be ~1.0.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Start at cruise speed, level
    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(120, 0, 0),
        player_throttle=0.5,
    )

    g_values = []
    for step in range(200):  # 4 seconds
        elevator = level_flight_pitch_from_state(env)
        action = np.array([[0.0, elevator, 0.0, 0.0, 0.0]], dtype=np.float32)
        env.step(action)

        g = env.get_state()['g_force']
        g_values.append(g)

        if step % 25 == 0:
            print(f"  step {step:3d}: G = {g:.2f}")

    avg_g = np.mean(g_values[-100:])  # Last 2 seconds
    RESULTS['g_level'] = avg_g

    status = "OK" if 0.8 < avg_g < 1.2 else "CHECK"
    print(f"g_level:       {avg_g:.2f} G  (target: ~1.0) [{status}]")


def test_g_push_forward():
    """
    Push elevator forward - verify G decreases toward 0 and negative.
    Reset to level flight for each test to avoid looping artifacts.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)

    print("  Pushing forward (positive elevator = nose down):")
    min_g = float('inf')

    for elev in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Reset to level flight for each elevator setting
        env.reset()
        env.force_state(
            player_pos=(0, 0, 1500),
            player_vel=(150, 0, 0),
            player_throttle=1.0,
        )

        # Run for 10 steps (0.2 sec) and track min G
        test_min_g = float('inf')
        for _ in range(10):
            action = np.array([[1.0, elev, 0.0, 0.0, 0.0]], dtype=np.float32)
            env.step(action)
            g = env.get_state()['g_force']
            test_min_g = min(test_min_g, g)

        min_g = min(min_g, test_min_g)
        print(f"    elevator={elev:+.2f}: min G = {test_min_g:+.2f}")

    RESULTS['g_push'] = min_g

    # Full push should give low/negative G
    status = "OK" if min_g < 0.5 else "CHECK"
    print(f"g_push:        {min_g:+.2f} G  (push should give < 0.5G) [{status}]")


def test_g_pull_back():
    """
    Pull elevator back - verify G increases above 1.0.
    Reset to level flight for each test to avoid looping artifacts.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)

    print("  Pulling back (negative elevator = nose up):")
    max_g = float('-inf')

    for elev in [0.0, -0.25, -0.5, -0.75, -1.0]:
        # Reset to level flight for each elevator setting
        env.reset()
        env.force_state(
            player_pos=(0, 0, 1500),
            player_vel=(150, 0, 0),  # Higher speed for more G capability
            player_throttle=1.0,
        )

        # Run for 10 steps (0.2 sec) and track max G
        test_max_g = float('-inf')
        for _ in range(10):
            action = np.array([[1.0, elev, 0.0, 0.0, 0.0]], dtype=np.float32)
            env.step(action)
            g = env.get_state()['g_force']
            test_max_g = max(test_max_g, g)

        max_g = max(max_g, test_max_g)
        print(f"    elevator={elev:+.2f}: max G = {test_max_g:+.2f}")

    RESULTS['g_pull'] = max_g

    # Full pull should give high G (at 150 m/s, should hit ~5-6G)
    status = "OK" if max_g > 4.0 else "CHECK"
    print(f"g_pull:        {max_g:+.2f} G  (pull should give > 4.0G) [{status}]")


def test_g_limit_negative():
    """
    Full forward stick - verify G never goes below -1.5G (G_LIMIT_NEG).
    Physics should clamp acceleration to prevent exceeding this limit.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Start at high speed for maximum control authority
    env.force_state(
        player_pos=(0, 0, 2000),
        player_vel=(150, 0, 0),
        player_throttle=1.0,
    )

    g_min = float('inf')
    for step in range(150):  # 3 seconds of full push
        action = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # Full forward
        env.step(action)

        g = env.get_state()['g_force']
        g_min = min(g_min, g)

        if step % 25 == 0:
            print(f"  step {step:3d}: G = {g:+.2f} (min so far: {g_min:+.2f})")

    RESULTS['g_min'] = g_min

    # Should never go below -1.5G (with small tolerance)
    G_LIMIT_NEG = -1.5
    status = "OK" if g_min >= G_LIMIT_NEG - 0.1 else "FAIL"
    print(f"g_limit_neg:   {g_min:+.2f} G  (limit: {G_LIMIT_NEG}G) [{status}]")
    assert g_min >= G_LIMIT_NEG - 0.1, f"G went below limit: {g_min} < {G_LIMIT_NEG}"


def test_g_limit_positive():
    """
    Full back stick - verify G never exceeds 6G (G_LIMIT_POS).
    Physics should clamp acceleration to prevent exceeding this limit.
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Start at high speed for maximum G capability
    env.force_state(
        player_pos=(0, 0, 2000),
        player_vel=(180, 0, 0),  # Very fast
        player_throttle=1.0,
    )

    g_max = float('-inf')
    for step in range(150):  # 3 seconds of full pull
        action = np.array([[1.0, -1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # Full pull
        env.step(action)

        g = env.get_state()['g_force']
        g_max = max(g_max, g)

        if step % 25 == 0:
            print(f"  step {step:3d}: G = {g:+.2f} (max so far: {g_max:+.2f})")

    RESULTS['g_max'] = g_max

    # Should never exceed 6G (with small tolerance)
    G_LIMIT_POS = 6.0
    status = "OK" if g_max <= G_LIMIT_POS + 0.1 else "FAIL"
    print(f"g_limit_pos:   {g_max:+.2f} G  (limit: {G_LIMIT_POS}G) [{status}]")
    assert g_max <= G_LIMIT_POS + 0.1, f"G exceeded limit: {g_max} > {G_LIMIT_POS}"


def test_gentle_pitch_control():
    """
    Test that small elevator inputs produce proportional, gentle pitch changes.

    This is CRITICAL for fine aim adjustments - the agent must be able to make
    precise 2.5° corrections, not just bang-bang full deflection.

    Tests:
    1. -0.1 elevator: should give small pitch rate (~5°/s or less)
    2. -0.25 elevator: should give larger pitch rate (~10-15°/s)
    3. Verify linear relationship (not bang-bang)
    4. Calculate time to make 2.5° adjustment
    """
    env = Dogfight(num_envs=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)

    elevator_values = [-0.05, -0.1, -0.15, -0.2, -0.25, -0.3]
    pitch_rates = []

    print("  Testing gentle elevator inputs (negative = pull back = nose UP):")

    for elev in elevator_values:
        env.reset()

        # Start level at cruise speed
        env.force_state(
            player_pos=(0, 0, 1500),
            player_vel=(120, 0, 0),  # Cruise speed
            player_ori=(1.0, 0.0, 0.0, 0.0),  # Wings level
            player_throttle=0.7,
        )

        # Record initial pitch
        state = env.get_state()
        fwd_x_start, fwd_z_start = state['fwd_x'], state['fwd_z']
        pitch_start = np.arctan2(fwd_z_start, fwd_x_start)

        # Apply constant elevator for 1 second (50 steps)
        for step in range(50):
            action = np.array([[0.4, elev, 0.0, 0.0, 0.0]], dtype=np.float32)
            env.step(action)

        # Measure final pitch
        state = env.get_state()
        fwd_x_end, fwd_z_end = state['fwd_x'], state['fwd_z']
        pitch_end = np.arctan2(fwd_z_end, fwd_x_end)

        pitch_change_deg = np.degrees(pitch_end - pitch_start)
        pitch_rate = pitch_change_deg / 1.0  # degrees per second
        pitch_rates.append(pitch_rate)

        print(f"    elevator={elev:+.2f}: pitch_rate={pitch_rate:+.1f}°/s, pitch_change={pitch_change_deg:+.1f}°")

    # Check for proportional response
    # Ratio of pitch rates should roughly match ratio of elevator inputs
    rate_at_01 = pitch_rates[1]  # -0.1 elevator
    rate_at_025 = pitch_rates[4]  # -0.25 elevator

    # Store results
    RESULTS['pitch_rate_01'] = rate_at_01
    RESULTS['pitch_rate_025'] = rate_at_025

    # Calculate time to make 2.5° adjustment at -0.1 elevator
    if abs(rate_at_01) > 0.1:
        time_for_25deg = 2.5 / abs(rate_at_01)
    else:
        time_for_25deg = float('inf')

    RESULTS['time_for_25deg'] = time_for_25deg

    # Check proportionality: -0.25 should give ~2.5x the rate of -0.1
    expected_ratio = 2.5
    actual_ratio = rate_at_025 / rate_at_01 if abs(rate_at_01) > 0.1 else 0

    # Verify reasonable pitch rates (not too fast, not too slow)
    # -0.1 elevator should give roughly 3-8°/s (gentle but noticeable)
    gentle_ok = 2.0 < abs(rate_at_01) < 15.0
    proportional_ok = 1.5 < actual_ratio < 4.0  # Some non-linearity is OK
    can_aim = time_for_25deg < 2.0  # Should be able to make 2.5° adjustment in <2 seconds

    all_ok = gentle_ok and proportional_ok and can_aim
    status = "OK" if all_ok else "CHECK"

    print(f"  Results:")
    print(f"    -0.1 elevator gives {rate_at_01:+.1f}°/s (want 3-8°/s) [{gentle_ok and 'OK' or 'CHECK'}]")
    print(f"    -0.25/-0.1 ratio = {actual_ratio:.2f} (want ~2.5, linear) [{proportional_ok and 'OK' or 'CHECK'}]")
    print(f"    Time to adjust 2.5° at -0.1: {time_for_25deg:.2f}s (want <2s) [{can_aim and 'OK' or 'CHECK'}]")
    print(f"gentle_pitch:  rate@-0.1={rate_at_01:+.1f}°/s, 2.5°_time={time_for_25deg:.2f}s [{status}]")

    if not gentle_ok:
        if abs(rate_at_01) < 2.0:
            print(f"  WARNING: Pitch too sluggish! Agent can't make timely aim corrections.")
        else:
            print(f"  WARNING: Pitch too sensitive! Agent will overshoot aim.")

    if not proportional_ok:
        print(f"  WARNING: Non-linear pitch response - may indicate bang-bang controls.")


# =============================================================================
# OBSERVATION SCHEME TESTS
# =============================================================================

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
        env = Dogfight(num_envs=1, obs_scheme=scheme, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)

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
    env = Dogfight(num_envs=1, obs_scheme=2, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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


# =============================================================================
# DYNAMIC MANEUVER OBSERVATION TESTS
# =============================================================================

def test_obs_during_loop():
    """
    Full inside loop maneuver - verify observations during complete pitch cycle.

    Purpose: Ensure Euler angle observations (pitch) smoothly transition through
    full range [-1, 1] during a loop without discontinuities.

    Expected behavior:
    - Pitch sweeps through full range (0 → -0.5 (nose up 90°) → ±1 (inverted) → +0.5 → 0)
    - Roll stays near 0 throughout (wings level loop)
    - No sudden jumps in any observation (discontinuity = bug)

    This tests the quaternion→euler conversion under continuous rotation.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    Full 360° aileron roll - verify roll and horizon_visible observations.

    Purpose: Ensure roll observation smoothly transitions through ±180° without
    discontinuity, and horizon_visible follows expected pattern.

    Expected behavior (scheme 2):
    - Roll: 0 → -1 (90° right) → ±1 (inverted wrap) → +1 (270°) → 0
    - horizon_visible: 1 → 0 → -1 → 0 → 1

    The ±180° crossover is the critical test - if there's a wrap bug,
    roll will jump from +1 to -1 instantly instead of smoothly transitioning.
    """
    env = Dogfight(num_envs=1, obs_scheme=2, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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

    # Roll at MAX_ROLL_RATE=3.0 rad/s = 172°/s, so 360° takes ~2.1 seconds = 105 steps
    for step in range(120):  # ~2.4 seconds for full 360° with margin
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
    # 1. Roll covers most of range (near ±1)
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
    Vertical pitch (±90°) gimbal lock detection test.

    Purpose: Detect gimbal lock behavior when pitch reaches ±90° where
    the euler angle representation becomes singular.

    At pitch = ±90°:
    - roll = atan2(2*(w*x + y*z), 1 - 2*(x² + y²)) becomes undefined
    - May cause roll to snap/oscillate wildly

    This documents the behavior rather than asserting specific values,
    since gimbal lock is a known limitation of euler angles.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Test nose straight up (90° pitch)
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

    # Test nose straight down (-90° pitch)
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

    # Pitch should be near ±0.5 (90°/180° = 0.5)
    pitch_up_ok = abs(abs(pitch_up) - 0.5) < 0.15
    pitch_down_ok = abs(abs(pitch_down) - 0.5) < 0.15

    RESULTS['obs_vertical'] = all_bounded
    status = "OK" if all_bounded else "WARN"

    print(f"obs_vertical:  up=(pitch={pitch_up:.3f}, roll={roll_up:.3f}), down=(pitch={pitch_down:.3f}, roll={roll_down:.3f}) [{status}]")

    if not pitch_up_ok:
        print(f"    NOTE: Pitch up {pitch_up:.3f} not near ±0.5 (expected for 90° pitch)")
    if not pitch_down_ok:
        print(f"    NOTE: Pitch down {pitch_down:.3f} not near ±0.5")
    if not all_bounded:
        print(f"    WARNING: Observations out of bounds or NaN at vertical pitch")
    if abs(roll_up) > 0.3 or abs(roll_down) > 0.3:
        print(f"    NOTE: Roll unstable at vertical pitch (gimbal lock region)")

    env.close()
    return all_bounded


def test_obs_azimuth_crossover():
    """
    Target azimuth ±180° crossover test.

    Purpose: Verify azimuth doesn't jump discontinuously when target
    crosses from behind-left to behind-right (through ±180°).

    Risk: Azimuth might jump from +1 to -1 instantly instead of transitioning
    smoothly, causing RL agent to see huge observation delta.

    Test: Sweep opponent from right-behind through directly-behind to left-behind
    and check for discontinuities.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    azimuths = []
    y_positions = []

    # Sweep opponent from right-behind (y=-200) through left-behind (y=+200)
    # This forces azimuth to cross through ±180° (behind the player)
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

    # Verify azimuth range covers ±1 (behind = ±180°)
    az_min = min(azimuths)
    az_max = max(azimuths)
    range_ok = az_max > 0.8 and az_min < -0.8

    # Discontinuity at ±180° crossover is EXPECTED for atan2-based azimuth
    # This test documents the behavior - a discontinuity here is not necessarily
    # a bug, but agents should be aware of it
    has_discontinuity = len(azimuth_jumps) > 0

    RESULTS['obs_azimuth_cross'] = range_ok
    status = "OK" if range_ok else "CHECK"

    print(f"obs_az_cross:  range=[{az_min:.2f},{az_max:.2f}], discontinuities={len(azimuth_jumps)} [{status}]")

    if has_discontinuity:
        print(f"    NOTE: Azimuth has discontinuity at ±180° (expected for atan2)")
        for _, y_pos, prev_az, curr_az, delta in azimuth_jumps[:2]:
            print(f"    At y={y_pos:.0f}: azimuth {prev_az:.2f} -> {curr_az:.2f} (delta={delta:.2f})")
        print(f"    Consider: Use sin/cos encoding to avoid wrap-around for RL")

    if not range_ok:
        print(f"    WARNING: Azimuth didn't reach ±1 (behind player)")

    env.close()
    return range_ok


def test_obs_yaw_wrap():
    """
    Yaw observation ±180° wrap test.

    Purpose: Verify yaw observation behavior when heading crosses ±180°.
    Tests CONTINUOUS heading transition across the wrap boundary.

    The critical test: sweep from +170° to -170° (crossing +180°/-180°).
    If yaw wraps, we'll see a jump from ~+1 to ~-1.

    For RL, yaw wrap at ±180° is less problematic than roll wrap because:
    - Normal flight rarely involves facing directly backwards
    - Roll wrap happens during inverted flight (loops, barrel rolls)
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    yaws = []
    headings = []

    # Test 1: Sweep ACROSS the ±180° boundary (170° to 190° = -170°)
    # This is the critical test - continuous transition through the wrap point
    for heading_deg in range(170, 195, 2):  # 170° to 194° in 2° steps
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

    # Check for discontinuities at the ±180° crossing
    yaw_jumps = []
    for i in range(1, len(yaws)):
        delta = abs(yaws[i] - yaws[i-1])
        if delta > 0.3:  # 2° step should give ~0.022 change, 0.3 is a big jump
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
        print(f"    WRAP DETECTED at ±180° heading:")
        for h1, h2, y1, y2, delta in yaw_jumps[:2]:
            print(f"    heading {h1}°→{h2}°: yaw {y1:.2f} -> {y2:.2f} (delta={delta:.2f})")
        print(f"    Consider: Use sin/cos encoding for yaw to avoid wrap")
    else:
        print(f"    No discontinuity at ±180° crossing (yaw: {yaw_min:.2f} to {yaw_max:.2f})")

    env.close()
    return range_ok


def test_obs_elevation_extremes():
    """
    Elevation observation at ±90° (target directly above/below).

    Purpose: Verify elevation doesn't have singularity when target is
    directly above or below player. Elevation uses asin which is bounded
    by definition, so this should be stable.

    Test: Place target directly above and below player, verify elevation
    is correct and bounded.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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

    Non-unit quaternion → incorrect euler angles → bad observations.
    """
    env = Dogfight(num_envs=1, obs_scheme=0, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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


# =============================================================================
# OBS_PURSUIT (SCHEME 1) TESTS
# =============================================================================
# Observation layout for OBS_PURSUIT (13 observations):
#   0: speed        - clamp(speed/250, 0, 1)           [0, 1]
#   1: potential    - alt/3000                         [0, 1]
#   2: pitch        - pitch / (PI/2)                   [-1, 1]
#   3: roll         - roll / PI                        [-1, 1]  **WRAPS**
#   4: own_energy   - (potential + kinetic) / 2        [0, 1]
#   5: target_az    - target_az / PI                   [-1, 1]  **WRAPS**
#   6: target_el    - target_el / (PI/2)               [-1, 1]
#   7: dist         - clamp(dist/500, 0, 2) - 1        [-1, 1]
#   8: closure      - clamp(closure/250, -1, 1)        [-1, 1]
#   9: target_roll  - target_roll / PI                 [-1, 1]  **WRAPS**
#  10: target_pitch - target_pitch / (PI/2)            [-1, 1]
#  11: target_aspect- dot(opp_fwd, to_player)          [-1, 1]
#  12: energy_adv   - clamp(own_E - opp_E, -1, 1)      [-1, 1]


def test_obs_pursuit_bounds():
    """
    Run random maneuvers in OBS_PURSUIT (scheme 1) and verify all observations
    stay in valid ranges. This catches NaN/Inf/out-of-bounds issues.

    OBS_PURSUIT has 13 observations with specific bounds:
    - Indices 0, 1, 4: [0, 1] (speed, potential, own_energy)
    - All others: [-1, 1]
    """
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # 90° pitch, 100 m/s, low throttle
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
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    env.reset()

    # Start high, pitch down 45°
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
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
    action = np.array([[0.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # Some throttle

    # Head-on: opponent facing toward player (yaw=180° = facing -X)
    # Quaternion for yaw=180°: qw=0, qz=1
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(-100, 0, 0),
        opponent_ori=(0, 0, 0, 1),  # Yaw=180° = facing -X (toward player)
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

    # Beam: opponent perpendicular (yaw=-90° = facing +Y)
    # Quaternion for yaw=-90°: qw=cos(-45°)≈0.707, qz=sin(-45°)≈-0.707
    cos45 = np.cos(np.radians(-45))
    sin45 = np.sin(np.radians(-45))
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(0, 100, 0),
        opponent_ori=(cos45, 0, 0, sin45),  # Yaw=-90° = facing +Y
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
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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
    # Opponent facing -X (toward player): yaw=180° → qw=0, qz=1
    env.reset()
    env.force_state(
        player_pos=(0, 0, 1500), player_vel=(100, 0, 0),
        opponent_pos=(500, 0, 1500), opponent_vel=(-100, 0, 0),
        opponent_ori=(0, 0, 0, 1),  # Yaw=180° = facing -X
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

    Sweep target position around player (behind the player through ±180°)
    and check for large discontinuities in target_az.
    """
    env = Dogfight(num_envs=1, obs_scheme=1, render_mode=RENDER_MODE, render_fps=RENDER_FPS)
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

    # Verify azimuth range covers near ±1 (behind = ±180°)
    az_min = min(target_azs)
    az_max = max(target_azs)
    range_ok = az_max > 0.8 and az_min < -0.8

    # Discontinuity at ±180° crossover is EXPECTED for atan2-based azimuth
    has_discontinuity = len(az_jumps) > 0

    RESULTS['obs_pursuit_az_wrap'] = range_ok
    status = "OK" if range_ok else "CHECK"

    print(f"obs_pursuit_az_wrap: range=[{az_min:.2f},{az_max:.2f}], discontinuities={len(az_jumps)} [{status}]")

    if has_discontinuity:
        print(f"    NOTE: target_az has discontinuity at ±180° (expected for atan2)")
        for _, y_pos, prev_az, curr_az, delta in az_jumps[:2]:
            print(f"    At y={y_pos:.0f}: az {prev_az:.2f} -> {curr_az:.2f} (delta={delta:.2f})")
        print(f"    Consider: Use sin/cos encoding for RL training")

    if not range_ok:
        print(f"    WARNING: target_az didn't reach ±1 (behind player)")

    env.close()
    return range_ok


def print_summary():
    """Print summary table."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    def fmt(key):
        v = RESULTS.get(key)
        if v is None:
            return 'N/A'
        if isinstance(v, float):
            return f"{v:.1f}"
        return str(v)

    print(f"| Metric         | Result | P-51D Target |")
    print(f"|----------------|--------|--------------|")
    print(f"| max_speed      | {fmt('max_speed'):>6} | {P51D_MAX_SPEED:.0f} m/s |")
    print(f"| cruise_speed   | {fmt('cruise_speed'):>6} | - |")
    print(f"| stall_speed    | {fmt('stall_speed'):>6} | {P51D_STALL_SPEED:.0f} m/s |")
    print(f"| climb_rate     | {fmt('climb_rate'):>6} | {P51D_CLIMB_RATE:.0f} m/s |")
    print(f"| glide_L/D      | {fmt('glide_LD'):>6} | 14.6 |")
    print(f"| turn_rate      | {fmt('turn_rate'):>6} | 5.6°/s (45° bank) |")
    print(f"| rudder_yaw     | {fmt('rudder_yaw_rate'):>6} | 5-15°/s (wings lvl) |")
    print(f"| pitch_dir      | {fmt('pitch_direction'):>6} | DOWN (+elev) |")
    print(f"| roll_works     | {fmt('roll_works'):>6} | YES |")


if __name__ == "__main__":
    # Map test names to functions
    TESTS = {
        'max_speed': test_max_speed,
        'acceleration': test_acceleration,
        'deceleration': test_deceleration,
        'cruise_speed': test_cruise_speed,
        'stall_speed': test_stall_speed,
        'climb_rate': test_climb_rate,
        'glide_ratio': test_glide_ratio,
        'sustained_turn': test_sustained_turn,
        'turn_60': test_turn_60,
        'pitch_direction': test_pitch_direction,
        'roll_direction': test_roll_direction,
        'rudder_only_turn': test_rudder_only_turn,
        'knife_edge_pull': test_knife_edge_pull,
        'knife_edge_flight': test_knife_edge_flight,
        'mode_weights': test_mode_weights,
        # G-force tests
        'g_level_flight': test_g_level_flight,
        'g_push_forward': test_g_push_forward,
        'g_pull_back': test_g_pull_back,
        'g_limit_negative': test_g_limit_negative,
        'g_limit_positive': test_g_limit_positive,
        # Fine control tests
        'gentle_pitch': test_gentle_pitch_control,
        # Observation scheme tests (static)
        'obs_dimensions': test_obs_scheme_dimensions,
        'obs_identity': test_obs_identity_orientation,
        'obs_pitched': test_obs_pitched_up,
        'obs_target_angles': test_obs_target_angles,
        'obs_horizon': test_obs_horizon_visible,
        'obs_edge_cases': test_obs_edge_cases,
        'obs_bounds': test_obs_bounds,
        # Dynamic maneuver observation tests
        'obs_during_loop': test_obs_during_loop,
        'obs_during_roll': test_obs_during_roll,
        'obs_vertical_pitch': test_obs_vertical_pitch,
        'obs_azimuth_crossover': test_obs_azimuth_crossover,
        # Phase 2: Additional observation edge case tests
        'obs_yaw_wrap': test_obs_yaw_wrap,
        'obs_elevation_extremes': test_obs_elevation_extremes,
        'obs_complex_maneuver': test_obs_complex_maneuver,
        'quat_normalization': test_quaternion_normalization,
        # Phase 3: OBS_PURSUIT (scheme 1) comprehensive tests
        'obs_pursuit_bounds': test_obs_pursuit_bounds,
        'obs_pursuit_energy_climb': test_obs_pursuit_energy_conservation,
        'obs_pursuit_energy_dive': test_obs_pursuit_energy_dive,
        'obs_pursuit_energy_adv': test_obs_pursuit_energy_advantage,
        'obs_pursuit_aspect': test_obs_pursuit_target_aspect,
        'obs_pursuit_closure': test_obs_pursuit_closure_rate,
        'obs_pursuit_az_wrap': test_obs_pursuit_target_angles_wrap,
    }

    print("P-51D Physics Validation Tests")
    print("=" * 60)

    if ARGS.test:
        # Run single test
        if ARGS.test in TESTS:
            print(f"Running single test: {ARGS.test}")
            if RENDER_MODE:
                print("Rendering enabled - press ESC to exit")
            print("=" * 60)
            TESTS[ARGS.test]()
        else:
            print(f"Unknown test: {ARGS.test}")
            print(f"Available tests: {', '.join(TESTS.keys())}")
    else:
        # Run all tests
        print("Using force_state() for precise initial conditions")
        if RENDER_MODE:
            print("Rendering enabled - press ESC to exit")
        print("=" * 60)
        for test_func in TESTS.values():
            test_func()
        print_summary()

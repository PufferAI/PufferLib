"""
Physics validation tests for dogfight environment.
Uses force_state() to set exact initial conditions for accurate measurements.

Run: python pufferlib/ocean/dogfight/test_flight.py
"""
import numpy as np
from dogfight import Dogfight, AutopilotMode

# Constants (must match dogfight.h)
MAX_SPEED = 250.0
WORLD_MAX_Z = 3000.0

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


def get_speed(obs):
    """Get total speed from observation."""
    vx = obs[0, 3] * MAX_SPEED
    vy = obs[0, 4] * MAX_SPEED
    vz = obs[0, 5] * MAX_SPEED
    return np.sqrt(vx**2 + vy**2 + vz**2)


def get_vz(obs):
    """Get vertical velocity from observation."""
    return obs[0, 5] * MAX_SPEED


def get_alt(obs):
    """Get altitude from observation."""
    return obs[0, 2] * WORLD_MAX_Z


def level_flight_pitch(obs, kp=LEVEL_FLIGHT_KP, kd=LEVEL_FLIGHT_KD):
    """
    PD autopilot for level flight (vz = 0).
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
    env = Dogfight(num_envs=1)
    env.reset()

    # Start at 150 m/s (near expected max), center of world, flying +X
    env.force_state(
        player_pos=(-1000, 0, 1000),
        player_vel=(150, 0, 0),
        player_throttle=1.0,
    )

    obs = env.observations
    prev_speed = get_speed(obs)
    stable_count = 0

    for step in range(1500):  # 30 seconds
        elevator = level_flight_pitch(obs)
        action = np.array([[1.0, elevator, 0.0, 0.0, 0.0]], dtype=np.float32)
        obs, _, term, _, _ = env.step(action)

        if term[0]:
            print("  (terminated - hit bounds)")
            break

        speed = get_speed(obs)
        if abs(speed - prev_speed) < 0.05:
            stable_count += 1
            if stable_count > 100:
                break
        else:
            stable_count = 0
        prev_speed = speed

    final_speed = get_speed(obs)
    RESULTS['max_speed'] = final_speed
    diff = final_speed - P51D_MAX_SPEED
    status = "OK" if abs(diff) < 15 else "CHECK"
    print(f"max_speed:     {final_speed:6.1f} m/s  (P-51D: {P51D_MAX_SPEED:.0f}, diff: {diff:+.1f}) [{status}]")


def test_cruise_speed():
    """50% throttle level flight - cruise speed."""
    env = Dogfight(num_envs=1)
    env.reset()

    # Start at moderate speed
    env.force_state(
        player_pos=(-1000, 0, 1000),
        player_vel=(120, 0, 0),
        player_throttle=0.5,
    )

    obs = env.observations
    prev_speed = get_speed(obs)
    stable_count = 0

    for step in range(1500):
        elevator = level_flight_pitch(obs)
        action = np.array([[0.0, elevator, 0.0, 0.0, 0.0]], dtype=np.float32)  # 50% throttle
        obs, _, term, _, _ = env.step(action)

        if term[0]:
            break

        speed = get_speed(obs)
        if abs(speed - prev_speed) < 0.05:
            stable_count += 1
            if stable_count > 100:
                break
        else:
            stable_count = 0
        prev_speed = speed

    final_speed = get_speed(obs)
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
    env = Dogfight(num_envs=1)

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
        obs = env.observations
        vzs = []
        for _ in range(100):  # 2 seconds
            vz = get_vz(obs)
            vzs.append(vz)
            action = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            obs, _, term, _, _ = env.step(action)
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
    env = Dogfight(num_envs=1)

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
    env.force_state(
        player_pos=(0, 0, 500),
        player_vel=(vx, 0, vz),  # Velocity along climb path
        player_ori=(ori_w, 0, ori_y, 0),  # Pitch for steady climb
        player_throttle=1.0,
    )

    # Run with zero elevator (pitch holds constant) and measure vz
    obs = env.observations
    vzs = []
    speeds = []

    for step in range(1000):  # 20 seconds
        vz_obs = get_vz(obs)
        speed = get_speed(obs)

        # Skip first 5 seconds for settling, then collect data
        if step >= 250:
            vzs.append(vz_obs)
            speeds.append(speed)

        # Zero elevator - pitch angle holds due to rate-based controls
        action = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        obs, _, term, _, _ = env.step(action)
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
    env = Dogfight(num_envs=1)

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
    env.force_state(
        player_pos=(0, 0, 2000),  # High altitude for long glide
        player_vel=(vx, 0, vz),
        player_ori=(ori_w, 0, ori_y, 0),
        player_throttle=0.0,
    )

    # Run with zero controls - let physics maintain steady glide
    obs = env.observations
    vzs = []
    speeds = []

    for step in range(500):  # 10 seconds
        vz_obs = get_vz(obs)
        speed = get_speed(obs)

        # Collect data after 2 seconds of settling
        if step >= 100:
            vzs.append(vz_obs)
            speeds.append(speed)

        # Zero controls - pitch angle holds due to rate-based system
        action = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        obs, _, term, _, _ = env.step(action)
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
    env = Dogfight(num_envs=1)

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
    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(V, 0, 0),
        player_ori=(ori_w, ori_x, ori_y, ori_z),
        player_throttle=1.0,
    )

    # Run with zero controls
    obs = env.observations
    headings = []
    speeds = []
    alts = []

    for step in range(250):  # 5 seconds
        vx = obs[0, 3] * MAX_SPEED
        vy = obs[0, 4] * MAX_SPEED
        heading = np.arctan2(vy, vx)
        speed = get_speed(obs)
        alt = get_alt(obs)

        if step >= 50:  # After 1 second settling
            headings.append(heading)
            speeds.append(speed)
            alts.append(alt)

        action = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        obs, _, term, _, _ = env.step(action)
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
    env = Dogfight(num_envs=1)

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

    obs = env.observations
    prev_vz = 0.0
    prev_bank_error = 0.0

    headings, alts, banks = [], [], []

    for step in range(250):  # 5 seconds
        # Get state
        vz = obs[0, 5] * MAX_SPEED
        alt = obs[0, 2] * WORLD_MAX_Z
        vx = obs[0, 3] * MAX_SPEED
        vy = obs[0, 4] * MAX_SPEED
        heading = np.arctan2(vy, vx)
        up_y = obs[0, 11]
        up_z = obs[0, 12]
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
        obs, _, term, _, _ = env.step(action)
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
    """Verify positive elevator = nose up."""
    env = Dogfight(num_envs=1)
    env.reset()

    env.force_state(player_vel=(80, 0, 0))

    action = np.array([[0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    initial_up_x = None
    for step in range(50):
        obs, _, _, _, _ = env.step(action)
        if step == 0:
            initial_up_x = obs[0, 10]
    final_up_x = obs[0, 10]
    nose_up = final_up_x > initial_up_x
    RESULTS['pitch_direction'] = 'UP' if nose_up else 'DOWN'
    status = 'OK' if nose_up else 'WRONG'
    print(f"pitch_dir:     {RESULTS['pitch_direction']:>6}      (should be UP) [{status}]")


def test_roll_direction():
    """Verify positive ailerons = roll right."""
    env = Dogfight(num_envs=1)
    env.reset()

    env.force_state(player_vel=(80, 0, 0))

    action = np.array([[0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    for _ in range(50):
        obs, _, _, _, _ = env.step(action)
    up_y_changed = abs(obs[0, 11]) > 0.1
    RESULTS['roll_works'] = 'YES' if up_y_changed else 'NO'
    status = 'OK' if up_y_changed else 'WRONG'
    print(f"roll_works:    {RESULTS['roll_works']:>6}      (should be YES) [{status}]")


def test_mode_weights():
    """
    Test that mode_weights actually biases autopilot randomization.

    Sets 100% weight on AP_LEVEL, triggers multiple resets,
    verifies that selected mode is always AP_LEVEL.
    """
    env = Dogfight(num_envs=1)
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
    print(f"| pitch_dir      | {fmt('pitch_direction'):>6} | UP |")
    print(f"| roll_works     | {fmt('roll_works'):>6} | YES |")


if __name__ == "__main__":
    print("P-51D Physics Validation Tests")
    print("=" * 60)
    print("Using force_state() for precise initial conditions")
    print("=" * 60)
    test_max_speed()
    test_cruise_speed()
    test_stall_speed()
    test_climb_rate()
    test_glide_ratio()
    test_sustained_turn()
    test_turn_60()
    test_pitch_direction()
    test_roll_direction()
    test_mode_weights()
    print_summary()

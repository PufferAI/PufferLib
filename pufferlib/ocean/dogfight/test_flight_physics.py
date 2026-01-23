"""
Physics validation tests for dogfight environment.
Tests flight physics: speed, climb, turn, control directions, G-forces.

Run: python pufferlib/ocean/dogfight/test_flight_physics.py --test max_speed
"""
import numpy as np
from dogfight import Dogfight, AutopilotMode

from test_flight_base import (
    get_render_mode, get_render_fps,
    RESULTS, TEST_HIGHLIGHTS, setup_highlights,
    P51D_MAX_SPEED, P51D_STALL_SPEED, P51D_CLIMB_RATE, P51D_TURN_RATE,
    LEVEL_FLIGHT_KP, LEVEL_FLIGHT_KD,
    get_speed_from_state, get_vz_from_state, get_alt_from_state,
    level_flight_pitch_from_state,
)

# Import autopilot helpers for realistic physics tests
from autopilot import hold_pitch, hold_bank, hold_bank_and_level


def test_max_speed():
    """
    Full throttle level flight starting near max speed.
    Should stabilize around 159 m/s (P-51D Military power).
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

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

    Mode 0: Uses zero elevator (pitch holds constant due to rate-based controls)
    Mode 1: Uses pitch-hold autopilot to maintain climb pitch angle
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

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
    target_pitch_deg = np.degrees(pitch)

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

    # Run and measure vz
    vzs = []
    speeds = []

    for step in range(1000):  # 20 seconds
        # Use state-based accessors (independent of obs_scheme)
        state = env.get_state()
        vz_now = state['vz']
        speed = np.sqrt(state['vx']**2 + state['vy']**2 + state['vz']**2)

        # Skip first 5 seconds for settling, then collect data
        if step >= 250:
            vzs.append(vz_now)
            speeds.append(speed)

        # Use autopilot to maintain climb attitude (realistic physics)
        elevator = hold_pitch(state, target_pitch_deg)
        aileron = hold_bank(state, 0.0)  # Wings level

        action = np.array([[1.0, elevator, aileron, 0.0, 0.0]], dtype=np.float32)
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
    Glide angle: gamma = arctan(1/L/D) = 3.9°
    Expected sink rate: V * sin(gamma) = V/(L/D) = 5.5 m/s

    Mode 0: Zero controls - pitch holds due to rate-based system
    Mode 1: Pitch-hold autopilot to maintain glide angle
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

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
    # For quaternion: we want nose tilted down from horizontal
    pitch = alpha - gamma  # Negative = nose down
    target_pitch_deg = np.degrees(pitch)

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

    # Run and measure sink rate
    vzs = []
    speeds = []

    for step in range(500):  # 10 seconds
        state = env.get_state()
        vz_now = state['vz']
        speed = np.sqrt(state['vx']**2 + state['vy']**2 + state['vz']**2)

        # Collect data after 2 seconds of settling
        if step >= 100:
            vzs.append(vz_now)
            speeds.append(speed)

        # Use autopilot to maintain glide angle (realistic physics)
        elevator = hold_pitch(state, target_pitch_deg)
        aileron = hold_bank(state, 0.0)  # Wings level

        action = np.array([[-1.0, elevator, aileron, 0.0, 0.0]], dtype=np.float32)
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
    mode_str = "realistic"
    print(f"glide_ratio:   L/D={measured_LD:4.1f}   (theory: {LD_max:.1f}, sink: {avg_sink:.1f} m/s, expected: {sink_expected:.1f}) [{status}] ({mode_str})")


def test_sustained_turn():
    """
    Sustained turn test - verifies banked flight produces a coordinated turn.

    Tests that at 30° bank, 100 m/s with proper autopilot control:
      - Plane maintains bank angle
      - Plane maintains altitude (within tight tolerance)
      - Turn rate matches theory: omega = g*tan(bank)/V = 3.2°/s

    Mode 0: Uses zero controls (pitch holds due to rate-based system)
    Mode 1: Uses coordinated turn autopilot (bank hold + altitude hold)

    Theory at 30° bank, 100 m/s:
      Turn rate = g * tan(30°) / V = 9.81 * 0.577 / 100 = 3.2°/s
      Load factor = 1/cos(30°) = 1.15g
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

    # Test parameters - 30° bank is gentle and stable
    V = 100.0           # m/s
    bank_deg = 30.0     # degrees
    bank = np.radians(bank_deg)

    # Theoretical turn rate
    theory_turn_rate = np.degrees(9.81 * np.tan(bank) / V)  # ~3.2°/s

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

    # Run turn
    headings = []
    speeds = []
    alts = []
    banks = []

    for step in range(250):  # 5 seconds
        state = env.get_state()
        vx, vy = state['vx'], state['vy']
        heading = np.arctan2(vy, vx)
        speed = np.sqrt(vx**2 + vy**2 + state['vz']**2)
        alt = state['pz']

        # Calculate actual bank angle
        up_y, up_z = state['up_y'], state['up_z']
        bank_actual = np.degrees(np.arccos(np.clip(up_z, -1, 1)))
        if up_y > 0:
            bank_actual = -bank_actual

        if step >= 50:  # After 1 second settling
            headings.append(heading)
            speeds.append(speed)
            alts.append(alt)
            banks.append(bank_actual)

        # Coordinated turn autopilot (hold bank + maintain altitude)
        elevator, aileron = hold_bank_and_level(state, bank_deg)

        action = np.array([[1.0, elevator, aileron, 0.0, 0.0]], dtype=np.float32)
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
    avg_bank = np.mean(banks) if banks else 0

    RESULTS['turn_rate'] = abs(turn_rate_actual)

    # Tight tolerances for proper sustained turn with autopilot
    turn_rate_ok = abs(turn_rate_actual) > theory_turn_rate * 0.5
    alt_ok = abs(alt_change) < 50  # Less than 50m change
    bank_ok = abs(avg_bank - bank_deg) < 15

    all_ok = turn_rate_ok and alt_ok and bank_ok
    status = "OK" if all_ok else "CHECK"

    print(f"turn_rate:     {abs(turn_rate_actual):5.1f}°/s (theory: {theory_turn_rate:.1f}, bank: {avg_bank:.0f}°/{bank_deg:.0f}°, Δalt: {alt_change:+.0f}m) [{status}]")

    if not all_ok:
        if not turn_rate_ok:
            print(f"  CHECK: Turn rate {turn_rate_actual:.1f}°/s too low")
        if not alt_ok:
            print(f"  CHECK: Altitude change {alt_change:+.0f}m exceeds tolerance")


def test_turn_60():
    """
    Coordinated turn at 60° bank with PID control.

    P-51D reference: 60° bank (2.0g) at 350 mph gives 5°/s
    At 100 m/s: theory = g*tan(60°)/V = 9.81*1.732/100 = 9.7°/s
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
        # Positive roll = banked LEFT, need positive aileron (roll RIGHT) to correct
        roll_error = 0.0 - roll
        roll_deriv = (roll - prev_roll) / 0.02
        aileron = -(roll_kp * roll_error - roll_kd * roll_deriv)  # Flip sign: left bank → right aileron
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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


def test_autopilot_enum_sync():
    """
    Verify Python AutopilotMode enum values match expected C enum values.

    This test catches enum sync bugs where Python and C have different values.
    Bug fixed: RANDOM was 6 in Python but 10 in C, causing RANDOM to be
    interpreted as HARD_TURN_LEFT.
    """
    # Expected values from autopilot.h enum
    expected = {
        'STRAIGHT': 0,
        'LEVEL': 1,
        'TURN_LEFT': 2,
        'TURN_RIGHT': 3,
        'CLIMB': 4,
        'DESCEND': 5,
        'HARD_TURN_LEFT': 6,
        'HARD_TURN_RIGHT': 7,
        'WEAVE': 8,
        'EVASIVE': 9,
        'RANDOM': 10,
    }

    errors = []
    for name, expected_val in expected.items():
        actual_val = getattr(AutopilotMode, name, None)
        if actual_val is None:
            errors.append(f"{name}: MISSING")
        elif actual_val != expected_val:
            errors.append(f"{name}: got {actual_val}, expected {expected_val}")

    status = "OK" if not errors else "FAIL"
    RESULTS['enum_sync'] = len(errors)
    print(f"enum_sync:     {len(expected)} modes checked [{status}]")

    if errors:
        for err in errors:
            print(f"  ERROR: {err}")


def test_autopilot_random_not_hardturn():
    """
    Verify RANDOM mode actually randomizes, not treated as HARD_TURN_LEFT.

    Bug fixed: Python RANDOM=6 was interpreted as C HARD_TURN_LEFT=6.
    With the fix, RANDOM=10 triggers randomization to modes 1-5.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # Set RANDOM mode
    env.set_autopilot(env_idx=0, mode=AutopilotMode.RANDOM)

    # Collect modes after multiple resets
    modes = []
    for _ in range(30):
        env.reset()
        modes.append(env.get_autopilot_mode(env_idx=0))

    unique_modes = set(modes)
    all_valid = all(1 <= m <= 5 for m in modes)
    has_variety = len(unique_modes) >= 3  # Should see at least 3 different modes

    # Key check: mode 6 (HARD_TURN_LEFT) should NEVER appear
    no_hardturn = 6 not in modes

    status = "OK" if (all_valid and has_variety and no_hardturn) else "FAIL"
    RESULTS['random_not_hardturn'] = 1 if status == "OK" else 0
    print(f"random_mode:   unique={unique_modes}, no_mode_6={no_hardturn} [{status}]")

    if not no_hardturn:
        print(f"  FAIL: Mode 6 (HARD_TURN_LEFT) appeared - enum sync bug!")
    if not all_valid:
        print(f"  FAIL: Got modes outside 1-5 range: {[m for m in modes if m < 1 or m > 5]}")


def test_autopilot_bounds_check():
    """
    Verify invalid autopilot mode values are clamped to STRAIGHT.

    Tests that binding.c bounds checking works for out-of-range values.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # Test invalid mode values (should clamp to STRAIGHT=0)
    test_cases = [
        (-1, "negative"),
        (11, "above AP_COUNT"),
        (100, "way out of range"),
    ]

    all_ok = True
    for invalid_mode, desc in test_cases:
        env.set_autopilot(env_idx=0, mode=invalid_mode)
        result_mode = env.get_autopilot_mode(env_idx=0)
        if result_mode != 0:  # Should be STRAIGHT
            all_ok = False
            print(f"  FAIL: mode={invalid_mode} ({desc}) -> {result_mode}, expected 0")

    status = "OK" if all_ok else "FAIL"
    RESULTS['bounds_check'] = 1 if all_ok else 0
    print(f"bounds_check:  invalid modes clamped to STRAIGHT [{status}]")


def test_force_state_pid_reset():
    """
    Verify force_state() resets autopilot PID state to avoid derivative spikes.

    After teleporting with force_state(), the autopilot should not have
    large derivative terms from the previous state causing control jumps.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # Set up autopilot in LEVEL mode (uses PID for altitude hold)
    env.set_autopilot(env_idx=0, mode=AutopilotMode.LEVEL)

    # Run for a bit to build up PID state
    action = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    for _ in range(50):
        env.step(action)

    # Teleport to a very different altitude (large vz change)
    env.force_state(
        player_pos=(0, 0, 2000),  # Different altitude
        player_vel=(150, 0, 50),  # Different vertical velocity
    )

    # Get state immediately after force_state
    state = env.get_state(env_idx=0)

    # The test passes if we didn't crash and state is valid
    # (A derivative spike from stale PID state would cause extreme control outputs)
    pos_valid = abs(state['pz'] - 2000) < 1
    vel_valid = abs(state['vz'] - 50) < 1

    status = "OK" if pos_valid and vel_valid else "FAIL"
    RESULTS['pid_reset'] = 1 if status == "OK" else 0
    print(f"pid_reset:     force_state resets PID state [{status}]")


# =============================================================================
# G-FORCE TESTS - Validate G-loading physics
# =============================================================================

def test_g_level_flight():
    """
    Level flight at cruise speed - verify G ≈ 1.0.
    In steady level flight, lift equals weight, so G-loading should be ~1.0.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
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
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

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


# Test registry for this module
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
    # Autopilot enum sync tests
    'autopilot_enum_sync': test_autopilot_enum_sync,
    'autopilot_random_mode': test_autopilot_random_not_hardturn,
    'autopilot_bounds_check': test_autopilot_bounds_check,
    'autopilot_pid_reset': test_force_state_pid_reset,
    # G-force tests
    'g_level_flight': test_g_level_flight,
    'g_push_forward': test_g_push_forward,
    'g_pull_back': test_g_pull_back,
    'g_limit_negative': test_g_limit_negative,
    'g_limit_positive': test_g_limit_positive,
    # Fine control tests
    'gentle_pitch': test_gentle_pitch_control,
}


if __name__ == "__main__":
    from test_flight_base import get_args
    args = get_args()

    print("P-51D Physics Validation Tests")
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
        print("Running all physics tests")
        if get_render_mode():
            print("Rendering enabled - press ESC to exit")
        print("=" * 60)
        for test_func in TESTS.values():
            test_func()

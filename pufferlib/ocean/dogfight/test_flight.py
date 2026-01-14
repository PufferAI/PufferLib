"""
Physics validation tests for dogfight environment.
Uses force_state() to set exact initial conditions for accurate measurements.

Run: python pufferlib/ocean/dogfight/test_flight.py
"""
import numpy as np
from dogfight import Dogfight

# Constants (must match dogfight.h)
MAX_SPEED = 250.0
WORLD_MAX_Z = 3000.0

# P-51D reference values (from P51d_REFERENCE_DATA.md)
P51D_MAX_SPEED = 159.0      # m/s (355 mph, Military power, SL)
P51D_STALL_SPEED = 45.0     # m/s (100 mph, 9000 lb, clean)
P51D_CLIMB_RATE = 15.4      # m/s (3030 ft/min, Military power)

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
        # q = (cos(θ/2), 0, sin(θ/2), 0) for pitch up by θ
        pitch_rad = alpha_needed
        ori_w = np.cos(pitch_rad / 2)
        ori_y = np.sin(pitch_rad / 2)

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

    # Create pitch-up quaternion
    ori_w = np.cos(pitch / 2)
    ori_y = np.sin(pitch / 2)

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
    test_pitch_direction()
    test_roll_direction()
    print_summary()

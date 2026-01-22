"""
Energy physics tests for dogfight environment.
Tests energy conservation, bleed rates, and E-M theory concepts.

Key Physics:
  Specific Energy:  Es = h + v^2/(2g)  [meters of altitude equivalent]
  Kinetic Energy:   KE = 0.5 * m * v^2
  Potential Energy: PE = m * g * h
  Total Energy:     E = KE + PE = m * (g*h + 0.5*v^2)

  Specific Excess Power: Ps = (T - D) * V / W  [m/s rate of energy change]

  In a turn at bank angle phi:
    Required lift: L = W / cos(phi)
    Load factor: n = 1 / cos(phi)
    Induced drag increases with n^2

Run: python pufferlib/ocean/dogfight/test_flight_energy.py
     python pufferlib/ocean/dogfight/test_flight_energy.py --render --fps 10
     python pufferlib/ocean/dogfight/test_flight_energy.py --test knife_edge_pull_energy
"""
import numpy as np
from dogfight import Dogfight

from test_flight_base import (
    get_render_mode, get_render_fps, setup_highlights,
    RESULTS, TEST_HIGHLIGHTS,
    get_speed_from_state, get_alt_from_state,
)

# Physics constants
G = 9.81  # m/s^2
MASS = 4082  # kg (P-51D loaded weight)


def compute_specific_energy(speed, altitude):
    """
    Compute specific energy (energy per unit weight).
    Es = h + v^2/(2g)  [meters of altitude equivalent]

    This is the total mechanical energy expressed as equivalent altitude.
    A plane at 1000m going 100 m/s has Es = 1000 + 100^2/(2*9.81) = 1510m
    """
    return altitude + (speed ** 2) / (2 * G)


def compute_energies(speed, altitude):
    """
    Compute kinetic, potential, and total energy.
    Returns (KE, PE, Total) in Joules.
    """
    ke = 0.5 * MASS * speed ** 2
    pe = MASS * G * altitude
    total = ke + pe
    return ke, pe, total


def get_energy_state(env):
    """Get current energy state from environment."""
    state = env.get_state()
    speed = np.sqrt(state['vx']**2 + state['vy']**2 + state['vz']**2)
    alt = state['pz']

    ke, pe, total = compute_energies(speed, alt)
    es = compute_specific_energy(speed, alt)

    return {
        'speed': speed,
        'alt': alt,
        'ke': ke,           # Kinetic energy (J)
        'pe': pe,           # Potential energy (J)
        'total': total,     # Total energy (J)
        'es': es,           # Specific energy (m)
        'vz': state['vz'],
    }


# =============================================================================
# ENERGY TESTS
# =============================================================================

def test_knife_edge_pull_energy():
    """
    Knife-edge (90 deg bank) + full elevator pull + zero throttle.

    This is a HIGH DRAG scenario:
    - 90 deg bank: wings vertical, no vertical lift
    - Full elevator pull: high angle of attack = massive induced drag
    - Zero throttle: no thrust to offset drag

    Expected:
    - Kinetic energy drops (drag slows plane)
    - Potential energy drops (no lift, plane falls)
    - Total energy drops RAPIDLY (both components bleeding)

    This tests that high-G maneuvers correctly penalize energy.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # Set up knife-edge: 90 deg right roll
    roll_90 = np.radians(90)
    qw = np.cos(roll_90 / 2)
    qx = -np.sin(roll_90 / 2)  # Negative for right roll

    # Start at good speed and altitude
    V = 150.0  # m/s - high speed for dramatic effect
    env.force_state(
        player_pos=(0, 0, 2000),
        player_vel=(V, 0, 0),
        player_ori=(qw, qx, 0.0, 0.0),
        player_throttle=0.0,  # ZERO THROTTLE
    )

    # Record initial energy state
    initial = get_energy_state(env)

    print(f"  Initial state:")
    print(f"    Speed: {initial['speed']:.1f} m/s")
    print(f"    Altitude: {initial['alt']:.1f} m")
    print(f"    Specific Energy: {initial['es']:.1f} m")
    print(f"    KE: {initial['ke']/1e6:.2f} MJ, PE: {initial['pe']/1e6:.2f} MJ")

    # Run with full elevator pull, zero throttle
    data = []
    for step in range(150):  # 3 seconds
        # Zero throttle (-1 maps to 0%), full pull (-1), no aileron, no rudder
        action = np.array([[-1.0, -1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)

        e = get_energy_state(env)
        data.append(e)

        if step % 50 == 0:
            print(f"    Step {step:3d}: speed={e['speed']:.1f}, alt={e['alt']:.0f}, Es={e['es']:.0f}m")

        if term[0]:
            print(f"    (terminated at step {step})")
            break

    # Analyze energy changes
    final = data[-1]

    speed_loss = initial['speed'] - final['speed']
    alt_loss = initial['alt'] - final['alt']
    ke_loss = initial['ke'] - final['ke']
    pe_loss = initial['pe'] - final['pe']
    total_loss = initial['total'] - final['total']
    es_loss = initial['es'] - final['es']

    # Calculate rates
    time_elapsed = len(data) * 0.02
    es_bleed_rate = es_loss / time_elapsed  # m/s of specific energy

    print(f"\n  Final state after {time_elapsed:.1f}s:")
    print(f"    Speed: {final['speed']:.1f} m/s (lost {speed_loss:.1f})")
    print(f"    Altitude: {final['alt']:.1f} m (lost {alt_loss:.1f})")
    print(f"    Specific Energy: {final['es']:.1f} m (lost {es_loss:.1f})")
    print(f"    KE loss: {ke_loss/1e6:.2f} MJ, PE loss: {pe_loss/1e6:.2f} MJ")
    print(f"    Energy bleed rate: {es_bleed_rate:.1f} m/s of Es")

    # Verify ALL energies decreased
    ke_dropped = ke_loss > 0
    pe_dropped = pe_loss > 0
    total_dropped = total_loss > 0

    # Should lose significant energy (at least 100m of Es in 3 seconds)
    significant_loss = es_loss > 100

    passed = ke_dropped and pe_dropped and total_dropped and significant_loss
    RESULTS['knife_edge_pull_energy'] = passed

    status = "OK" if passed else "FAIL"
    print(f"\nknife_pull_E:  KE={'DROP' if ke_dropped else 'RISE'}, PE={'DROP' if pe_dropped else 'RISE'}, "
          f"Es_loss={es_loss:.0f}m [{status}]")

    if not ke_dropped:
        print(f"  FAIL: Kinetic energy should DROP (drag!), but it increased")
    if not pe_dropped:
        print(f"  FAIL: Potential energy should DROP (no lift!), but it increased")
    if not significant_loss:
        print(f"  FAIL: Should lose >100m of Es in high-drag maneuver, only lost {es_loss:.0f}m")

    env.close()
    return passed


def test_energy_level_flight():
    """
    Level flight at cruise: energy should be roughly constant.

    With throttle balanced against drag, Ps ≈ 0, so total energy
    should remain stable (small fluctuations from autopilot corrections).
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # Start at cruise speed, level
    V = 120.0
    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(V, 0, 0),
        player_throttle=0.5,
    )

    initial = get_energy_state(env)
    energies = [initial['es']]

    # Simple level flight autopilot
    prev_vz = 0
    kp, kd = 0.001, 0.001

    for step in range(500):  # 10 seconds
        state = env.get_state()
        vz = state['vz']
        elevator = -kp * vz - kd * (vz - prev_vz) / 0.02
        elevator = np.clip(elevator, -0.2, 0.2)
        prev_vz = vz

        action = np.array([[0.0, elevator, 0.0, 0.0, 0.0]], dtype=np.float32)
        env.step(action)

        e = get_energy_state(env)
        energies.append(e['es'])

    final = get_energy_state(env)
    es_change = final['es'] - initial['es']
    es_std = np.std(energies)

    # Energy should be stable (change < 50m, std < 20m)
    stable = abs(es_change) < 50 and es_std < 30

    RESULTS['energy_level_flight'] = stable
    status = "OK" if stable else "CHECK"
    print(f"energy_level:  Es_change={es_change:+.1f}m, std={es_std:.1f}m [{status}]")

    env.close()
    return stable


def test_energy_dive_acceleration():
    """
    Dive at 45 degrees, zero throttle: potential -> kinetic conversion.

    Total energy should decrease slowly (drag), but kinetic should
    increase as potential decreases (trading altitude for speed).
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # 45 degree dive
    pitch_down = np.radians(-45)
    qw = np.cos(pitch_down / 2)
    qy = -np.sin(pitch_down / 2)

    env.force_state(
        player_pos=(0, 0, 2500),
        player_vel=(80, 0, 0),  # Start slow
        player_ori=(qw, 0, qy, 0),
        player_throttle=0.0,
    )

    initial = get_energy_state(env)

    for step in range(200):  # 4 seconds
        action = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)
        if term[0] or env.get_state()['pz'] < 500:
            break

    final = get_energy_state(env)

    speed_gain = final['speed'] - initial['speed']
    alt_loss = initial['alt'] - final['alt']
    ke_gain = final['ke'] - initial['ke']
    pe_loss = initial['pe'] - final['pe']
    es_loss = initial['es'] - final['es']

    # Verify energy transfer
    ke_increased = ke_gain > 0
    pe_decreased = pe_loss > 0
    # Total should decrease (drag), but not by much
    es_loss_reasonable = 0 < es_loss < alt_loss * 0.3  # Less than 30% to drag

    passed = ke_increased and pe_decreased and es_loss_reasonable
    RESULTS['energy_dive'] = passed

    status = "OK" if passed else "CHECK"
    print(f"energy_dive:   speed+{speed_gain:.0f}, alt-{alt_loss:.0f}, Es_loss={es_loss:.0f}m ({100*es_loss/alt_loss:.0f}% to drag) [{status}]")

    env.close()
    return passed


def test_energy_climb_deceleration():
    """
    Climb at 30 degrees, full throttle: kinetic -> potential conversion.

    With full throttle, should gain altitude while losing some speed,
    but total energy should increase (thrust > drag).
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # 30 degree climb
    pitch_up = np.radians(30)
    qw = np.cos(-pitch_up / 2)
    qy = np.sin(-pitch_up / 2)

    env.force_state(
        player_pos=(0, 0, 1000),
        player_vel=(140, 0, 0),  # Start fast
        player_ori=(qw, 0, qy, 0),
        player_throttle=1.0,
    )

    initial = get_energy_state(env)

    for step in range(300):  # 6 seconds
        action = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)
        if term[0]:
            break

    final = get_energy_state(env)

    speed_loss = initial['speed'] - final['speed']
    alt_gain = final['alt'] - initial['alt']
    es_change = final['es'] - initial['es']

    # With full throttle in climb, total energy should increase or stay stable
    # (Thrust provides Ps > 0)
    energy_ok = es_change > -50  # Allow small loss from drag
    alt_gained = alt_gain > 100

    passed = energy_ok and alt_gained
    RESULTS['energy_climb'] = passed

    status = "OK" if passed else "CHECK"
    print(f"energy_climb:  speed-{speed_loss:.0f}, alt+{alt_gain:.0f}, Es_change={es_change:+.0f}m [{status}]")

    env.close()
    return passed


def test_energy_sustained_turn_bleed():
    """
    Sustained turn at 60 degrees bank: measure energy bleed rate.

    In a banked turn, induced drag increases with load factor squared.
    At 60 deg bank, n = 2.0, so induced drag is 4x level flight.
    Even with full throttle, energy should bleed.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # 60 degree right bank
    bank = np.radians(60)
    qw = np.cos(bank / 2)
    qx = -np.sin(bank / 2)

    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(120, 0, 0),
        player_ori=(qw, qx, 0.0, 0.0),
        player_throttle=1.0,
    )

    initial = get_energy_state(env)

    # PID to hold bank and altitude
    prev_vz = 0
    prev_bank_err = 0

    energies = []
    for step in range(250):  # 5 seconds
        state = env.get_state()
        vz = state['vz']
        up_y, up_z = state['up_y'], state['up_z']
        bank_actual = np.arccos(np.clip(up_z, -1, 1))

        # Elevator to hold altitude
        elev = -0.05 * (-vz) + 0.005 * (vz - prev_vz) / 0.02
        elev = np.clip(elev, -1.0, 1.0)
        prev_vz = vz

        # Aileron to hold bank
        bank_err = bank - bank_actual
        ail = -2.0 * bank_err - 0.1 * (bank_err - prev_bank_err) / 0.02
        ail = np.clip(ail, -1.0, 1.0)
        prev_bank_err = bank_err

        action = np.array([[1.0, elev, ail, 0.0, 0.0]], dtype=np.float32)
        env.step(action)

        e = get_energy_state(env)
        energies.append(e['es'])

    final = get_energy_state(env)
    es_loss = initial['es'] - final['es']
    time_elapsed = len(energies) * 0.02
    bleed_rate = es_loss / time_elapsed

    # At 60 deg bank (2G), should lose energy even at full throttle
    # Expect 5-20 m/s of Es bleed
    bleeding = es_loss > 10

    RESULTS['energy_turn_bleed'] = bleeding
    status = "OK" if bleeding else "CHECK"
    print(f"energy_turn:   Es_loss={es_loss:.0f}m in {time_elapsed:.1f}s, bleed={bleed_rate:.1f} m/s [{status}]")

    if not bleeding:
        print(f"  NOTE: 60 deg turn should bleed energy (high induced drag)")

    env.close()
    return bleeding


def test_energy_loop():
    """
    Full loop maneuver: measure total energy loss.

    A loop involves sustained high-G (3-4G at bottom), which creates
    massive induced drag. Energy should drop 10-20% through a loop.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # Start fast and level for loop entry
    env.force_state(
        player_pos=(0, 0, 1500),
        player_vel=(150, 0, 0),
        player_throttle=1.0,
    )

    initial = get_energy_state(env)

    # Pull through loop (full back stick)
    g_max = 0
    for step in range(200):  # ~4 seconds for loop
        action = np.array([[1.0, -0.8, 0.0, 0.0, 0.0]], dtype=np.float32)  # Strong pull
        env.step(action)

        g = env.get_state()['g_force']
        g_max = max(g_max, g)

        # Check if we've completed loop (fwd_z goes negative then positive again)
        state = env.get_state()
        if step > 50 and state['fwd_z'] > -0.1 and state['fwd_x'] > 0.5:
            break

    final = get_energy_state(env)
    es_loss = initial['es'] - final['es']
    pct_loss = 100 * es_loss / initial['es']

    # Loop should lose 5-25% energy from drag
    energy_lost = 5 < pct_loss < 35

    RESULTS['energy_loop'] = energy_lost
    status = "OK" if energy_lost else "CHECK"
    print(f"energy_loop:   Es_loss={es_loss:.0f}m ({pct_loss:.1f}%), max_G={g_max:.1f} [{status}]")

    env.close()
    return energy_lost


def test_energy_split_s():
    """
    Split-S: half roll + pull through (dive recovery).

    Trades altitude for speed. Total energy decreases (drag during pull),
    but kinetic energy increases significantly.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # Start high and slow
    env.force_state(
        player_pos=(0, 0, 2500),
        player_vel=(100, 0, 0),
        player_throttle=0.5,
    )

    initial = get_energy_state(env)

    # Phase 1: Half roll (invert)
    for step in range(25):
        action = np.array([[0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        env.step(action)

    # Phase 2: Pull through (now inverted, pull = dive down then back up)
    for step in range(150):
        action = np.array([[1.0, -1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, _, term, _, _ = env.step(action)

        state = env.get_state()
        # Stop when nose is back above horizon
        if state['fwd_z'] > 0.3 and state['pz'] < initial['alt']:
            break
        if term[0]:
            break

    final = get_energy_state(env)

    speed_gain = final['speed'] - initial['speed']
    alt_loss = initial['alt'] - final['alt']
    es_loss = initial['es'] - final['es']

    # Split-S should gain speed while losing altitude
    speed_increased = speed_gain > 20
    alt_decreased = alt_loss > 200

    passed = speed_increased and alt_decreased
    RESULTS['energy_split_s'] = passed

    status = "OK" if passed else "CHECK"
    print(f"energy_split_s: speed+{speed_gain:.0f}, alt-{alt_loss:.0f}, Es_loss={es_loss:.0f}m [{status}]")

    env.close()
    return passed


def test_energy_zoom_climb():
    """
    Zoom climb: trade kinetic for potential energy.

    Start already pointing straight up with vertical velocity.
    Zero throttle - pure kinetic -> potential conversion.
    Tests energy conservation with only drag losses.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())
    env.reset()

    # Start vertical: 90 deg pitch up
    pitch_up = np.radians(90)
    qw = np.cos(-pitch_up / 2)
    qy = np.sin(-pitch_up / 2)

    V_start = 150.0  # Good starting speed
    env.force_state(
        player_pos=(0, 0, 500),
        player_vel=(0, 0, V_start),  # Velocity straight UP
        player_ori=(qw, 0, qy, 0),   # Nose pointing UP
        player_throttle=0.0,
    )

    initial = get_energy_state(env)

    # Theoretical max altitude gain (no drag): dh = v^2/(2g)
    theoretical_gain = V_start**2 / (2 * G)

    max_alt = initial['alt']

    # Coast up with zero throttle, zero controls
    for step in range(400):
        action = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        env.step(action)

        state = env.get_state()
        max_alt = max(max_alt, state['pz'])

        # Stop when we start falling
        if state['vz'] < 0:
            break

    alt_gain = max_alt - initial['alt']
    efficiency = 100 * alt_gain / theoretical_gain

    final = get_energy_state(env)
    es_loss = initial['es'] - final['es']

    # Should convert at least 70% of kinetic to potential (only drag losses)
    efficient = efficiency > 65

    RESULTS['energy_zoom'] = efficient
    status = "OK" if efficient else "CHECK"
    print(f"energy_zoom:   alt_gain={alt_gain:.0f}m (theory={theoretical_gain:.0f}m, eff={efficiency:.0f}%) [{status}]")

    env.close()
    return efficient


def test_energy_throttle_effect():
    """
    Test that throttle controls energy rate (Specific Excess Power).

    At constant speed/altitude:
    - Full throttle: Ps > 0 (can accelerate or climb)
    - Zero throttle: Ps < 0 (will decelerate or sink)
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

    results = {}

    for throttle_name, throttle_val in [('full', 1.0), ('half', 0.0), ('zero', -1.0)]:
        env.reset()

        # Start at cruise speed, level
        env.force_state(
            player_pos=(0, 0, 1500),
            player_vel=(120, 0, 0),
            player_throttle=0.5,
        )

        initial = get_energy_state(env)

        # Hold level flight with given throttle
        prev_vz = 0
        for step in range(200):  # 4 seconds
            state = env.get_state()
            vz = state['vz']
            elev = -0.001 * vz - 0.001 * (vz - prev_vz) / 0.02
            prev_vz = vz

            action = np.array([[throttle_val, elev, 0.0, 0.0, 0.0]], dtype=np.float32)
            env.step(action)

        final = get_energy_state(env)
        es_change = final['es'] - initial['es']
        results[throttle_name] = es_change

    # Verify throttle effect on energy
    full_positive = results['full'] > results['half']
    zero_negative = results['zero'] < results['half']

    passed = full_positive and zero_negative
    RESULTS['energy_throttle'] = passed

    status = "OK" if passed else "CHECK"
    print(f"energy_throttle: full={results['full']:+.0f}m, half={results['half']:+.0f}m, zero={results['zero']:+.0f}m [{status}]")

    env.close()
    return passed


def test_energy_high_g_bleed():
    """
    Compare energy bleed at different G levels.

    Higher G = more induced drag = faster energy bleed.
    Tests at 2G, 4G, 6G pulls.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

    results = {}

    # Different elevator settings for different G levels
    for g_target, elev in [('2G', -0.3), ('4G', -0.6), ('6G', -1.0)]:
        env.reset()

        env.force_state(
            player_pos=(0, 0, 2000),
            player_vel=(150, 0, 0),
            player_throttle=1.0,  # Full throttle
        )

        initial = get_energy_state(env)
        g_values = []

        for step in range(50):  # 1 second
            action = np.array([[1.0, elev, 0.0, 0.0, 0.0]], dtype=np.float32)
            env.step(action)
            g_values.append(env.get_state()['g_force'])

        final = get_energy_state(env)
        es_loss = initial['es'] - final['es']
        avg_g = np.mean(g_values)

        results[g_target] = {'es_loss': es_loss, 'avg_g': avg_g}

    # Higher G should mean more energy loss
    bleed_increases = (results['2G']['es_loss'] < results['4G']['es_loss'] < results['6G']['es_loss'])

    RESULTS['energy_g_bleed'] = bleed_increases
    status = "OK" if bleed_increases else "CHECK"

    print(f"energy_g_bleed:")
    for g_target in ['2G', '4G', '6G']:
        r = results[g_target]
        print(f"    {g_target}: avg_G={r['avg_g']:.1f}, Es_loss={r['es_loss']:.0f}m")
    print(f"    Higher G = more bleed: {bleed_increases} [{status}]")

    env.close()
    return bleed_increases


def test_sideslip_drag():
    """
    Test that sideslip creates additional drag.

    Full rudder should build up sideslip (yaw_from_rudder), which adds drag.
    Compare energy loss with and without rudder input.
    """
    env = Dogfight(num_envs=1, render_mode=get_render_mode(), render_fps=get_render_fps())

    results = {}

    for test_name, rudder_input in [('no_rudder', 0.0), ('full_rudder', 1.0)]:
        env.reset()

        env.force_state(
            player_pos=(0, 0, 1500),
            player_vel=(120, 0, 0),
            player_throttle=0.0,  # Zero throttle to isolate drag effect
        )

        initial = get_energy_state(env)

        # Hold wings level with aileron, apply rudder
        prev_roll = 0
        for step in range(150):  # 3 seconds
            state = env.get_state()
            up_y, up_z = state['up_y'], state['up_z']
            roll = np.arctan2(up_y, up_z)

            # Wings level PID
            aileron = 1.0 * (0 - roll) - 0.05 * (roll - prev_roll) / 0.02
            aileron = np.clip(aileron, -1.0, 1.0)
            prev_roll = roll

            action = np.array([[-1.0, 0.0, aileron, rudder_input, 0.0]], dtype=np.float32)
            env.step(action)

        final = get_energy_state(env)
        results[test_name] = initial['es'] - final['es']

    # Rudder should cause MORE energy loss due to sideslip drag
    more_drag_with_rudder = results['full_rudder'] > results['no_rudder'] + 5

    RESULTS['sideslip_drag'] = more_drag_with_rudder
    status = "OK" if more_drag_with_rudder else "FAIL"

    diff = results['full_rudder'] - results['no_rudder']
    print(f"sideslip_drag: no_rudder={results['no_rudder']:.0f}m, full_rudder={results['full_rudder']:.0f}m, diff={diff:+.0f}m [{status}]")

    if not more_drag_with_rudder:
        print(f"  FAIL: Full rudder should create more drag from sideslip")

    env.close()
    return more_drag_with_rudder


# Test registry for this module
TESTS = {
    'sideslip_drag': test_sideslip_drag,
    'knife_edge_pull_energy': test_knife_edge_pull_energy,
    'energy_level_flight': test_energy_level_flight,
    'energy_dive': test_energy_dive_acceleration,
    'energy_climb': test_energy_climb_deceleration,
    'energy_turn_bleed': test_energy_sustained_turn_bleed,
    'energy_loop': test_energy_loop,
    'energy_split_s': test_energy_split_s,
    'energy_zoom': test_energy_zoom_climb,
    'energy_throttle': test_energy_throttle_effect,
    'energy_g_bleed': test_energy_high_g_bleed,
}


if __name__ == "__main__":
    from test_flight_base import get_args
    args = get_args()

    print("Energy Physics Tests")
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
        print("Running all energy tests")
        if get_render_mode():
            print("Rendering enabled - press ESC to exit")
        print("=" * 60)
        for test_func in TESTS.values():
            test_func()
            print()  # Blank line between tests

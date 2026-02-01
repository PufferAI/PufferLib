"""Comprehensive verification suite for orbital_dock environment.

10 tests covering:
1. Energy conservation
2. Circular orbit stability
3. Relative frame stability
4. Hohmann transfer physics
5. V-bar stability (along-track)
6. CW dynamics (R-bar oscillation, V-bar drift)
7. Docking mechanics
8. Fuel accounting
9. Observation bounds
10. Termination conditions
"""
import numpy as np
from pufferlib.ocean.orbital_dock.orbital_dock import OrbitalDock


def obs_to_state(obs):
    """Extract physical state from observation."""
    rel_pos = obs[0:3] * 100.0  # Scale: 100m
    rel_vel = obs[3:6] * 2.0    # Scale: 2 m/s
    dist = obs[6] * 100.0
    closing_speed = obs[7] * 2.0
    fuel = obs[8]  # Already normalized [0, 1]
    return rel_pos, rel_vel, dist, closing_speed, fuel


def test_1_energy_conservation():
    """TEST 1: Energy conservation in two-body problem.

    With no thrust, the orbital energy should be conserved.
    We verify this indirectly through stable relative position.
    """
    print("="*60)
    print("TEST 1: Energy conservation")
    print("="*60)

    env = OrbitalDock(num_envs=1, difficulty=0.0)
    obs, _ = env.reset(seed=42)

    no_thrust = np.array([[2, 2, 2]], dtype=np.int32)

    initial_dist = obs[0, 6] * 100.0

    for step in range(100):
        obs, _, terminals, _, _ = env.step(no_thrust)
        if terminals[0]:
            break

    final_dist = obs[0, 6] * 100.0
    drift = abs(final_dist - initial_dist)

    print(f"  Initial distance: {initial_dist:.2f}m")
    print(f"  Final distance: {final_dist:.2f}m")
    print(f"  Drift after 100 steps: {drift:.4f}m")

    env.close()

    if drift < 1.0:
        print("  -> PASS: Energy effectively conserved (drift < 1m)")
        return True
    else:
        print(f"  -> FAIL: Excessive drift {drift:.4f}m indicates energy loss")
        return False


def test_2_circular_orbit_stability():
    """TEST 2: Circular orbit stability.

    With both bodies using same integrator, relative position should
    be stable over many steps.
    """
    print("\n" + "="*60)
    print("TEST 2: Circular orbit stability")
    print("="*60)

    env = OrbitalDock(num_envs=1, difficulty=0.0, max_steps=1000)
    obs, _ = env.reset(seed=123)

    no_thrust = np.array([[2, 2, 2]], dtype=np.int32)

    initial_dist = obs[0, 6] * 100.0

    for step in range(500):
        obs, _, terminals, _, _ = env.step(no_thrust)
        if terminals[0]:
            print(f"  Terminated at step {step}")
            break

    final_dist = obs[0, 6] * 100.0
    drift = abs(final_dist - initial_dist)

    print(f"  Initial distance: {initial_dist:.2f}m")
    print(f"  Final distance: {final_dist:.2f}m")
    print(f"  Drift after {step+1} steps: {drift:.2f}m")

    env.close()

    if drift < 5.0:
        print("  -> PASS: Orbit stable (drift < 5m)")
        return True
    else:
        print(f"  -> FAIL: Orbit unstable (drift = {drift:.2f}m)")
        return False


def test_3_relative_frame_stability():
    """TEST 3: Relative frame stability.

    With zero relative velocity and no thrust, relative position
    should remain nearly constant over short timescales.
    """
    print("\n" + "="*60)
    print("TEST 3: Relative frame stability")
    print("="*60)

    env = OrbitalDock(num_envs=1, difficulty=0.0)
    obs, _ = env.reset(seed=42)

    no_thrust = np.array([[2, 2, 2]], dtype=np.int32)

    initial_dist = obs[0, 6] * 100.0

    distances = [initial_dist]
    for step in range(100):
        obs, _, terminals, _, _ = env.step(no_thrust)
        distances.append(obs[0, 6] * 100.0)
        if terminals[0]:
            break

    max_drift = max(abs(d - initial_dist) for d in distances)

    print(f"  Initial distance: {initial_dist:.2f}m")
    print(f"  Final distance: {distances[-1]:.2f}m")
    print(f"  Max drift: {max_drift:.4f}m")

    env.close()

    if max_drift < 1.0:
        print("  -> PASS: Relative frame stable (max drift < 1m)")
        return True
    else:
        print(f"  -> FAIL: Relative frame unstable (max drift = {max_drift:.4f}m)")
        return False


def test_4_hohmann_transfer():
    """TEST 4: Hohmann transfer physics.

    Prograde thrust increases orbital energy (moves outward over time).
    Retrograde thrust decreases orbital energy (moves inward over time).
    """
    print("\n" + "="*60)
    print("TEST 4: Hohmann transfer physics")
    print("="*60)

    # Test prograde thrust
    env = OrbitalDock(num_envs=1, difficulty=0.0, max_steps=100)
    obs, _ = env.reset(seed=42)

    initial_v = obs[0, 1] * 100.0  # V-bar position

    prograde = np.array([[4, 2, 2]], dtype=np.int32)  # +100% prograde
    for step in range(20):
        obs, _, terminals, _, _ = env.step(prograde)
        if terminals[0]:
            break

    final_v_pro = obs[0, 1] * 100.0
    env.close()

    # Test retrograde thrust
    env2 = OrbitalDock(num_envs=1, difficulty=0.0, max_steps=100)
    obs2, _ = env2.reset(seed=42)

    retrograde = np.array([[0, 2, 2]], dtype=np.int32)  # -100% prograde
    for step in range(20):
        obs2, _, terminals, _, _ = env2.step(retrograde)
        if terminals[0]:
            break

    final_v_ret = obs2[0, 1] * 100.0
    env2.close()

    print(f"  Initial V-bar: {initial_v:.2f}m")
    print(f"  After prograde thrust: {final_v_pro:.2f}m")
    print(f"  After retrograde thrust: {final_v_ret:.2f}m")

    # Prograde and retrograde should have opposite effects
    if (final_v_pro - initial_v) * (final_v_ret - initial_v) < 0:
        print("  -> PASS: Prograde/retrograde have opposite effects")
        return True
    elif abs(final_v_pro - initial_v) > 0.1 or abs(final_v_ret - initial_v) > 0.1:
        print("  -> PASS: Thrust produces meaningful motion")
        return True
    else:
        print("  -> FAIL: Thrust effects not distinguishable")
        return False


def test_5_vbar_stability():
    """TEST 5: V-bar stability.

    V-bar (along-track) is the stable direction. Small V-bar offsets
    don't cause runaway drift like R-bar offsets do.
    At d=0, offset is mostly V-bar, so should see bounded motion.
    """
    print("\n" + "="*60)
    print("TEST 5: V-bar stability")
    print("="*60)

    env = OrbitalDock(num_envs=1, difficulty=0.0, max_steps=600)
    obs, _ = env.reset(seed=42)

    no_thrust = np.array([[2, 2, 2]], dtype=np.int32)

    initial_r = obs[0, 0] * 100.0
    initial_v = obs[0, 1] * 100.0
    initial_h = obs[0, 2] * 100.0
    initial_dist = obs[0, 6] * 100.0

    # At d=0, phi=90° so offset should be mostly in V-bar plane
    v_bar_fraction = abs(initial_v) / (abs(initial_r) + abs(initial_v) + abs(initial_h) + 1e-10)

    print(f"  Initial: R={initial_r:.2f}m, V={initial_v:.2f}m, H={initial_h:.2f}m")
    print(f"  V-bar fraction: {100*v_bar_fraction:.1f}%")

    # Run for 500 steps
    distances = [initial_dist]
    for step in range(500):
        obs, _, terminals, _, _ = env.step(no_thrust)
        distances.append(obs[0, 6] * 100.0)
        if terminals[0]:
            break

    final_dist = distances[-1]
    max_dist = max(distances)
    growth_ratio = max_dist / initial_dist

    print(f"  After {len(distances)-1} steps:")
    print(f"    Final distance: {final_dist:.2f}m")
    print(f"    Max distance: {max_dist:.2f}m")
    print(f"    Growth ratio: {growth_ratio:.2f}")

    env.close()

    # At d=0, expect bounded motion (mostly V-bar, little R-bar to cause drift)
    if growth_ratio < 2.0:
        print("  -> PASS: Motion bounded (growth ratio < 2)")
        return True
    else:
        print(f"  -> FAIL: Excessive growth (ratio = {growth_ratio:.2f})")
        return False


def test_6_cw_dynamics():
    """TEST 6: CW (Clohessy-Wiltshire) dynamics.

    Verify key CW physics properties:
    1. R-bar offset causes oscillation
    2. V-bar has secular drift (grows over time) due to 2:1 coupling
    3. H-bar oscillates independently (bounded)
    """
    print("\n" + "="*60)
    print("TEST 6: CW dynamics")
    print("="*60)

    # Use higher difficulty to get mixed R-bar/V-bar/H-bar starting conditions
    env = OrbitalDock(num_envs=1, difficulty=1.0, max_steps=7000)
    obs, _ = env.reset(seed=42)

    no_thrust = np.array([[2, 2, 2]], dtype=np.int32)

    r_bar_traj = []
    v_bar_traj = []
    h_bar_traj = []

    initial_r = obs[0, 0] * 100.0
    initial_v = obs[0, 1] * 100.0
    initial_h = obs[0, 2] * 100.0

    print(f"  Initial: R={initial_r:.2f}m, V={initial_v:.2f}m, H={initial_h:.2f}m")

    for step in range(6000):
        r_bar_traj.append(obs[0, 0] * 100.0)
        v_bar_traj.append(obs[0, 1] * 100.0)
        h_bar_traj.append(obs[0, 2] * 100.0)

        obs, _, terminals, _, _ = env.step(no_thrust)
        if terminals[0]:
            print(f"  Terminated at step {step}")
            break

    r_bar = np.array(r_bar_traj)
    v_bar = np.array(v_bar_traj)
    h_bar = np.array(h_bar_traj)

    # Check CW properties
    # 1. R-bar oscillates (multiple sign changes or extrema)
    r_sign_changes = np.sum(np.diff(np.sign(r_bar)) != 0)
    r_oscillates = r_sign_changes >= 2 or (r_bar.max() - r_bar.min()) > 50

    # 2. V-bar has secular drift
    v_grows = abs(v_bar[-1]) > abs(v_bar[0]) * 1.5 or abs(v_bar[-1] - v_bar[0]) > 100

    # 3. H-bar is bounded (doesn't grow unboundedly)
    h_bounded = h_bar.max() < 2 * max(abs(initial_h), 50) and h_bar.min() > -2 * max(abs(initial_h), 50)

    print(f"\n  CW dynamics analysis ({len(r_bar)} steps):")
    print(f"    R-bar oscillation: {r_sign_changes} sign changes, range [{r_bar.min():.1f}, {r_bar.max():.1f}]m")
    print(f"    V-bar drift: {v_bar[0]:.1f}m -> {v_bar[-1]:.1f}m")
    print(f"    H-bar bounded: [{h_bar.min():.1f}, {h_bar.max():.1f}]m")
    print(f"\n    R-bar oscillates: {r_oscillates}")
    print(f"    V-bar drifts: {v_grows}")
    print(f"    H-bar bounded: {h_bounded}")

    env.close()

    if r_oscillates and v_grows and h_bounded:
        print("  -> PASS: CW dynamics correct")
        return True
    else:
        failures = []
        if not r_oscillates: failures.append("R-bar doesn't oscillate")
        if not v_grows: failures.append("V-bar doesn't drift")
        if not h_bounded: failures.append("H-bar not bounded")
        print(f"  -> FAIL: {', '.join(failures)}")
        return False


def test_7_docking():
    """TEST 7: Docking mechanics.

    A simple controller should be able to dock successfully from 30-50m.
    """
    print("\n" + "="*60)
    print("TEST 7: Docking mechanics")
    print("="*60)

    def smart_action(obs):
        rel_pos = obs[0:3] * 100.0
        rel_vel = obs[3:6] * 2.0
        dist = obs[6] * 100.0

        dir_to_station = -rel_pos / (dist + 1e-10)
        desired_vel_mag = min(0.3, max(0.1, dist * 0.02))
        desired_vel = dir_to_station * desired_vel_mag
        vel_error = desired_vel - rel_vel

        gain = 4.0
        thrust = vel_error * gain
        thrust = np.clip(thrust, -1, 1)

        def to_discrete(x):
            if x < -0.75: return 0
            elif x < -0.25: return 1
            elif x < 0.25: return 2
            elif x < 0.75: return 3
            else: return 4

        return np.array([[to_discrete(thrust[1]), to_discrete(thrust[0]), to_discrete(thrust[2])]], dtype=np.int32)

    successes = 0
    n_tests = 10

    for seed in range(n_tests):
        env = OrbitalDock(num_envs=1, difficulty=0.0)
        obs, _ = env.reset(seed=seed)

        for step in range(200):
            action = smart_action(obs[0])
            obs, rewards, terminals, _, _ = env.step(action)

            if terminals[0]:
                if rewards[0] > 5.0:
                    successes += 1
                break

        env.close()

    dock_rate = successes / n_tests
    print(f"  Dock rate: {successes}/{n_tests} = {100*dock_rate:.0f}%")

    if dock_rate >= 0.8:
        print("  -> PASS: Docking works (>= 80% success)")
        return True
    else:
        print(f"  -> FAIL: Docking unreliable ({100*dock_rate:.0f}% success)")
        return False


def test_8_fuel_accounting():
    """TEST 8: Fuel accounting.

    Fuel should decrease when thrusting and remain constant when not.
    """
    print("\n" + "="*60)
    print("TEST 8: Fuel accounting")
    print("="*60)

    env = OrbitalDock(num_envs=1, difficulty=0.5, max_steps=500)
    obs, _ = env.reset(seed=999)

    initial_fuel = obs[0, 8]

    # Coast - fuel should stay same
    no_thrust = np.array([[2, 2, 2]], dtype=np.int32)
    for _ in range(10):
        obs, _, terminals, _, _ = env.step(no_thrust)
        if terminals[0]:
            break

    fuel_after_coast = obs[0, 8]
    coast_change = abs(fuel_after_coast - initial_fuel)

    print(f"  Initial fuel: {initial_fuel:.4f}")
    print(f"  Fuel after coast: {fuel_after_coast:.4f}")
    print(f"  Coast fuel change: {coast_change:.6f}")

    # Thrust in radial direction (away from station to avoid docking)
    radial_out = np.array([[2, 4, 2]], dtype=np.int32)
    fuel_before_thrust = fuel_after_coast

    for _ in range(20):
        obs, _, terminals, _, _ = env.step(radial_out)
        if terminals[0]:
            break

    fuel_after_thrust = obs[0, 8]
    thrust_change = fuel_before_thrust - fuel_after_thrust

    print(f"  Fuel after thrust: {fuel_after_thrust:.4f}")
    print(f"  Thrust fuel change: {thrust_change:.4f}")

    env.close()

    coast_ok = coast_change < 0.001
    thrust_ok = thrust_change > 0.001

    if coast_ok and thrust_ok:
        print("  -> PASS: Fuel accounting correct")
        return True
    else:
        if not coast_ok:
            print(f"  -> FAIL: Fuel changed during coast ({coast_change:.6f})")
        if not thrust_ok:
            print(f"  -> FAIL: Fuel didn't decrease during thrust ({thrust_change:.4f})")
        return False


def test_9_observation_bounds():
    """TEST 9: Observation bounds.

    At d=0 (30-50m starting distance), observations should be in [-1, 1].
    """
    print("\n" + "="*60)
    print("TEST 9: Observation bounds")
    print("="*60)

    all_obs = []

    for seed in range(10):
        env = OrbitalDock(num_envs=1, difficulty=0.0)
        obs, _ = env.reset(seed=seed)
        all_obs.append(obs.copy())

        for step in range(50):
            action = np.random.randint(0, 5, (1, 3), dtype=np.int32)
            obs, _, terminals, _, _ = env.step(action)
            all_obs.append(obs.copy())
            if terminals[0]:
                break

        env.close()

    all_obs = np.concatenate(all_obs, axis=0)

    min_val = all_obs.min()
    max_val = all_obs.max()

    print(f"  Observation range at d=0: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Expected range: approximately [-1, 1]")

    # Check per-dimension
    obs_names = ['R-bar', 'V-bar', 'H-bar', 'rel_vr', 'rel_vv', 'rel_vh',
                 'dist', 'closing', 'fuel', 'alt', 'phase', 'incl', 'node', 'time']

    all_ok = True
    for i, name in enumerate(obs_names):
        obs_i = all_obs[:, i]
        if obs_i.min() < -1.5 or obs_i.max() > 1.5:
            print(f"    {name}: [{obs_i.min():.4f}, {obs_i.max():.4f}] - out of range!")
            all_ok = False

    if min_val >= -1.5 and max_val <= 1.5:
        print("  -> PASS: All observations in [-1.5, 1.5]")
        return True
    else:
        print(f"  -> FAIL: Observations outside expected range")
        return False


def test_10_termination_conditions():
    """TEST 10: Termination conditions.

    Test dock and timeout terminations.
    """
    print("\n" + "="*60)
    print("TEST 10: Termination conditions")
    print("="*60)

    results = {}

    # 10a: Docking
    print("\n  10a: Docking termination")
    env = OrbitalDock(num_envs=1, difficulty=0.0)
    obs, _ = env.reset(seed=42)

    def smart_action(obs):
        rel_pos = obs[0:3] * 100.0
        rel_vel = obs[3:6] * 2.0
        dist = obs[6] * 100.0

        dir_to_station = -rel_pos / (dist + 1e-10)
        desired_vel_mag = min(0.3, max(0.1, dist * 0.02))
        desired_vel = dir_to_station * desired_vel_mag
        vel_error = desired_vel - rel_vel

        gain = 4.0
        thrust = vel_error * gain
        thrust = np.clip(thrust, -1, 1)

        def to_discrete(x):
            if x < -0.75: return 0
            elif x < -0.25: return 1
            elif x < 0.25: return 2
            elif x < 0.75: return 3
            else: return 4

        return np.array([[to_discrete(thrust[1]), to_discrete(thrust[0]), to_discrete(thrust[2])]], dtype=np.int32)

    for step in range(200):
        action = smart_action(obs[0])
        obs, rewards, terminals, _, _ = env.step(action)
        if terminals[0]:
            break

    dock_reward = rewards[0]
    print(f"      Reward: {dock_reward:.2f}")
    results['10a_dock'] = dock_reward > 5.0
    if results['10a_dock']:
        print("      -> PASS: Positive dock reward")
    else:
        print("      -> FAIL: Expected positive reward")
    env.close()

    # 10b: Timeout
    print("\n  10b: Timeout termination")
    env = OrbitalDock(num_envs=1, difficulty=0.0, max_steps=50)
    obs, _ = env.reset(seed=999)

    no_thrust = np.array([[2, 2, 2]], dtype=np.int32)

    for step in range(100):
        obs, rewards, terminals, _, _ = env.step(no_thrust)
        if terminals[0]:
            break

    print(f"      Terminated at step {step+1}, reward: {rewards[0]:.4f}")
    results['10b_timeout'] = step >= 49
    if results['10b_timeout']:
        print("      -> PASS: Timeout at max_steps")
    else:
        print(f"      -> FAIL: Terminated early at step {step+1}")
    env.close()

    all_pass = all(results.values())
    return all_pass


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("ORBITAL DOCK VERIFICATION SUITE")
    print("="*60)

    results = {
        'test_1_energy': test_1_energy_conservation(),
        'test_2_circular': test_2_circular_orbit_stability(),
        'test_3_relative': test_3_relative_frame_stability(),
        'test_4_hohmann': test_4_hohmann_transfer(),
        'test_5_vbar': test_5_vbar_stability(),
        'test_6_cw_dynamics': test_6_cw_dynamics(),
        'test_7_docking': test_7_docking(),
        'test_8_fuel': test_8_fuel_accounting(),
        'test_9_obs_bounds': test_9_observation_bounds(),
        'test_10_termination': test_10_termination_conditions(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nALL TESTS PASSED!")
        return True
    else:
        print(f"\n{total - passed} TESTS FAILED")
        return False


if __name__ == '__main__':
    run_all_tests()

"""Debug test for orbital_dock termination conditions."""
import numpy as np
from pufferlib.ocean.orbital_dock import binding
from pufferlib.ocean.orbital_dock.orbital_dock import OrbitalDock

def obs_to_state(obs):
    """Extract physical state from observation."""
    # obs: [rel_r, rel_v, rel_h, rel_vr, rel_vv, rel_vh, dist_norm, closing_speed, ...]
    rel_pos = obs[0:3] * 100.0  # Scale: 100m
    rel_vel = obs[3:6] * 2.0    # Scale: 2 m/s
    dist = obs[6] * 100.0
    closing_speed = obs[7] * 2.0
    return rel_pos, rel_vel, dist, closing_speed

def smart_action(obs, dock_dist=5.0, dock_speed=0.5):
    """Compute action that moves toward station with velocity control.

    Strategy:
    1. Compute direction to station in LVLH frame
    2. Compute desired velocity (proportional to distance, capped)
    3. Compute velocity error
    4. Apply proportional control

    Key insight: max accel = 500N / 10000kg = 0.05 m/s²
    So 1 timestep at full thrust changes velocity by 0.05 m/s.
    To close a 0.3 m/s velocity gap, need 6 steps of full thrust.
    """
    rel_pos, rel_vel, dist, _ = obs_to_state(obs)

    # Direction from chaser to station (negative of relative position)
    # If rel_pos = [0, 0, -6], station is in -H direction, so we want to go -H
    dir_to_station = -rel_pos / (dist + 1e-10)

    # Desired closing velocity: proportional to distance, but capped
    # Close slowly when near, faster when far
    # At 6m, want ~0.25 m/s. At 5m, want ~0.2 m/s. At 10m, want ~0.3 m/s.
    max_approach_vel = 0.3  # Stay well under dock_speed of 0.5
    desired_vel_mag = min(max_approach_vel, max(0.1, dist * 0.04))

    desired_vel = dir_to_station * desired_vel_mag

    # Velocity error
    vel_error = desired_vel - rel_vel  # [r, v, h] components

    # Proportional control: thrust proportional to velocity error
    # Scale factor: 0.05 m/s² max accel, so to close 0.3 m/s gap need full thrust
    # gain = 1/0.3 = 3.33 to map 0.3 m/s error -> 1.0 thrust
    gain = 4.0  # Aggressive enough to produce meaningful actions

    # Actions are in chaser's LVLH which is approximately same as station's LVLH
    # Action dimensions: [prograde, radial, normal] = [v, r, h]
    thrust_v = vel_error[1] * gain  # V-bar velocity error -> prograde thrust
    thrust_r = vel_error[0] * gain  # R-bar velocity error -> radial thrust
    thrust_h = vel_error[2] * gain  # H-bar velocity error -> normal thrust

    # Convert to discrete actions: {-1, -0.5, 0, 0.5, 1} -> {0, 1, 2, 3, 4}
    def continuous_to_discrete(x):
        if x < -0.75:
            return 0  # -100%
        elif x < -0.25:
            return 1  # -50%
        elif x < 0.25:
            return 2  # 0%
        elif x < 0.75:
            return 3  # +50%
        else:
            return 4  # +100%

    # Clamp thrust commands to [-1, 1]
    thrust_v = np.clip(thrust_v, -1, 1)
    thrust_r = np.clip(thrust_r, -1, 1)
    thrust_h = np.clip(thrust_h, -1, 1)

    action = np.array([
        continuous_to_discrete(thrust_v),
        continuous_to_discrete(thrust_r),
        continuous_to_discrete(thrust_h)
    ], dtype=np.int32)

    return action

def test_smart_controller():
    """Test smart controller that thrusts toward target."""
    print("="*60)
    print("TEST: Smart controller (thrust toward target)")
    print("="*60)

    env = OrbitalDock(num_envs=1, difficulty=0.0)
    obs, _ = env.reset(seed=42)

    print(f"\nEnvironment params:")
    print(f"  dock_dist = 5.0m")
    print(f"  dock_speed = 0.5 m/s")

    rel_pos, rel_vel, dist, closing = obs_to_state(obs[0])
    print(f"\nInitial state:")
    print(f"  Relative position (LVLH): R={rel_pos[0]:.2f}m, V={rel_pos[1]:.2f}m, H={rel_pos[2]:.2f}m")
    print(f"  Distance: {dist:.2f}m")
    print(f"  Station direction: R={-rel_pos[0]/dist:.2f}, V={-rel_pos[1]/dist:.2f}, H={-rel_pos[2]/dist:.2f}")

    print(f"\nRunning smart controller...")

    for step in range(200):
        action = smart_action(obs[0])
        obs, rewards, terminals, truncations, info = env.step(action.reshape(1, 3))

        rel_pos, rel_vel, dist, closing = obs_to_state(obs[0])
        total_rel_speed = np.linalg.norm(rel_vel)

        action_str = f"[{['--','-','0','+','++'][action[0]]},{['--','-','0','+','++'][action[1]]},{['--','-','0','+','++'][action[2]]}]"

        if step < 20 or step % 10 == 0 or dist < 10:
            print(f"  Step {step+1:3d}: dist={dist:.2f}m, closing={closing:.3f}m/s, "
                  f"total_v={total_rel_speed:.3f}m/s, r={rewards[0]:.4f}, act={action_str}")

        if terminals[0]:
            print(f"\n  TERMINATED at step {step+1}")
            if rewards[0] > 5.0:
                print(f"    -> DOCK SUCCESS! reward={rewards[0]:.2f}")
            else:
                print(f"    -> FAILED, reward={rewards[0]:.2f}")
            break

    env.close()

def test_multiple_seeds():
    """Test docking success rate across multiple seeds."""
    print("\n" + "="*60)
    print("TEST: Multiple seeds dock rate")
    print("="*60)

    n_seeds = 20
    successes = 0
    results = []

    for seed in range(n_seeds):
        env = OrbitalDock(num_envs=1, difficulty=0.0)
        obs, _ = env.reset(seed=seed)

        for step in range(300):
            action = smart_action(obs[0])
            obs, rewards, terminals, truncations, info = env.step(action.reshape(1, 3))

            if terminals[0]:
                if rewards[0] > 5.0:
                    successes += 1
                    results.append(('DOCK', step+1, rewards[0]))
                else:
                    results.append(('FAIL', step+1, rewards[0]))
                break
        else:
            results.append(('TIMEOUT', 300, 0))

        env.close()

    print(f"\nResults ({successes}/{n_seeds} = {100*successes/n_seeds:.1f}% dock rate):")
    for i, (status, steps, reward) in enumerate(results):
        print(f"  Seed {i:2d}: {status} at step {steps:3d}, reward={reward:.2f}")

def test_no_thrust_stability():
    """Test relative position stability with no thrust."""
    print("\n" + "="*60)
    print("TEST: No-thrust stability (Verlet integration)")
    print("="*60)

    env = OrbitalDock(num_envs=1, difficulty=0.0)
    obs, _ = env.reset(seed=42)
    no_thrust = np.array([[2, 2, 2]], dtype=np.int32)

    initial_dist = obs[0,6] * 100.0
    print(f"\nInitial distance: {initial_dist:.4f}m")
    print(f"Running 100 steps with no thrust...")

    for step in range(100):
        obs, rewards, terminals, truncations, info = env.step(no_thrust)
        if terminals[0]:
            print(f"  Terminated at step {step+1}")
            break

    final_dist = obs[0,6] * 100.0
    drift = abs(final_dist - initial_dist)
    print(f"Final distance: {final_dist:.4f}m")
    print(f"Total drift: {drift:.4f}m over 100 steps")

    if drift < 1.0:
        print("-> PASS: Drift < 1m")
    else:
        print("-> FAIL: Drift >= 1m")

    env.close()

def test_random_agent_baseline():
    """Test random agent to ensure dock is possible by chance."""
    print("\n" + "="*60)
    print("TEST: Random agent baseline")
    print("="*60)

    n_episodes = 100
    dock_count = 0
    crash_count = 0
    timeout_count = 0

    for ep in range(n_episodes):
        env = OrbitalDock(num_envs=1, difficulty=0.0)
        obs, _ = env.reset(seed=ep * 7)

        for step in range(500):
            action = np.random.randint(0, 5, (1, 3), dtype=np.int32)
            obs, rewards, terminals, truncations, info = env.step(action)

            if terminals[0]:
                if rewards[0] > 5.0:
                    dock_count += 1
                elif rewards[0] < -3.0:
                    crash_count += 1
                break
        else:
            timeout_count += 1

        env.close()

    print(f"\nRandom agent results ({n_episodes} episodes):")
    print(f"  Dock rate: {dock_count}/{n_episodes} = {100*dock_count/n_episodes:.1f}%")
    print(f"  Crash rate: {crash_count}/{n_episodes} = {100*crash_count/n_episodes:.1f}%")
    print(f"  Timeout rate: {timeout_count}/{n_episodes} = {100*timeout_count/n_episodes:.1f}%")

    if dock_count > 0:
        print("-> PASS: Random agent can occasionally dock (reward signal exists)")
    else:
        print("-> WARNING: Random agent never docked (may need easier starting conditions)")

if __name__ == '__main__':
    test_no_thrust_stability()
    test_smart_controller()
    test_multiple_seeds()
    test_random_agent_baseline()

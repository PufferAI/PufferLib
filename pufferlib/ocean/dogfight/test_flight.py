"""
Physics sanity tests for dogfight environment.
Outputs values for manual recording in physics_log.md

Run: cd pufferlib/ocean/dogfight && python test_flight.py
"""
import numpy as np
from dogfight import Dogfight

# Constants (must match dogfight.h)
MAX_SPEED = 250.0
WORLD_MAX_Z = 3000.0

# Theoretical values
THEORETICAL_MAX_SPEED = 143.7  # m/s
THEORETICAL_STALL_SPEED = 39.5  # m/s

RESULTS = {}


def get_speed(obs):
    vx, vy, vz = obs[0, 3] * MAX_SPEED, obs[0, 4] * MAX_SPEED, obs[0, 5] * MAX_SPEED
    return np.sqrt(vx**2 + vy**2 + vz**2)

def get_alt(obs):
    return obs[0, 2] * WORLD_MAX_Z

def test_max_speed():
    """Full throttle level flight - max speed."""
    env = Dogfight(num_envs=1)
    env.reset()
    action = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    for _ in range(1000):  # 20s
        obs, _, term, _, _ = env.step(action)
        if term[0]: env.reset()
    RESULTS['max_speed_100'] = get_speed(obs)
    print(f"max_speed_100:     {RESULTS['max_speed_100']:6.1f} m/s  (expected ~{THEORETICAL_MAX_SPEED:.0f})")

def test_cruise_speed():
    """50% throttle level flight - cruise speed."""
    env = Dogfight(num_envs=1)
    env.reset()
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # 50% throttle
    for _ in range(1000):
        obs, _, term, _, _ = env.step(action)
        if term[0]: env.reset()
    RESULTS['cruise_speed_50'] = get_speed(obs)
    print(f"cruise_speed_50:   {RESULTS['cruise_speed_50']:6.1f} m/s")

def test_zero_throttle():
    """Zero throttle - plane dives to maintain energy."""
    env = Dogfight(num_envs=1)
    env.reset()
    action = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    min_speed = 999
    for _ in range(500):
        obs, _, term, _, _ = env.step(action)
        if term[0]: break
        min_speed = min(min_speed, get_speed(obs))
    RESULTS['min_speed_0_throttle'] = min_speed
    RESULTS['final_speed_0_throttle'] = get_speed(obs)
    print(f"min_speed_0_throt: {min_speed:6.1f} m/s  (stall ~{THEORETICAL_STALL_SPEED:.0f})")
    print(f"final_speed_0_thr: {RESULTS['final_speed_0_throttle']:6.1f} m/s  (diving)")

def test_dive_30deg():
    """Zero throttle, 30° pitch down - stable dive speed."""
    env = Dogfight(num_envs=1)
    env.reset()
    action = np.array([[-1.0, -0.3, 0.0, 0.0, 0.0]], dtype=np.float32)
    for _ in range(500):
        obs, _, term, _, _ = env.step(action)
        if term[0]: break
    RESULTS['dive_30deg_speed'] = get_speed(obs)
    print(f"dive_30deg_speed:  {RESULTS['dive_30deg_speed']:6.1f} m/s")


def test_dive_45deg():
    """Zero throttle, 45° pitch down - stable dive speed."""
    env = Dogfight(num_envs=1)
    env.reset()
    action = np.array([[-1.0, -0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
    for _ in range(500):
        obs, _, term, _, _ = env.step(action)
        if term[0]: break
    RESULTS['dive_45deg_speed'] = get_speed(obs)
    print(f"dive_45deg_speed:  {RESULTS['dive_45deg_speed']:6.1f} m/s")


def test_climb_rate():
    """Full throttle, pitch up - climb rate."""
    env = Dogfight(num_envs=1)
    obs = env.reset()[0]
    initial_alt = get_alt(obs)
    action = np.array([[1.0, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)
    for _ in range(500):  # 10s
        obs, _, term, _, _ = env.step(action)
        if term[0]: break
    final_alt = get_alt(obs)
    climb_rate = (final_alt - initial_alt) / 10.0
    RESULTS['climb_rate'] = climb_rate
    print(f"climb_rate:        {climb_rate:6.1f} m/s")


def test_pitch_direction():
    """Verify positive elevator = nose up."""
    env = Dogfight(num_envs=1)
    env.reset()
    action = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    initial_up_x = None
    for step in range(50):
        obs, _, _, _, _ = env.step(action)
        if step == 0: initial_up_x = obs[0, 10]
    final_up_x = obs[0, 10]
    nose_up = final_up_x > initial_up_x
    RESULTS['pitch_direction'] = 'UP' if nose_up else 'DOWN'
    print(f"pitch_direction:   {RESULTS['pitch_direction']}  ({'OK' if nose_up else 'WRONG'})")


def test_roll_direction():
    """Verify positive ailerons = roll right."""
    env = Dogfight(num_envs=1)
    env.reset()
    action = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    for _ in range(50):
        obs, _, _, _, _ = env.step(action)
    up_y_changed = abs(obs[0, 11]) > 0.1
    RESULTS['roll_works'] = 'YES' if up_y_changed else 'NO'
    print(f"roll_works:        {RESULTS['roll_works']}")


def fmt(key):
    v = RESULTS.get(key)
    if v is None: return 'N/A'
    if isinstance(v, float): return f"{v:.1f}"
    return str(v)

def print_summary():
    """Print copy-pasteable summary."""
    print("\n" + "="*50)
    print("SUMMARY (copy to physics_log.md)")
    print("="*50)
    print(f"| max_speed_100      | {fmt('max_speed_100'):>6} | ~{THEORETICAL_MAX_SPEED:.0f} expected |")
    print(f"| cruise_speed_50    | {fmt('cruise_speed_50'):>6} | |")
    print(f"| min_speed_0_throt  | {fmt('min_speed_0_throttle'):>6} | ~{THEORETICAL_STALL_SPEED:.0f} stall |")
    print(f"| dive_30deg_speed   | {fmt('dive_30deg_speed'):>6} | |")
    print(f"| dive_45deg_speed   | {fmt('dive_45deg_speed'):>6} | |")
    print(f"| climb_rate         | {fmt('climb_rate'):>6} | m/s |")
    print(f"| pitch_direction    | {fmt('pitch_direction'):>6} | should be UP |")
    print(f"| roll_works         | {fmt('roll_works'):>6} | should be YES |")


if __name__ == "__main__":
    print("Physics Sanity Tests")
    print("="*50)
    test_max_speed()
    test_cruise_speed()
    test_zero_throttle()
    test_dive_30deg()
    test_dive_45deg()
    test_climb_rate()
    test_pitch_direction()
    test_roll_direction()
    print_summary()

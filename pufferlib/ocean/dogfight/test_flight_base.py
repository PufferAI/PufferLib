"""
Shared infrastructure for dogfight flight physics tests.
Importable by all test modules.

Run: python pufferlib/ocean/dogfight/test_flight.py
     python pufferlib/ocean/dogfight/test_flight.py --render  # with visualization
     python pufferlib/ocean/dogfight/test_flight.py --render --test pitch_direction  # single test
"""
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='P-51D Physics Validation Tests')
    parser.add_argument('--render', action='store_true', help='Enable visual rendering')
    parser.add_argument('--fps', type=int, default=50, help='Target FPS when rendering (default 50 = real-time, try 5-10 for slow-mo)')
    parser.add_argument('--test', type=str, default=None, help='Run specific test only')
    parser.add_argument('--physics-mode', type=int, default=0, help='Physics mode: 0=simplified (default), 1=realistic')
    return parser.parse_args()


# Parse args once at module load - can be overridden by test modules
_ARGS = None

def get_args():
    """Get parsed args, parsing only once."""
    global _ARGS
    if _ARGS is None:
        _ARGS = parse_args()
    return _ARGS


def get_render_mode():
    """Get render mode from args."""
    args = get_args()
    return 'human' if args.render else None


def get_render_fps():
    """Get render FPS from args."""
    args = get_args()
    return args.fps if args.render else None


def get_physics_mode():
    """Get physics mode from args (0=simplified, 1=realistic)."""
    args = get_args()
    return args.physics_mode


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

# Shared results dictionary for summary
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
    if get_render_mode() and test_name in TEST_HIGHLIGHTS:
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


# =============================================================================
# Mode 1 autopilot helpers (uses autopilot_mode1 module)
# =============================================================================

def is_mode1():
    """Check if current physics mode is Mode 1 (realistic)."""
    return get_physics_mode() == 1


def get_mode1_autopilot():
    """
    Lazily import autopilot_mode1 module.
    Returns the module or None if not needed (Mode 0).
    """
    if not is_mode1():
        return None
    from autopilot_mode1 import (
        hold_pitch, hold_vz, hold_bank, damp_yaw,
        hold_bank_and_level, hold_pitch_and_bank, full_autopilot,
        get_pitch_deg, get_bank_deg, DEFAULT_GAINS
    )
    return {
        'hold_pitch': hold_pitch,
        'hold_vz': hold_vz,
        'hold_bank': hold_bank,
        'damp_yaw': damp_yaw,
        'hold_bank_and_level': hold_bank_and_level,
        'hold_pitch_and_bank': hold_pitch_and_bank,
        'full_autopilot': full_autopilot,
        'get_pitch_deg': get_pitch_deg,
        'get_bank_deg': get_bank_deg,
        'DEFAULT_GAINS': DEFAULT_GAINS,
    }

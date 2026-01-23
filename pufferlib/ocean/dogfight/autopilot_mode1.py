"""
Mode 1 Autopilot Helpers for Flight Tests

Mode 1 (realistic 6DOF physics) has stability derivatives that create
nose-down moments at positive AOA. Tests need active control to hold
attitudes that Mode 0 (simplified) holds passively.

PID Gains from pid_tune.py sweep (straight_level_mode1 scenario):
    pitch_kp: 0.2, pitch_kd: 0.1   - controls vz/pitch via elevator
    roll_kp: 1.0,  roll_kd: 0.1    - controls bank via aileron
    yaw_kp: 0.1,   yaw_kd: 0.02    - damps yaw rate via rudder

Key insight: Mode 1 physics uses angular velocities (omega) directly,
so we read omega_x/y/z from state for D terms instead of finite differences.
"""

import numpy as np


# Default gains from pid_tune.py sweep
DEFAULT_GAINS = {
    # Elevator (pitch/vz control)
    'pitch_kp': 0.2,
    'pitch_kd': 0.1,
    # Aileron (bank control)
    'roll_kp': 1.0,
    'roll_kd': 0.1,
    # Rudder (yaw damping)
    'yaw_kp': 0.1,
    'yaw_kd': 0.02,
}


def get_pitch_deg(state):
    """Get pitch angle in degrees from state's forward vector."""
    return np.degrees(np.arcsin(np.clip(state['fwd_z'], -1.0, 1.0)))


def get_bank_deg(state):
    """
    Get bank angle in degrees.
    Positive = right bank, Negative = left bank.
    """
    up_z, up_y = state['up_z'], state['up_y']
    bank = np.arccos(np.clip(up_z, -1.0, 1.0))
    # up_y < 0 means canopy tilted right = right bank (positive)
    return np.degrees(bank if up_y < 0 else -bank)


def get_heading_deg(state):
    """Get heading in degrees (0=+X, 90=+Y)."""
    return np.degrees(np.arctan2(state['fwd_y'], state['fwd_x']))


def hold_pitch(state, target_pitch_deg, gains=None):
    """
    Hold a specific pitch angle using PD control.

    Args:
        state: Dict from env.get_state()
        target_pitch_deg: Desired pitch angle in degrees (positive = nose up)
        gains: Dict with 'pitch_kp', 'pitch_kd' (uses defaults if None)

    Returns:
        elevator: Control input [-1, 1]
    """
    if gains is None:
        gains = DEFAULT_GAINS

    pitch = get_pitch_deg(state)
    omega_pitch = np.degrees(state['omega_y'])  # Pitch rate from physics

    error = target_pitch_deg - pitch

    # Negative elevator = pull = nose UP
    # So if pitch is below target (error > 0), we need negative elevator
    # D term opposes pitch rate
    elevator = -gains['pitch_kp'] * error - gains['pitch_kd'] * omega_pitch

    return np.clip(elevator, -1.0, 1.0)


def hold_vz(state, target_vz, gains=None):
    """
    Hold a target vertical speed (vz) using PD control.

    Good for level flight (target_vz=0) or constant rate climb/descent.

    Args:
        state: Dict from env.get_state()
        target_vz: Desired vertical speed in m/s (positive = climbing)
        gains: Dict with 'pitch_kp', 'pitch_kd' (uses defaults if None)

    Returns:
        elevator: Control input [-1, 1]
    """
    if gains is None:
        gains = DEFAULT_GAINS

    vz = state['vz']
    omega_pitch = np.degrees(state['omega_y'])

    error = target_vz - vz

    # If descending (vz < target), error > 0, need nose UP (negative elevator)
    # Scale error to match pitch-based control (rough conversion: 5 m/s ~ 3 deg pitch)
    elevator = -gains['pitch_kp'] * 0.6 * error - gains['pitch_kd'] * omega_pitch

    return np.clip(elevator, -1.0, 1.0)


def hold_bank(state, target_bank_deg, gains=None):
    """
    Hold a specific bank angle using PD control.

    Args:
        state: Dict from env.get_state()
        target_bank_deg: Desired bank angle (positive = right bank)
        gains: Dict with 'roll_kp', 'roll_kd' (uses defaults if None)

    Returns:
        aileron: Control input [-1, 1]
    """
    if gains is None:
        gains = DEFAULT_GAINS

    bank = get_bank_deg(state)
    omega_roll = np.degrees(state['omega_x'])  # Roll rate from physics

    error = target_bank_deg - bank

    # Positive aileron = roll right
    # If bank is below target (error > 0), need positive aileron
    # D term opposes roll rate
    aileron = gains['roll_kp'] * error - gains['roll_kd'] * omega_roll

    return np.clip(aileron, -1.0, 1.0)


def damp_yaw(state, gains=None):
    """
    Damp yaw rate to zero (straight flight).

    Args:
        state: Dict from env.get_state()
        gains: Dict with 'yaw_kp', 'yaw_kd' (uses defaults if None)

    Returns:
        rudder: Control input [-1, 1]
    """
    if gains is None:
        gains = DEFAULT_GAINS

    omega_yaw = np.degrees(state['omega_z'])  # Yaw rate from physics

    # Target yaw rate = 0, so error = -omega_yaw
    # D term is just omega_yaw itself
    rudder = -gains['yaw_kp'] * omega_yaw - gains['yaw_kd'] * omega_yaw

    return np.clip(rudder, -1.0, 1.0)


def hold_bank_and_level(state, target_bank_deg, gains=None):
    """
    Coordinated turn: hold bank angle, keep nose level (vz ~ 0).

    In a banked turn, the lift vector is tilted, so some extra back pressure
    is needed to maintain altitude. This function combines bank hold with
    vz-based pitch control.

    Args:
        state: Dict from env.get_state()
        target_bank_deg: Desired bank angle (positive = right bank)
        gains: Dict with all gains (uses defaults if None)

    Returns:
        (elevator, aileron): Tuple of control inputs [-1, 1]
    """
    if gains is None:
        gains = DEFAULT_GAINS

    aileron = hold_bank(state, target_bank_deg, gains)

    # In a banked turn, need extra back pressure proportional to bank angle
    # Load factor n = 1/cos(bank), so for 30 deg bank need ~1.15x lift
    bank_rad = np.radians(abs(target_bank_deg))
    if bank_rad < np.radians(80):
        # Extra pitch needed increases with bank angle
        extra_pitch_bias = -0.05 * (1/np.cos(bank_rad) - 1) * 10  # Scaled pull
    else:
        extra_pitch_bias = -0.3  # Near knife-edge, just add pull

    # Base level flight + extra pull for turn
    elevator = hold_vz(state, 0.0, gains) + extra_pitch_bias
    elevator = np.clip(elevator, -1.0, 1.0)

    return elevator, aileron


def hold_pitch_and_bank(state, target_pitch_deg, target_bank_deg, gains=None):
    """
    Hold both pitch angle and bank angle.

    Useful for setting up specific flight conditions (climb + turn, etc).

    Args:
        state: Dict from env.get_state()
        target_pitch_deg: Desired pitch angle (positive = nose up)
        target_bank_deg: Desired bank angle (positive = right bank)
        gains: Dict with all gains (uses defaults if None)

    Returns:
        (elevator, aileron): Tuple of control inputs [-1, 1]
    """
    if gains is None:
        gains = DEFAULT_GAINS

    elevator = hold_pitch(state, target_pitch_deg, gains)
    aileron = hold_bank(state, target_bank_deg, gains)

    return elevator, aileron


def full_autopilot(state, target_pitch_deg=0.0, target_bank_deg=0.0,
                   target_vz=None, damp_yaw_rate=True, gains=None):
    """
    Full 3-axis autopilot for stable flight.

    Can operate in pitch-hold or vz-hold mode for elevator.

    Args:
        state: Dict from env.get_state()
        target_pitch_deg: Desired pitch angle (used if target_vz is None)
        target_bank_deg: Desired bank angle (positive = right bank)
        target_vz: If provided, holds vz instead of pitch
        damp_yaw_rate: Whether to damp yaw oscillations
        gains: Dict with all gains (uses defaults if None)

    Returns:
        (elevator, aileron, rudder): Tuple of control inputs [-1, 1]
    """
    if gains is None:
        gains = DEFAULT_GAINS

    # Elevator: vz-hold or pitch-hold
    if target_vz is not None:
        elevator = hold_vz(state, target_vz, gains)
    else:
        elevator = hold_pitch(state, target_pitch_deg, gains)

    # Aileron: bank hold
    aileron = hold_bank(state, target_bank_deg, gains)

    # Rudder: yaw damping
    rudder = damp_yaw(state, gains) if damp_yaw_rate else 0.0

    return elevator, aileron, rudder


# Convenience function to check if we're in Mode 1
def is_mode1(physics_mode):
    """Check if physics_mode is realistic (Mode 1)."""
    return physics_mode == 1

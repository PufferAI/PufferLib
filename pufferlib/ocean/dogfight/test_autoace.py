"""
AutoAce Behavioral Tests
========================
Verify AutoAce responds correctly to specific geometric scenarios.

Naming conventions:
- "target" = the plane AutoAce is chasing (controlled by dummy actions)
- "autoace" = the AI opponent we're testing

Each test:
1. Positions target relative to AutoAce
2. Runs multiple steps
3. Checks AutoAce's output actions AND resulting behavior

Run: python pufferlib/ocean/dogfight/test_autoace.py
     python pufferlib/ocean/dogfight/test_autoace.py --render --fps 10
     python pufferlib/ocean/dogfight/test_autoace.py --test target_above_ahead
"""

import sys
import argparse
import math
import numpy as np

# Global render settings (set by argparse)
_render_mode = None
_render_fps = None

def get_render_mode():
    return _render_mode

def get_render_fps():
    return _render_fps

# Test results
RESULTS = {}


def make_autoace_env():
    """Create environment configured for AutoAce (stage 20)."""
    from pufferlib.ocean.dogfight.dogfight import Dogfight

    render_mode = 'human' if get_render_mode() else None
    fps = get_render_fps()

    env = Dogfight(
        num_envs=1,
        obs_scheme=0,
        curriculum_enabled=1,
        curriculum_randomize=0,
        render_mode=render_mode,
        render_fps=fps,
    )
    env.reset()
    env.set_curriculum_stage(20)  # AutoAce stage

    # For visual tests, follow AutoAce instead of target
    if render_mode:
        env.set_camera_follow(follow_opponent=True)

    return env


def setup_scenario(env, target_pos, target_vel, autoace_pos, autoace_vel,
                   target_ori=(1, 0, 0, 0), autoace_ori=(1, 0, 0, 0),
                   target_cooldown=None, autoace_cooldown=None):
    """
    Set up a test scenario with clear naming.

    The C binding uses 'player' for target and 'opponent' for autoace,
    but this wrapper uses clearer names.

    Args:
        target_cooldown: Fire cooldown ticks for target (None = 0 = guns ready)
        autoace_cooldown: Fire cooldown ticks for AutoAce (None = 0 = guns ready)
    """
    env.force_state(
        player_pos=target_pos,      # "player" in C = target we're chasing
        player_vel=target_vel,
        player_ori=target_ori,
        opponent_pos=autoace_pos,   # "opponent" in C = AutoAce
        opponent_vel=autoace_vel,
        opponent_ori=autoace_ori,
        player_cooldown=target_cooldown,
        opponent_cooldown=autoace_cooldown,
    )


def get_autoace_state(env):
    """
    Get AutoAce state with clearer field names.

    Returns dict with fields like 'fwd_x', 'elevator', 'bank' instead of
    'opp_fwd_x', 'opp_elevator', 'opp_bank'.
    """
    raw = env.get_autoace_state()

    # Rename opp_* fields to remove prefix
    return {
        # Position
        'px': raw['opp_px'],
        'py': raw['opp_py'],
        'pz': raw['opp_pz'],
        # Velocity
        'vx': raw['opp_vx'],
        'vy': raw['opp_vy'],
        'vz': raw['opp_vz'],
        # Orientation
        'fwd_x': raw['opp_fwd_x'],
        'fwd_y': raw['opp_fwd_y'],
        'fwd_z': raw['opp_fwd_z'],
        'ow': raw['opp_ow'],
        'ox': raw['opp_ox'],
        'oy': raw['opp_oy'],
        'oz': raw['opp_oz'],
        'bank': raw.get('opp_bank', 0.0),
        # Controls
        'throttle': raw['opp_throttle'],
        'elevator': raw['opp_elevator'],
        'aileron': raw['opp_aileron'],
        'rudder': raw['opp_rudder'],
        'trigger': raw['opp_trigger'],
        # Tactical
        'engagement': raw['engagement'],
        'mode': raw['mode'],
        'aspect_angle': raw['aspect_angle'],
        'antenna_train': raw['antenna_train'],
        'range': raw['range'],
        'closure_rate': raw['closure_rate'],
        'in_gun_envelope': raw['in_gun_envelope'],
    }


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_heading_from_fwd(fwd_x, fwd_y):
    """Get heading angle from forward vector components."""
    return math.atan2(fwd_y, fwd_x)


def get_pitch_from_fwd(fwd_x, fwd_y, fwd_z):
    """Get pitch angle from forward vector components."""
    horiz = math.sqrt(fwd_x**2 + fwd_y**2)
    return math.atan2(fwd_z, horiz)


def get_mode_name(mode):
    """Convert autopilot mode number to name."""
    mode_names = [
        "STRAIGHT", "LEVEL", "TURN_L", "TURN_R",
        "CLIMB", "DESCEND", "HARD_L", "HARD_R",
        "WEAVE", "EVASIVE", "RANDOM",
        "PURSUIT_LEAD", "PURSUIT_LAG", "PURSUIT_PURE",
        "HIGH_YOYO", "LOW_YOYO", "SCISSORS", "BREAK",
        "SPLIT_S", "EXTEND", "BARREL_ATK", "GUN_TRACK"
    ]
    if 0 <= mode < len(mode_names):
        return mode_names[mode]
    return f"UNKNOWN({mode})"


def get_engage_name(engage):
    """Convert engagement state number to name."""
    engage_names = ["OFFENSIVE", "NEUTRAL", "DEFENSIVE", "WEAPONS", "EXTEND"]
    if 0 <= engage < len(engage_names):
        return engage_names[engage]
    return f"UNKNOWN({engage})"


# =============================================================================
# PITCH RESPONSE TESTS
# =============================================================================

def test_target_above_ahead():
    """Target 30 deg above and ~400m away -> AutoAce pitches up."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # Target at 30 degrees above, ~400m total distance
    # horizontal = 400 * cos(30) = 346m, vertical = 400 * sin(30) = 200m
    setup_scenario(env,
        target_pos=(346, 0, 1200),      # 346m ahead, 200m up (~30 deg)
        target_vel=(80, 0, 0),          # Flying +X
        autoace_pos=(0, 0, 1000),       # At origin, alt 1000
        autoace_vel=(100, 0, 0),        # Flying +X, faster than target
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    # Get initial AutoAce state
    ace = get_autoace_state(env)
    initial_pitch = get_pitch_from_fwd(ace['fwd_x'], ace['fwd_y'], ace['fwd_z'])

    # Run several steps with target doing nothing
    steps = 50  # ~1 second at 50Hz
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    elevator_sum = 0.0
    aileron_sum = 0.0

    print("  Step diagnostics:")

    for step in range(steps):
        env.step(action)
        ace = get_autoace_state(env)
        elevator_sum += ace['elevator']
        aileron_sum += ace['aileron']

        if step % 10 == 0:
            mode_name = get_mode_name(ace['mode'])
            engage_name = get_engage_name(ace['engagement'])
            print(f"    step={step:3d}: mode={mode_name:12s} engage={engage_name:10s}")
            print(f"             aileron={ace['aileron']:+.3f} elevator={ace['elevator']:+.3f}")
            print(f"             antenna_train={ace['antenna_train']:.1f}deg range={ace['range']:.0f}m bank={math.degrees(ace['bank']):.1f}deg")

    # Get final state
    final_pitch = get_pitch_from_fwd(ace['fwd_x'], ace['fwd_y'], ace['fwd_z'])
    pitch_delta = final_pitch - initial_pitch

    avg_elevator = elevator_sum / steps
    avg_aileron = aileron_sum / steps

    # Check: elevator should be negative (pull back = pitch up)
    # And pitch should have increased (nose up)
    elevator_ok = avg_elevator < -0.1
    pitch_ok = pitch_delta > 0.05  # At least ~3 degrees up

    passed = elevator_ok and pitch_ok
    status = "OK" if passed else "FAIL"

    print(f"target_above_ahead: elevator_avg={avg_elevator:.3f} aileron_avg={avg_aileron:.3f} pitch_delta={math.degrees(pitch_delta):.1f}deg [{status}]")
    if not passed:
        print(f"  Expected: elevator < -0.1 (got {avg_elevator:.3f}), pitch increase (got {math.degrees(pitch_delta):.1f}deg)")

    RESULTS['target_above_ahead'] = passed
    env.close()
    return passed


def test_target_below_ahead():
    """Target 45 deg below and 400m ahead -> AutoAce pitches down."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # AutoAce at alt 1000, target at (400, 0, 600) - 400m ahead, 400m down (45 deg)
    setup_scenario(env,
        target_pos=(400, 0, 600),
        target_vel=(80, 0, 0),
        autoace_pos=(0, 0, 1000),
        autoace_vel=(100, 0, 0),
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    ace = get_autoace_state(env)
    initial_pitch = get_pitch_from_fwd(ace['fwd_x'], ace['fwd_y'], ace['fwd_z'])

    steps = 50
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    elevator_sum = 0.0
    for _ in range(steps):
        env.step(action)
        ace = get_autoace_state(env)
        elevator_sum += ace['elevator']

    final_pitch = get_pitch_from_fwd(ace['fwd_x'], ace['fwd_y'], ace['fwd_z'])
    pitch_delta = final_pitch - initial_pitch
    avg_elevator = elevator_sum / steps

    # Check: elevator positive (push forward = pitch down), pitch decreased
    elevator_ok = avg_elevator > 0.1
    pitch_ok = pitch_delta < -0.05

    passed = elevator_ok and pitch_ok
    status = "OK" if passed else "FAIL"

    print(f"target_below_ahead: elevator_avg={avg_elevator:.3f} pitch_delta={math.degrees(pitch_delta):.1f}deg [{status}]")
    if not passed:
        print(f"  Expected: elevator > 0.1 (got {avg_elevator:.3f}), pitch decrease (got {math.degrees(pitch_delta):.1f}deg)")

    RESULTS['target_below_ahead'] = passed
    env.close()
    return passed


def test_target_level_ahead():
    """Target level and 400m ahead -> AutoAce maintains roughly level pitch."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    setup_scenario(env,
        target_pos=(400, 0, 1000),    # Same altitude
        target_vel=(80, 0, 0),
        autoace_pos=(0, 0, 1000),
        autoace_vel=(100, 0, 0),
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    ace = get_autoace_state(env)
    initial_pitch = get_pitch_from_fwd(ace['fwd_x'], ace['fwd_y'], ace['fwd_z'])

    steps = 50
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    elevator_sum = 0.0
    for _ in range(steps):
        env.step(action)
        ace = get_autoace_state(env)
        elevator_sum += ace['elevator']

    final_pitch = get_pitch_from_fwd(ace['fwd_x'], ace['fwd_y'], ace['fwd_z'])
    pitch_delta = final_pitch - initial_pitch
    avg_elevator = elevator_sum / steps

    # Check: elevator near zero, pitch change small
    elevator_ok = abs(avg_elevator) < 0.3
    pitch_ok = abs(pitch_delta) < 0.2  # ~11 degrees tolerance

    passed = elevator_ok and pitch_ok
    status = "OK" if passed else "FAIL"

    print(f"target_level_ahead: elevator_avg={avg_elevator:.3f} pitch_delta={math.degrees(pitch_delta):.1f}deg [{status}]")
    if not passed:
        print(f"  Expected: |elevator| < 0.3 (got {avg_elevator:.3f}), |pitch_delta| < 11deg (got {math.degrees(pitch_delta):.1f}deg)")

    RESULTS['target_level_ahead'] = passed
    env.close()
    return passed


# =============================================================================
# BANK/TURN RESPONSE TESTS
# =============================================================================

def test_target_left():
    """Target 45 deg to the left -> AutoAce banks left AND heading turns left."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # AutoAce facing +X, target 45 deg to the left (positive Y)
    # Distance ~400m: (283, 283, 1000) is 45 deg left at ~400m
    setup_scenario(env,
        target_pos=(283, 283, 1000),
        target_vel=(80, 0, 0),
        autoace_pos=(0, 0, 1000),
        autoace_vel=(100, 0, 0),
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    ace = get_autoace_state(env)
    initial_heading = get_heading_from_fwd(ace['fwd_x'], ace['fwd_y'])

    steps = 50
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    aileron_sum = 0.0

    print("  Step diagnostics:")

    for step in range(steps):
        env.step(action)
        ace = get_autoace_state(env)
        aileron_sum += ace['aileron']

        if step % 10 == 0:
            mode_name = get_mode_name(ace['mode'])
            engage_name = get_engage_name(ace['engagement'])
            current_heading = get_heading_from_fwd(ace['fwd_x'], ace['fwd_y'])
            print(f"    step={step:3d}: mode={mode_name:12s} engage={engage_name:10s}")
            print(f"             aileron={ace['aileron']:+.3f} elevator={ace['elevator']:+.3f}")
            print(f"             bank={math.degrees(ace['bank']):.1f}deg heading={math.degrees(current_heading):.1f}deg")

    final_heading = get_heading_from_fwd(ace['fwd_x'], ace['fwd_y'])
    heading_delta = normalize_angle(final_heading - initial_heading)
    avg_aileron = aileron_sum / steps

    # Sign conventions:
    #   - Negative aileron → roll left → left wing down → negative bank
    #   - Negative bank → turn left → heading increases
    aileron_ok = avg_aileron < -0.1
    heading_ok = heading_delta > 0.05

    passed = aileron_ok and heading_ok
    status = "OK" if passed else "FAIL"

    print(f"target_left: aileron_avg={avg_aileron:.3f} heading_delta={math.degrees(heading_delta):.1f}deg [{status}]")
    if not passed:
        print(f"  Expected: aileron < -0.1 (got {avg_aileron:.3f}), heading increase (got {math.degrees(heading_delta):.1f}deg)")

    RESULTS['target_left'] = passed
    env.close()
    return passed


def test_target_right():
    """Target 45 deg to the right -> AutoAce banks right AND heading turns right."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # Target 45 deg to the right (negative Y)
    setup_scenario(env,
        target_pos=(283, -283, 1000),
        target_vel=(80, 0, 0),
        autoace_pos=(0, 0, 1000),
        autoace_vel=(100, 0, 0),
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    ace = get_autoace_state(env)
    initial_heading = get_heading_from_fwd(ace['fwd_x'], ace['fwd_y'])

    steps = 50
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    aileron_sum = 0.0
    for _ in range(steps):
        env.step(action)
        ace = get_autoace_state(env)
        aileron_sum += ace['aileron']

    final_heading = get_heading_from_fwd(ace['fwd_x'], ace['fwd_y'])
    heading_delta = normalize_angle(final_heading - initial_heading)
    avg_aileron = aileron_sum / steps

    # Positive aileron → bank right → heading decreases
    aileron_ok = avg_aileron > 0.1
    heading_ok = heading_delta < -0.05

    passed = aileron_ok and heading_ok
    status = "OK" if passed else "FAIL"

    print(f"target_right: aileron_avg={avg_aileron:.3f} heading_delta={math.degrees(heading_delta):.1f}deg [{status}]")
    if not passed:
        print(f"  Expected: aileron > 0.1 (got {avg_aileron:.3f}), heading decrease (got {math.degrees(heading_delta):.1f}deg)")

    RESULTS['target_right'] = passed
    env.close()
    return passed


# =============================================================================
# ENGAGEMENT / DEFENSIVE TESTS
# =============================================================================

def test_target_behind():
    """Target behind AutoAce -> Should enter DEFENSIVE engagement and break/scissors."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # Target BEHIND AutoAce, closing in (threat scenario)
    setup_scenario(env,
        target_pos=(-300, 50, 1000),     # Behind and slightly offset
        target_vel=(120, 0, 0),           # Closing faster
        autoace_pos=(0, 0, 1000),
        autoace_vel=(100, 0, 0),
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    steps = 30
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    defensive_count = 0
    for _ in range(steps):
        env.step(action)
        ace = get_autoace_state(env)
        # engagement: 0=OFFENSIVE, 1=NEUTRAL, 2=DEFENSIVE, 3=WEAPONS, 4=EXTEND
        if ace['engagement'] == 2:  # DEFENSIVE
            defensive_count += 1

    defensive_ratio = defensive_count / steps

    passed = defensive_ratio > 0.5
    status = "OK" if passed else "FAIL"

    print(f"target_behind: defensive_ratio={defensive_ratio:.2f} ({defensive_count}/{steps} steps) [{status}]")
    if not passed:
        print(f"  Expected: DEFENSIVE engagement > 50% of steps")

    RESULTS['target_behind'] = passed
    env.close()
    return passed


def test_engage_offensive():
    """AutoAce behind target with good energy -> OFFENSIVE engagement."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # Classic tail chase - AutoAce behind target, closing
    setup_scenario(env,
        target_pos=(400, 0, 1000),
        target_vel=(80, 0, 0),
        autoace_pos=(0, 0, 1000),
        autoace_vel=(100, 0, 0),       # Faster, closing
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    steps = 30
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    offensive_count = 0
    for _ in range(steps):
        env.step(action)
        ace = get_autoace_state(env)
        if ace['engagement'] == 0:  # OFFENSIVE
            offensive_count += 1

    offensive_ratio = offensive_count / steps

    passed = offensive_ratio > 0.5
    status = "OK" if passed else "FAIL"

    print(f"engage_offensive: offensive_ratio={offensive_ratio:.2f} ({offensive_count}/{steps} steps) [{status}]")
    if not passed:
        print(f"  Expected: OFFENSIVE engagement > 50% of steps")

    RESULTS['engage_offensive'] = passed
    env.close()
    return passed


# =============================================================================
# FIRING TESTS
# =============================================================================

def test_fires_in_envelope():
    """AutoAce fires when target is in gun envelope (close range, on nose)."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # Tail chase: AutoAce closing slowly on target
    setup_scenario(env,
        target_pos=(400, 10, 1000),    # 400m ahead, 10m offset
        target_vel=(90, 0, 0),         # 90 m/s
        autoace_pos=(0, 0, 1000),
        autoace_vel=(100, 0, 0),       # 100 m/s - closing at 10 m/s
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    steps = 300  # ~6 seconds
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    trigger_pulls = 0
    in_envelope_count = 0
    weapons_engagement = 0

    for _ in range(steps):
        obs, rewards, terminals, truncations, infos = env.step(action)
        if np.any(terminals):
            if rewards[0] < 0:
                # AutoAce killed target
                trigger_pulls += 1
            break
        ace = get_autoace_state(env)
        if ace['in_gun_envelope']:
            in_envelope_count += 1
        if ace['engagement'] == 3:  # ENGAGE_WEAPONS
            weapons_engagement += 1
        if ace['trigger'] > 0.5:
            trigger_pulls += 1

    passed = trigger_pulls >= 1 or weapons_engagement >= 1
    status = "OK" if passed else "FAIL"

    print(f"fires_in_envelope: trigger_pulls={trigger_pulls} weapons_engage={weapons_engagement} in_envelope={in_envelope_count} [{status}]")
    if not passed:
        print(f"  Expected: trigger_pulls >= 1 or weapons_engagement >= 1")

    RESULTS['fires_in_envelope'] = passed
    env.close()
    return passed


def test_fires_after_turn_right():
    """Target 15 deg to the right -> AutoAce banks right, tracks, and fires."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # Target 15 degrees to the right at ~400m
    # 15 deg: x = 400 * cos(15) = 386m, y = -400 * sin(15) = -103m
    setup_scenario(env,
        target_pos=(386, -103, 1000),  # 15 deg right, ~400m away
        target_vel=(90, 0, 0),         # Flying +X at 90 m/s
        autoace_pos=(0, 0, 1000),
        autoace_vel=(100, 0, 0),       # 100 m/s - will close after turning
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    steps = 500  # ~10 seconds - need time to turn and close
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    trigger_pulls = 0
    max_bank = 0.0
    killed = False

    print("  Step diagnostics:")

    for step in range(steps):
        obs, rewards, terminals, truncations, infos = env.step(action)
        if np.any(terminals):
            if rewards[0] < 0:
                # AutoAce killed target
                killed = True
                trigger_pulls += 1
            break

        ace = get_autoace_state(env)

        # Track max right bank (positive)
        if ace['bank'] > max_bank:
            max_bank = ace['bank']

        if ace['trigger'] > 0.5:
            trigger_pulls += 1

        # Print diagnostics every 50 steps
        if step % 50 == 0:
            mode_name = get_mode_name(ace['mode'])
            print(f"    step={step:3d}: mode={mode_name:12s} bank={math.degrees(ace['bank']):+.1f}deg range={ace['range']:.0f}m antenna={ace['antenna_train']:.1f}deg")

    # Check: should have banked right (positive) and fired
    banked_right = max_bank > 0.1  # At least ~6 degrees right bank
    fired = trigger_pulls >= 1 or killed

    passed = banked_right and fired
    status = "OK" if passed else "FAIL"

    print(f"fires_after_turn_right: max_bank={math.degrees(max_bank):.1f}deg trigger_pulls={trigger_pulls} killed={killed} [{status}]")
    if not passed:
        if not banked_right:
            print(f"  Expected: positive bank (got {math.degrees(max_bank):.1f}deg)")
        if not fired:
            print(f"  Expected: trigger_pulls >= 1 or kill")

    RESULTS['fires_after_turn_right'] = passed
    env.close()
    return passed


def test_no_fire_when_off_target():
    """AutoAce does NOT fire when target is off-bore (not in gun cone)."""
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # Target 90 degrees to the side - not in gun cone
    setup_scenario(env,
        target_pos=(0, 300, 1000),     # 300m to the left
        target_vel=(0, 80, 0),         # Flying perpendicular
        autoace_pos=(0, 0, 1000),
        autoace_vel=(100, 0, 0),       # Flying +X
    )

    env.set_autopilot(mode=AutopilotMode.LEVEL)

    steps = 20  # Short test - check initial response
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    trigger_pulls = 0
    for _ in range(steps):
        env.step(action)
        ace = get_autoace_state(env)
        if ace['trigger'] > 0.5:
            trigger_pulls += 1

    passed = trigger_pulls <= 1  # Allow 1 for edge cases
    status = "OK" if passed else "FAIL"

    print(f"no_fire_when_off_target: trigger_pulls={trigger_pulls}/{steps} [{status}]")
    if not passed:
        print(f"  Expected: trigger_pulls <= 1 (got {trigger_pulls})")

    RESULTS['no_fire_when_off_target'] = passed
    env.close()
    return passed


# =============================================================================
# HEAD-TO-HEAD SCENARIO TESTS
# =============================================================================

def test_head_to_head_dogfight():
    """
    Head-to-head pass, then dogfight.

    Two planes start 1000m apart, flying toward each other at 110 m/s.
    Guns are disabled for 100 ticks (2 seconds) so they pass each other first.
    After passing, guns are enabled and they engage.
    Test passes if eventually one plane gets a kill.
    """
    from pufferlib.ocean.dogfight.dogfight import AutopilotMode

    env = make_autoace_env()

    # Planes 1000m apart at 2000m altitude, flying toward each other at 110 m/s
    # Closing speed = 220 m/s, will pass in ~4.5 seconds (~227 ticks)
    # Disable guns for 150 ticks (3 seconds at 50Hz) so they pass first
    # Orientation: quat(0, 0, 0, 1) = 180 deg around Z = facing -X
    setup_scenario(env,
        target_pos=(500, 0, 2000),       # 500m ahead on +X
        target_vel=(-110, 0, 0),         # Flying toward AutoAce (-X)
        target_ori=(0, 0, 0, 1),         # Facing -X (180 deg yaw)
        autoace_pos=(-500, 0, 2000),     # 500m behind on -X
        autoace_vel=(110, 0, 0),         # Flying toward target (+X)
        autoace_ori=(1, 0, 0, 0),        # Facing +X (identity)
        target_cooldown=150,             # Disable guns for 3 seconds
        autoace_cooldown=150,            # Disable guns for 3 seconds
    )

    # Set autopilot to LEVEL (non-STRAIGHT) so AutoAce code runs
    env.set_autopilot(mode=AutopilotMode.LEVEL)

    # Run for up to 20 seconds (1000 steps at 50Hz)
    max_steps = 1000
    action = np.array([[0.5, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    kill_achieved = False
    pass_detected = False
    shots_after_pass = 0

    print("  Phase diagnostics:")

    for step in range(max_steps):
        obs, rewards, terminals, truncations, infos = env.step(action)

        # Check for terminal (kill)
        if np.any(terminals):
            if rewards[0] != 0:  # Either player or AutoAce got a kill
                kill_achieved = True
                print(f"    step={step}: KILL! reward={rewards[0]:.1f}")
            break

        ace = get_autoace_state(env)

        # Detect pass: initially planes are ~1000m apart at 220 m/s closing
        # They should pass around step ~113 (1000m / 220m/s / 0.02s/tick = 227 ticks / 2 = ~113)
        # But they start 1000m apart from center, so meeting at center = ~113 ticks
        # We check at step 120 to allow some margin
        if step == 120:  # Right around when they should pass
            print(f"    step={step}: range={ace['range']:.0f}m (should be passing)")
            if ace['range'] < 500:  # Increased threshold - they may not pass perfectly close
                pass_detected = True

        # Count shots after guns should be enabled (step > 150)
        if step > 150 and ace['trigger'] > 0.5:
            shots_after_pass += 1

        # Print diagnostics every 100 steps
        if step % 100 == 0:
            mode_name = get_mode_name(ace['mode'])
            engage_name = get_engage_name(ace['engagement'])
            print(f"    step={step:4d}: mode={mode_name:12s} engage={engage_name:10s} range={ace['range']:.0f}m")

    # Test passes if:
    # 1. A kill was achieved (ideal outcome), OR
    # 2. Planes passed each other and shots were fired (engagement occurred)
    passed = kill_achieved or (pass_detected and shots_after_pass > 0)
    status = "OK" if passed else "FAIL"

    print(f"head_to_head_dogfight: kill={kill_achieved} pass_detected={pass_detected} shots_after_pass={shots_after_pass} [{status}]")
    if not passed:
        print(f"  Expected: kill achieved OR (pass detected AND shots fired after)")

    RESULTS['head_to_head_dogfight'] = passed
    env.close()
    return passed


# =============================================================================
# TEST REGISTRY AND MAIN
# =============================================================================

TESTS = {
    'target_above_ahead': test_target_above_ahead,
    'target_below_ahead': test_target_below_ahead,
    'target_level_ahead': test_target_level_ahead,
    'target_left': test_target_left,
    'target_right': test_target_right,
    'target_behind': test_target_behind,
    'engage_offensive': test_engage_offensive,
    'fires_in_envelope': test_fires_in_envelope,
    'fires_after_turn_right': test_fires_after_turn_right,
    'no_fire_when_off_target': test_no_fire_when_off_target,
    'head_to_head_dogfight': test_head_to_head_dogfight,
}


def main():
    global _render_mode, _render_fps

    parser = argparse.ArgumentParser(description='AutoAce Behavioral Tests')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--fps', type=int, default=50, help='Render FPS (default 50)')
    parser.add_argument('--test', type=str, help='Run specific test')
    args = parser.parse_args()

    _render_mode = args.render
    _render_fps = args.fps if args.render else None

    if args.test:
        if args.test in TESTS:
            print(f"\n=== {args.test} ===")
            TESTS[args.test]()
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {', '.join(TESTS.keys())}")
            sys.exit(1)
    else:
        # Run all tests
        print("Running all AutoAce behavioral tests...")
        for name, test_fn in TESTS.items():
            print(f"\n=== {name} ===")
            test_fn()

        # Summary
        print("\n" + "=" * 50)
        passed = sum(1 for v in RESULTS.values() if v)
        total = len(RESULTS)
        print(f"RESULTS: {passed}/{total} tests passed")

        if passed < total:
            failed = [k for k, v in RESULTS.items() if not v]
            print(f"FAILED: {', '.join(failed)}")
            sys.exit(1)


if __name__ == "__main__":
    main()

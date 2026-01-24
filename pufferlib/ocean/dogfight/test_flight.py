"""
Physics validation tests for dogfight environment.
Uses force_state() to set exact initial conditions for accurate measurements.

Run: python pufferlib/ocean/dogfight/test_flight.py
     python pufferlib/ocean/dogfight/test_flight.py --render  # with visualization
     python pufferlib/ocean/dogfight/test_flight.py --render --test pitch_direction  # single test

This is the main entry point that aggregates all test modules:
- test_flight_physics.py: Flight physics tests (speed, climb, turn, G-force)
- test_flight_obs_static.py: Static observation scheme tests
- test_flight_obs_dynamic.py: Dynamic maneuver observation tests
- test_flight_energy.py: Energy physics tests (conservation, bleed rates, E-M theory)

TODO - FLIGHT PHYSICS TESTS NEEDED:
=====================================
1. RUDDER-ONLY TURN TEST (HIGH PRIORITY)
   - Current MAX_YAW_RATE = 1.5 rad/s (86 deg/s) is WAY too high
   - P-51D rudder should give ~5-15 deg/s yaw rate max, with significant sideslip
   - Test: wings level, full rudder, measure actual yaw rate and heading change
   - Compare against P-51D flight test data (see P51d_REFERENCE_DATA.md)
   - Expected: rudder alone should NOT be effective for turning - need bank

2. COORDINATED TURN TEST
   - Bank to 30, 45, 60 deg and measure sustained turn rate
   - P-51D should get ~17.5 deg/s at max sustained (corner velocity)
   - Verify turn rate vs bank angle relationship

3. ROLL RATE TEST
   - Full aileron deflection, measure time to roll 90 and 360 deg
   - P-51D: ~90-100 deg/s roll rate at 300 mph

4. PITCH AUTHORITY TEST
   - Full elevator, measure pitch rate and G-loading
   - Should be speed-dependent (less authority at low speed)
"""

from test_flight_base import (
    get_args, get_render_mode,
    RESULTS,
    P51D_MAX_SPEED, P51D_STALL_SPEED, P51D_CLIMB_RATE,
)

# Import test registries from each module
from test_flight_physics import TESTS as PHYSICS_TESTS
from test_flight_obs_static import TESTS as OBS_STATIC_TESTS
from test_flight_obs_dynamic import TESTS as OBS_DYNAMIC_TESTS
from test_flight_energy import TESTS as ENERGY_TESTS

# Aggregate all tests into a single registry
TESTS = {
    **PHYSICS_TESTS,
    **OBS_STATIC_TESTS,
    **OBS_DYNAMIC_TESTS,
    **ENERGY_TESTS,
}


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
    print(f"| turn_rate      | {fmt('turn_rate'):>6} | 5.6 deg/s (45 deg bank) |")
    print(f"| rudder_yaw     | {fmt('rudder_yaw_rate'):>6} | 5-15 deg/s (wings lvl) |")
    print(f"| pitch_dir      | {fmt('pitch_direction'):>6} | DOWN (+elev) |")
    print(f"| roll_works     | {fmt('roll_works'):>6} | YES |")


if __name__ == "__main__":
    args = get_args()

    print("P-51D Physics Validation Tests")
    print("=" * 60)

    if args.test:
        # Run single test
        if args.test in TESTS:
            print(f"Running single test: {args.test}")
            if get_render_mode():
                print("Rendering enabled - press ESC to exit")
            print("=" * 60)
            TESTS[args.test]()
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {', '.join(sorted(TESTS.keys()))}")
    else:
        # Run all tests
        print("Using force_state() for precise initial conditions")
        if get_render_mode():
            print("Rendering enabled - press ESC to exit")
        print("=" * 60)
        for test_func in TESTS.values():
            test_func()
        print_summary()

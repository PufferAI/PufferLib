"""SKIP stub for ocean/dogfight/test_selfplay.py (3.0 source).

The 3.0 tests in pufferlib/ocean/dogfight/test_selfplay.py exercise the
opponent-pool / training-loop scaffolding around `train_dual_selfplay.py`:
pool persistence, opponent sampling modes, the env<->pool integration
callback, a short end-to-end training loop, and obs-scheme filtering when
sampling pool members. None of that is ported to 4.0 yet — the
`train_dual_selfplay.py` script is Phase 3 work that depends on a
self-play opponent module that hasn't landed.

This stub keeps the test inventory complete so that `run_all.sh` reports
the deferred suite by name. Each test prints [SKIP] and main() returns 0.

To un-skip, port `train_dual_selfplay.py` (or its 4.0 equivalent) and the
opponent-pool module, then re-implement these checks against them.

Tests covered (1:1 with the 3.0 file):
  - test_pool_creation_and_persistence : on-disk pool format, save/load
  - test_pool_selection_modes          : random vs latest vs uniform sampling
  - test_env_with_pool_integration     : env consumes opponent from pool
  - test_callback_wiring               : training-loop callback updates pool
  - test_short_training_loop           : short end-to-end self-play loop
  - test_obs_scheme_filtering          : pool filters out wrong-scheme members
"""
import sys

SKIP_MSG = "[SKIP - needs train_dual_selfplay.py port]"


def test_pool_creation_and_persistence(failures):
    print(f"  test_pool_creation_and_persistence  {SKIP_MSG}")


def test_pool_selection_modes(failures):
    print(f"  test_pool_selection_modes           {SKIP_MSG}")


def test_env_with_pool_integration(failures):
    print(f"  test_env_with_pool_integration      {SKIP_MSG}")


def test_callback_wiring(failures):
    print(f"  test_callback_wiring                {SKIP_MSG}")


def test_short_training_loop(failures):
    print(f"  test_short_training_loop            {SKIP_MSG}")


def test_obs_scheme_filtering(failures):
    print(f"  test_obs_scheme_filtering           {SKIP_MSG}")


TESTS = [
    test_pool_creation_and_persistence,
    test_pool_selection_modes,
    test_env_with_pool_integration,
    test_callback_wiring,
    test_short_training_loop,
    test_obs_scheme_filtering,
]


def main():
    failures = []
    for t in TESTS:
        t(failures)
    print(f"\ntest_selfplay: 0/{len(TESTS)} run, {len(TESTS)} skipped (deferred)")
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""SKIP stub for ocean/dogfight/test_self_play.py (3.0 source).

The 3.0 tests in pufferlib/ocean/dogfight/test_self_play.py exercise
opponent-checkpoint loading via the `Dogfight` Python wrapper class
(`from pufferlib.ocean.dogfight.dogfight import Dogfight`) and require
trained .pt files glob'd from `experiments/`. Both prerequisites are
deferred indefinitely in 4.0:

  - The per-env Python wrapper was removed in PufferLib 4.0; envs are
    now binding-driven (`pufferlib/ocean/env_binding.h`). Whether to
    re-introduce a `dogfight.py` wrapper or write 4.0-native equivalents
    is a Phase 3 decision.
  - The training scaffolding (`train_dual_selfplay.py`) that produces
    those checkpoints has not been ported yet.

This stub keeps the test inventory complete so that `run_all.sh` shows
the deferred suite by name. Each test prints [SKIP] with the reason and
main() returns 0 (skips do not fail the suite).

To un-skip, port the dogfight Python wrapper (or write a 4.0 equivalent)
and re-implement these checks against it.

Tests covered (1:1 with the 3.0 file):
  - test_no_checkpoint        : env runs with no opponent policy (autopilot)
  - test_load_checkpoint      : opponent .pt loads as frozen policy
  - test_opponent_uses_policy : opponent actions come from loaded policy
  - test_obs_scheme_mismatch  : env / checkpoint scheme mismatch raises
  - test_device_option        : --device cpu/cuda routes the policy correctly
"""
import sys

SKIP_MSG = "[SKIP - needs dogfight.py wrapper port]"


def test_no_checkpoint(failures):
    print(f"  test_no_checkpoint                 {SKIP_MSG}")


def test_load_checkpoint(failures):
    print(f"  test_load_checkpoint               {SKIP_MSG}")


def test_opponent_uses_policy(failures):
    print(f"  test_opponent_uses_policy          {SKIP_MSG}")


def test_obs_scheme_mismatch(failures):
    print(f"  test_obs_scheme_mismatch           {SKIP_MSG}")


def test_device_option(failures):
    print(f"  test_device_option                 {SKIP_MSG}")


TESTS = [
    test_no_checkpoint,
    test_load_checkpoint,
    test_opponent_uses_policy,
    test_obs_scheme_mismatch,
    test_device_option,
]


def main():
    failures = []
    for t in TESTS:
        t(failures)
    print(f"\ntest_self_play: 0/{len(TESTS)} run, {len(TESTS)} skipped (deferred)")
    return 0


if __name__ == '__main__':
    sys.exit(main())

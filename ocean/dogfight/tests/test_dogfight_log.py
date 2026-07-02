"""Tests for ocean/dogfight/dogfight_log.py.

Run directly: python ocean/dogfight/tests/test_dogfight_log.py
Run via suite: bash ocean/dogfight/tests/run_all.sh
Exits 0 on pass, 1 on any failure.
"""
import os
import re
import sys
import tempfile

# Make sibling modules importable (ocean/dogfight/ is not a Python package in 4.0).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dogfight_log import init_log, log, logger


def _reset_logger():
    """Drop any FileHandlers between tests so each gets a fresh path."""
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def test_named_log_path(failures):
    _reset_logger()
    with tempfile.TemporaryDirectory() as d:
        path = init_log(d, 'unit_named')
        if not os.path.exists(path):
            failures.append(f"named log file not created at {path}")
        if not os.path.basename(path).startswith('unit_named_'):
            failures.append(f"named log filename missing run_name prefix: {path}")
        if not re.search(r'\d{4}-\d{2}-\d{2}_\d{6}\.log$', path):
            failures.append(f"named log filename missing timestamp: {path}")


def test_default_log_path(failures):
    _reset_logger()
    with tempfile.TemporaryDirectory() as d:
        path = init_log(d)
        if not os.path.basename(path).startswith('dogfight_'):
            failures.append(f"unnamed log should default to dogfight_ prefix: {path}")


def test_creates_nested_dir(failures):
    _reset_logger()
    with tempfile.TemporaryDirectory() as d:
        nested = os.path.join(d, 'a', 'b', 'c')
        init_log(nested, 'nested')
        if not os.path.isdir(nested):
            failures.append(f"init_log did not create nested dir {nested}")


def test_structured_lines_written(failures):
    _reset_logger()
    with tempfile.TemporaryDirectory() as d:
        path = init_log(d, 'wrote')
        log('[ROUND] num=42 event=start')
        log('[RATING] player=foo rating=1234')
        log('[ERROR] something broke')
        for h in list(logger.handlers):
            h.flush()
        with open(path) as f:
            content = f.read()
        for tag in ('[ROUND]', '[RATING]', '[ERROR]', 'log_started'):
            if tag not in content:
                failures.append(f"missing tag {tag} in log content")
        if not re.search(r'^\d{2}:\d{2}:\d{2} ', content, re.MULTILINE):
            failures.append("log lines missing HH:MM:SS timestamp prefix")


TESTS = [
    test_named_log_path,
    test_default_log_path,
    test_creates_nested_dir,
    test_structured_lines_written,
]


def main():
    failures = []
    for t in TESTS:
        before = len(failures)
        try:
            t(failures)
        except Exception as e:
            failures.append(f"{t.__name__}: {type(e).__name__}: {e}")
        status = 'OK' if len(failures) == before else 'FAIL'
        print(f"  {t.__name__:35s} [{status}]")
    _reset_logger()
    print(f"\ndogfight_log: {len(TESTS) - len([f for f in failures])}/{len(TESTS)} passed")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

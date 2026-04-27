import argparse
import os
import time
from types import SimpleNamespace

from craftax_parity import run


STRESS_CASES = [
    {
        "name": "mixed-wide",
        "seeds": 64,
        "steps": 10000,
        "policy": "mixed",
        "action_seed": 0,
    },
    {
        "name": "descend-boss-target",
        "seeds": 16,
        "steps": 30000,
        "policy": "descend",
        "action_seed": 1,
    },
    {
        "name": "suicide-terminal-target",
        "seeds": 32,
        "steps": 5000,
        "policy": "suicide",
        "action_seed": 2,
    },
    {
        "name": "combat-projectile-xp",
        "seeds": 16,
        "steps": 5000,
        "policy": "combat",
        "action_seed": 3,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
    )
    args = parser.parse_args()

    started = time.monotonic()
    for case in STRESS_CASES:
        case_started = time.monotonic()
        print(
            "RUN craftax parity stress "
            f"name={case['name']} seeds={case['seeds']} steps={case['steps']} "
            f"policy={case['policy']} action_seed={case['action_seed']} "
            f"atol={args.atol:g}",
            flush=True,
        )
        status = run(
            SimpleNamespace(
                seeds=case["seeds"],
                seed_start=args.seed_start,
                steps=case["steps"],
                action_seed=case["action_seed"],
                atol=args.atol,
                policy=case["policy"],
                reset_on_done=True,
                num_threads=args.num_threads,
            )
        )
        elapsed = time.monotonic() - case_started
        if status != 0:
            print(
                "FAIL craftax parity stress "
                f"name={case['name']} elapsed={elapsed:.1f}s",
                flush=True,
            )
            raise SystemExit(status)
        print(
            "PASS craftax parity stress case "
            f"name={case['name']} elapsed={elapsed:.1f}s",
            flush=True,
        )

    elapsed = time.monotonic() - started
    print(
        f"PASS craftax parity stress: cases={len(STRESS_CASES)} elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

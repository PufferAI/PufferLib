#!/usr/bin/env python3
"""Compare affine-lock reset throughput across initialization modes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_args(
    mode: int,
    start_depth: int,
    max_depth: int,
    total_agents: int,
    num_buffers: int,
    num_threads: int,
):
    from pufferlib.pufferl import load_config

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = load_config("affine_lock")
    finally:
        sys.argv = old_argv

    args["env"]["initialization_mode"] = mode
    args["env"]["start_depth"] = start_depth
    args["env"]["max_depth"] = max_depth
    args["vec"]["total_agents"] = total_agents
    args["vec"]["num_buffers"] = num_buffers
    args["vec"]["num_threads"] = num_threads
    return args


def bench_mode(mode: int, start_depth: int, max_depth: int, total_agents: int, num_buffers: int,
        num_threads: int, warmups: int, resets: int) -> tuple[float, float]:
    from pufferlib import _C

    args = load_args(
        mode, start_depth, max_depth, total_agents, num_buffers, num_threads)
    start_init = time.perf_counter()
    vec = _C.create_vec(args, 0)
    init_seconds = time.perf_counter() - start_init
    try:
        for _ in range(warmups):
            vec.reset()

        start = time.perf_counter()
        for _ in range(resets):
            vec.reset()
        elapsed = time.perf_counter() - start
    finally:
        vec.close()

    env_resets_per_second = total_agents * resets / elapsed
    return init_seconds, env_resets_per_second


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=4096)
    parser.add_argument("--buffers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--start-depth", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--resets", type=int, default=128)
    args = parser.parse_args()

    results = {}
    for mode in (3, 4):
        init_seconds, reset_rate = bench_mode(
            mode,
            args.start_depth,
            args.max_depth,
            args.agents,
            args.buffers,
            args.threads,
            args.warmups,
            args.resets,
        )
        results[mode] = reset_rate
        print(
            f"mode {mode}: init={init_seconds:.4f}s "
            f"reset_rate={reset_rate:,.0f} env-resets/s"
        )

    if results[3] > 0:
        print(f"mode4/mode3 reset_rate={results[4] / results[3]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

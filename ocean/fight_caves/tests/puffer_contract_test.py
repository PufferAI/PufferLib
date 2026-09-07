#!/usr/bin/env python3
"""Exercise Fight Caves through PufferLib's compiled CPU interface."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def fail(message: str) -> None:
    raise AssertionError(f"puffer_contract_test: {message}")


def main() -> int:
    try:
        from pufferlib import _C
    except ImportError as exc:
        raise RuntimeError(
            "Fight Caves Puffer backend is unavailable; run "
            "'./build.sh fight_caves --cpu' first"
        ) from exc
    if getattr(_C, "env_name", None) != "fight_caves":
        fail(f"backend was built for {getattr(_C, 'env_name', None)!r}")
    if getattr(_C, "gpu", None) != 0:
        fail("acceptance test requires the CPU backend")

    from pufferlib.pufferl import load_config

    previous_argv = sys.argv[:]
    try:
        sys.argv = ["puffer_contract_test"]
        args = load_config("fight_caves")
    finally:
        sys.argv = previous_argv
    args["vec"].update(total_agents=8, num_buffers=1, num_threads=1)

    vec = _C.create_vec(args, 0)
    try:
        if vec.obs_size != 320:
            fail(f"expected 320 observations, got {vec.obs_size}")
        if vec.num_atns != 3:
            fail(f"expected 3 action heads, got {vec.num_atns}")
        if list(vec.act_sizes) != [17, 9, 8]:
            fail(f"unexpected action dimensions: {list(vec.act_sizes)}")
        if vec.obs_dtype != "FloatTensor" or vec.obs_elem_size != 4:
            fail(
                f"unexpected observation type: {vec.obs_dtype}/{vec.obs_elem_size}"
            )

        obs_storage = (ctypes.c_float * (vec.total_agents * vec.obs_size)).from_address(
            vec.obs_ptr
        )
        reward_storage = (ctypes.c_float * vec.total_agents).from_address(
            vec.rewards_ptr
        )
        terminal_storage = (ctypes.c_float * vec.total_agents).from_address(
            vec.terminals_ptr
        )
        observations = np.ctypeslib.as_array(obs_storage).reshape(
            vec.total_agents, vec.obs_size
        )
        rewards = np.ctypeslib.as_array(reward_storage)
        terminals = np.ctypeslib.as_array(terminal_storage)

        vec.reset()
        if not np.isfinite(observations).all():
            fail("reset observations contain non-finite values")
        mask = observations[:, -34:]
        if not np.logical_or(mask == 0.0, mask == 1.0).all():
            fail("float action mask contains a value other than zero or one")
        for start, stop in ((0, 17), (17, 26), (26, 34)):
            if not (mask[:, start:stop].sum(axis=1) >= 1).all():
                fail("an action head has no legal action")

        actions = np.zeros((vec.total_agents, vec.num_atns), dtype=np.float32)
        terminal_count = 0
        for _ in range(6000):
            vec.cpu_step(actions.ctypes.data)
            if not np.isfinite(observations).all():
                fail("step observations contain non-finite values")
            if not np.isfinite(rewards).all():
                fail("rewards contain non-finite values")
            if not np.logical_or(terminals == 0.0, terminals == 1.0).all():
                fail("terminal buffer contains a value other than zero or one")
            terminal_count += int(terminals.sum())
            if terminal_count:
                break
        if terminal_count == 0:
            fail("no terminal/autoreset boundary was observed")
    finally:
        vec.close()

    print(f"puffer_contract_test: passed ({terminal_count} terminal transitions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

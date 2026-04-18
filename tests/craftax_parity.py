import argparse
import ctypes
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name


OBS_SIZE = 8268
NUM_ACTIONS = 43


def _preload_nccl():
    root = Path(__file__).resolve().parents[1]
    nccl = root / ".venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
    if nccl.exists():
        ctypes.CDLL(str(nccl), mode=ctypes.RTLD_GLOBAL)


def import_c_env():
    _preload_nccl()
    import pufferlib._C as cmod

    env_name = getattr(cmod, "env_name", None)
    if env_name != "craftax":
        raise RuntimeError(
            f"pufferlib._C is compiled for {env_name!r}, expected 'craftax'. "
            "Run: uv run --with pybind11 --with rich_argparse ./build.sh craftax"
        )
    return cmod


def float_view(ptr, count):
    array_t = ctypes.c_float * count
    return np.ctypeslib.as_array(array_t.from_address(ptr))


class JaxCraftaxBatch:
    def __init__(self, seeds):
        self.env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
        self.params = self.env.default_params
        self.rngs = []
        self.states = []
        self.obs = []
        for seed in seeds:
            rng = jax.random.PRNGKey(int(seed))
            rng, reset_key = jax.random.split(rng)
            obs, state = self.env.reset(reset_key, self.params)
            self.rngs.append(rng)
            self.states.append(state)
            self.obs.append(np.asarray(obs, dtype=np.float32).reshape(-1))

    def step(self, actions):
        obs_out = []
        rewards = []
        dones = []
        for i, action in enumerate(actions):
            rng, step_key = jax.random.split(self.rngs[i])
            obs, state, reward, done, _info = self.env.step(
                step_key, self.states[i], int(action), self.params
            )
            self.rngs[i] = rng
            self.states[i] = state
            obs_out.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            rewards.append(float(reward))
            dones.append(bool(done))
        self.obs = obs_out
        return (
            np.stack(obs_out, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
        )


def make_c_vec(cmod, num_envs, seed_offset):
    args = {
        "vec": {
            "total_agents": num_envs,
            "num_buffers": 1,
            "num_threads": 1,
        },
        "env": {
            "seed_offset": seed_offset,
        },
    }
    vec = cmod.create_vec(args, 0)
    if vec.obs_size != OBS_SIZE:
        raise RuntimeError(f"C obs_size={vec.obs_size}, expected {OBS_SIZE}")
    if vec.num_atns != 1:
        raise RuntimeError(f"C num_atns={vec.num_atns}, expected 1")
    if list(vec.act_sizes) != [NUM_ACTIONS]:
        raise RuntimeError(f"C act_sizes={vec.act_sizes}, expected [{NUM_ACTIONS}]")
    vec.reset()
    obs = float_view(vec.obs_ptr, num_envs * OBS_SIZE).reshape(num_envs, OBS_SIZE)
    rewards = float_view(vec.rewards_ptr, num_envs)
    terminals = float_view(vec.terminals_ptr, num_envs)
    return vec, obs, rewards, terminals


def action_plan(seeds, steps, action_seed):
    rng = np.random.default_rng(action_seed)
    return rng.integers(0, NUM_ACTIONS, size=(steps, len(seeds)), dtype=np.int32)


def first_obs_diff(ref, got, atol):
    diff = np.abs(ref - got)
    idx = int(np.argmax(diff))
    max_diff = float(diff[idx])
    if max_diff <= atol:
        return None
    return idx, max_diff, float(ref[idx]), float(got[idx])


def section_for_index(idx):
    map_size = 9 * 11 * 37
    item_size = 9 * 11 * 5
    mob_size = 9 * 11 * 5 * 8
    light_size = 9 * 11
    if idx < map_size:
        return "map_one_hot"
    idx -= map_size
    if idx < item_size:
        return "item_one_hot"
    idx -= item_size
    if idx < mob_size:
        return "mob_one_hot"
    idx -= mob_size
    if idx < light_size:
        return "light"
    return "inventory"


def compare_reset(ref_obs, c_obs, seeds, atol):
    for env_i, seed in enumerate(seeds):
        diff = first_obs_diff(ref_obs[env_i], c_obs[env_i], atol)
        if diff is not None:
            idx, max_diff, ref_value, c_value = diff
            print(
                "RESET DIVERGENCE "
                f"seed={seed} obs_index={idx} section={section_for_index(idx)} "
                f"abs_diff={max_diff:.8g} jax={ref_value:.8g} c={c_value:.8g}"
            )
            return False
    return True


def run(args):
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")

    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int64)
    actions = action_plan(seeds, args.steps, args.action_seed)

    cmod = import_c_env()
    ref = JaxCraftaxBatch(seeds)
    ref_obs = np.stack(ref.obs, axis=0)

    vec, c_obs, c_rewards, c_terminals = make_c_vec(cmod, len(seeds), int(seeds[0]))
    try:
        if not compare_reset(ref_obs, c_obs.copy(), seeds, args.atol):
            return 1

        action_buf = np.zeros((len(seeds), 1), dtype=np.float32)
        for step in range(args.steps):
            step_actions = actions[step]
            action_buf[:, 0] = step_actions.astype(np.float32)

            ref_obs, ref_rewards, ref_dones = ref.step(step_actions)
            vec.cpu_step(action_buf.ctypes.data)

            c_obs_snapshot = c_obs.copy()
            c_rewards_snapshot = c_rewards.copy()
            c_dones_snapshot = c_terminals.copy().astype(bool)

            for env_i, seed in enumerate(seeds):
                reward_diff = abs(float(ref_rewards[env_i]) - float(c_rewards_snapshot[env_i]))
                done_match = bool(ref_dones[env_i]) == bool(c_dones_snapshot[env_i])
                obs_diff = first_obs_diff(ref_obs[env_i], c_obs_snapshot[env_i], args.atol)
                if reward_diff > args.atol or not done_match or obs_diff is not None:
                    print(
                        "STEP DIVERGENCE "
                        f"seed={seed} step={step} action={int(step_actions[env_i])}"
                    )
                    print(
                        f"reward: jax={float(ref_rewards[env_i]):.8g} "
                        f"c={float(c_rewards_snapshot[env_i]):.8g} "
                        f"abs_diff={reward_diff:.8g}"
                    )
                    print(
                        f"done: jax={bool(ref_dones[env_i])} "
                        f"c={bool(c_dones_snapshot[env_i])}"
                    )
                    if obs_diff is None:
                        print("obs: ok")
                    else:
                        idx, max_diff, ref_value, c_value = obs_diff
                        print(
                            "obs: "
                            f"index={idx} section={section_for_index(idx)} "
                            f"abs_diff={max_diff:.8g} "
                            f"jax={ref_value:.8g} c={c_value:.8g}"
                        )
                    return 1

        print(
            f"PASS craftax parity: seeds={args.seeds} steps={args.steps} "
            f"atol={args.atol:g} action_seed={args.action_seed}"
        )
        return 0
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import configparser
import ctypes
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

EXPECTED_MY_LOG_KEYS = [
    "perf",
    "score",
    "solve_rate",
    "max_depth_solve",
    "episode_return",
    "episode_length",
    "timeout_rate",
    "min_win_moves",
    "solved_min_win_moves",
    "conditional_solve_steps",
    "conditional_solve_efficiency",
    "d6_solve_rate",
    "d8_solve_rate",
    "d16_solve_rate",
    "n",
]


def parse_int(value):
    return int(value.replace("_", ""))


def parse_float(value):
    return float(value.replace("_", ""))


def assert_sweep_mean(config, section, expected):
    assert parse_float(config[section]["mean"]) == expected


def check_config():
    config = configparser.ConfigParser()
    config.read(ROOT / "config" / "default.ini")
    config.read(ROOT / "config" / "affine_lock.ini")

    assert config["base"]["env_name"] == "affine_lock"
    assert parse_int(config["vec"]["total_agents"]) == 4096
    assert parse_int(config["vec"]["num_buffers"]) == 2
    assert parse_int(config["vec"]["num_threads"]) == 16
    assert parse_int(config["policy"]["hidden_size"]) == 256
    assert parse_int(config["policy"]["num_layers"]) == 3
    assert parse_int(config["env"]["seed"]) == 42
    assert parse_int(config["env"]["start_depth"]) == 2
    assert parse_int(config["env"]["max_depth"]) == 16
    assert parse_int(config["env"]["perf_weighting"]) == 1
    assert parse_int(config["train"]["total_timesteps"]) == 200_000_000
    assert parse_int(config["train"]["horizon"]) == 64
    assert parse_int(config["train"]["minibatch_size"]) == 8192
    assert parse_float(config["train"]["learning_rate"]) == 0.012
    assert parse_float(config["train"]["ent_coef"]) == 0.2
    assert parse_float(config["train"]["gamma"]) == 0.8
    assert parse_float(config["train"]["gae_lambda"]) == 0.995
    assert parse_float(config["train"]["replay_ratio"]) == 3.0
    assert parse_float(config["train"]["clip_coef"]) == 0.83
    assert parse_float(config["train"]["vf_coef"]) == 4.75
    assert parse_float(config["train"]["vf_clip_coef"]) == 0.8
    assert parse_float(config["train"]["max_grad_norm"]) == 3.0
    assert parse_float(config["train"]["vtrace_rho_clip"]) == 1.4
    assert parse_float(config["train"]["vtrace_c_clip"]) == 3.75
    assert "prio_alpha" not in config["train"]
    assert "prio_beta0" not in config["train"]
    assert_sweep_mean(config, "sweep.train.total_timesteps", 200_000_000.0)
    assert_sweep_mean(config, "sweep.vec.total_agents", 4096.0)
    assert_sweep_mean(config, "sweep.vec.num_buffers", 2.0)
    assert_sweep_mean(config, "sweep.policy.hidden_size", 256.0)
    assert_sweep_mean(config, "sweep.policy.num_layers", 3.0)
    assert_sweep_mean(config, "sweep.train.horizon", 64.0)
    assert_sweep_mean(config, "sweep.train.minibatch_size", 8192.0)
    assert_sweep_mean(config, "sweep.train.learning_rate", 0.012)
    assert_sweep_mean(config, "sweep.train.ent_coef", 0.2)
    assert_sweep_mean(config, "sweep.train.gamma", 0.8)
    assert_sweep_mean(config, "sweep.train.gae_lambda", 0.995)
    assert_sweep_mean(config, "sweep.train.replay_ratio", 3.0)
    assert_sweep_mean(config, "sweep.train.clip_coef", 0.83)
    assert_sweep_mean(config, "sweep.train.vf_coef", 4.75)
    assert_sweep_mean(config, "sweep.train.vf_clip_coef", 0.8)
    assert_sweep_mean(config, "sweep.train.max_grad_norm", 3.0)
    assert "sweep.train.prio_alpha" not in config
    assert "sweep.train.prio_beta0" not in config
    assert config["sweep"]["metric"] == "perf"
    assert config["sweep"]["goal"] == "maximize"

    sweep_ts = config["sweep.train.total_timesteps"]
    min_steps = parse_int(sweep_ts["min"])
    max_steps = parse_int(sweep_ts["max"])
    assert min_steps == 100_000_000
    assert max_steps == 200_000_000

    assert parse_int(config["sweep.train.horizon"]["min"]) == 32
    assert parse_int(config["sweep.train.horizon"]["max"]) == 128
    assert parse_int(config["sweep.policy.hidden_size"]["min"]) == 64
    assert parse_int(config["sweep.policy.hidden_size"]["max"]) == 512
    assert float(config["sweep.policy.num_layers"]["min"]) == 1.0
    assert float(config["sweep.policy.num_layers"]["max"]) == 4.0
    assert parse_int(config["sweep.vec.total_agents"]["min"]) == 4096
    assert parse_int(config["sweep.vec.total_agents"]["max"]) == 16_384
    assert float(config["sweep.vec.num_buffers"]["min"]) == 1.0
    assert float(config["sweep.vec.num_buffers"]["max"]) == 4.0
    assert parse_int(config["sweep.train.minibatch_size"]["min"]) == 8192
    assert parse_int(config["sweep.train.minibatch_size"]["max"]) == 131_072
    assert float(config["sweep.train.replay_ratio"]["min"]) == 1.0
    assert float(config["sweep.train.replay_ratio"]["max"]) == 4.0
    assert float(config["sweep.train.vf_clip_coef"]["min"]) == 0.001
    assert float(config["sweep.train.vf_clip_coef"]["max"]) == 5.0
    assert float(config["sweep.train.vf_coef"]["min"]) == 0.1
    assert float(config["sweep.train.vf_coef"]["max"]) == 8.0

    min_batch_size = (
        parse_int(config["sweep.vec.total_agents"]["min"])
        * parse_int(config["sweep.train.horizon"]["min"])
    )
    max_minibatch_size = parse_int(config["sweep.train.minibatch_size"]["max"])
    min_replay_ratio = float(config["sweep.train.replay_ratio"]["min"])
    assert min_replay_ratio * min_batch_size >= max_minibatch_size


def check_header_text():
    header = (ROOT / "ocean" / "affine_lock" / "affine_lock.h").read_text()
    assert "#define OBS_SIZE (TIMER_INDEX + 1)" in header
    assert "#define ACT_SIZES {NUM_ACTIONS}" in header
    assert "#define NUM_ATNS 1" in header
    assert "typedef" in header and "obs_t" in header
    assert "void puf_init(" in header
    assert "void puf_reset(" in header
    assert "void puf_step(" in header
    assert "void puf_log(" in header
    assert "void puf_close(" in header
    assert "void puf_render(" in header
    assert "env->agents[" in header

    log_keys = re.findall(r'dict_set\(out,\s*"([^"]+)"', header)
    assert log_keys == EXPECTED_MY_LOG_KEYS
    assert len(log_keys) <= 32


def float_buffer(ptr, count):
    return (ctypes.c_float * count).from_address(ptr)


def check_backend_metadata():
    from pufferlib import _C
    from pufferlib.pufferl import load_config

    assert _C.env_name == "affine_lock"
    assert _C.gpu == 1

    def load_affine_args(extra_argv):
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], *extra_argv]
            return load_config("affine_lock")
        finally:
            sys.argv = old_argv

    base_args = load_affine_args([])
    assert base_args["env_name"] == "affine_lock"
    assert base_args["vec"]["total_agents"] == 4096
    assert base_args["vec"]["num_buffers"] == 2
    assert base_args["policy"]["hidden_size"] == 256
    assert base_args["policy"]["num_layers"] == 3
    assert base_args["train"]["horizon"] == 64
    assert base_args["train"]["minibatch_size"] == 8192
    assert base_args["train"]["learning_rate"] == 0.012
    assert base_args["train"]["ent_coef"] == 0.2
    assert base_args["train"]["gamma"] == 0.8
    assert base_args["train"]["gae_lambda"] == 0.995
    assert base_args["train"]["replay_ratio"] == 3.0
    assert base_args["train"]["clip_coef"] == 0.83
    assert base_args["train"]["vf_coef"] == 4.75
    assert base_args["train"]["vf_clip_coef"] == 0.8
    assert base_args["train"]["max_grad_norm"] == 3.0
    assert base_args["train"]["vtrace_rho_clip"] == 1.4
    assert base_args["train"]["vtrace_c_clip"] == 3.75
    assert "prio_alpha" not in base_args["train"]
    assert "prio_beta0" not in base_args["train"]

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = load_config("affine_lock")
    finally:
        sys.argv = old_argv
    args["vec"]["total_agents"] = 2
    args["vec"]["num_buffers"] = 1
    vec = _C.create_vec(args, 0)
    try:
        assert vec.obs_size == 33
        assert vec.obs_dtype == "FloatTensor"
        assert list(vec.act_sizes) == [8]

        obs = float_buffer(vec.obs_ptr, vec.total_agents * vec.obs_size)
        rewards = float_buffer(vec.rewards_ptr, vec.total_agents)
        terminals = float_buffer(vec.terminals_ptr, vec.total_agents)

        vec.reset()
        assert list(rewards) == [0.0, 0.0]
        assert list(terminals) == [0.0, 0.0]
        for env_id in range(vec.total_agents):
            timer = obs[env_id * vec.obs_size + 32]
            assert timer == 0.0

        actions = (ctypes.c_float * vec.total_agents)(8.0, 8.0)
        vec.cpu_step(ctypes.addressof(actions))
        assert list(rewards) == [-1.0, -1.0]
        assert list(terminals) == [1.0, 1.0]

        logs = vec.log()
        assert logs["n"] == 2.0
        assert logs["timeout_rate"] == 1.0
        assert logs["solve_rate"] == 0.0
        assert logs["episode_length"] == 1.0
        assert logs["episode_return"] == -1.0
        assert "perf" in logs
        assert "min_win_moves" in logs
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-backend", action="store_true")
    args = parser.parse_args()

    check_config()
    check_header_text()
    if args.require_backend:
        check_backend_metadata()


if __name__ == "__main__":
    main()

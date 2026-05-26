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
    "solve_rate",
    "max_depth_solve",
    "episode_return",
    "episode_length",
    "timeout_rate",
    "invalid_rate",
    "min_win_moves",
    "solved_min_win_moves",
    "conditional_solve_steps",
    "conditional_solve_efficiency",
    "depth_2_solve_rate",
    "depth_4_solve_rate",
    "depth_8_solve_rate",
    "depth_16_solve_rate",
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
    assert parse_int(config["env"]["initialization_mode"]) == 2
    assert "debug_log_level" not in config["env"]
    assert "short_solve_audit_enabled" not in config["env"]
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
    assert parse_float(config["train"]["beta1"]) == 0.5
    assert parse_float(config["train"]["beta2"]) == 0.9915
    assert parse_float(config["train"]["eps"]) == 0.0001
    assert parse_float(config["train"]["vtrace_rho_clip"]) == 1.4
    assert parse_float(config["train"]["vtrace_c_clip"]) == 3.75
    assert parse_float(config["train"]["prio_alpha"]) == 0.055
    assert parse_float(config["train"]["prio_beta0"]) == 0.161
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
    assert_sweep_mean(config, "sweep.train.beta1", 0.5)
    assert_sweep_mean(config, "sweep.train.beta2", 0.9915)
    assert_sweep_mean(config, "sweep.train.eps", 0.0001)
    assert_sweep_mean(config, "sweep.train.vtrace_rho_clip", 1.4)
    assert_sweep_mean(config, "sweep.train.vtrace_c_clip", 3.75)
    assert_sweep_mean(config, "sweep.train.prio_alpha", 0.055)
    assert_sweep_mean(config, "sweep.train.prio_beta0", 0.161)
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
    assert parse_int(config["sweep.policy.hidden_size"]["max"]) == 256
    assert float(config["sweep.policy.num_layers"]["min"]) == 1.0
    assert float(config["sweep.policy.num_layers"]["max"]) == 4.0
    assert parse_int(config["sweep.vec.total_agents"]["min"]) == 4096
    assert parse_int(config["sweep.vec.total_agents"]["max"]) == 16_384
    assert float(config["sweep.vec.num_buffers"]["min"]) == 1.0
    assert float(config["sweep.vec.num_buffers"]["max"]) == 4.0
    assert parse_int(config["sweep.train.minibatch_size"]["min"]) == 8192
    assert parse_int(config["sweep.train.minibatch_size"]["max"]) == 131_072
    assert float(config["sweep.train.replay_ratio"]["min"]) == 1.0
    assert float(config["sweep.train.replay_ratio"]["max"]) == 3.0

    min_batch_size = (
        parse_int(config["sweep.vec.total_agents"]["min"])
        * parse_int(config["sweep.train.horizon"]["min"])
    )
    max_minibatch_size = parse_int(config["sweep.train.minibatch_size"]["max"])
    min_replay_ratio = float(config["sweep.train.replay_ratio"]["min"])
    assert min_replay_ratio * min_batch_size >= max_minibatch_size

    assert not (
        ROOT / "config" / "profiles" / "affine_lock_highthroughput.ini"
    ).exists()


def check_binding_text():
    header = (ROOT / "ocean" / "affine_lock" / "affine_lock.h").read_text()
    assert "#define AFFINE_LOCK_MAX_SOLUTION_DEPTH 16" in header
    assert "AFFINE_LOCK_INIT_EXACT_DISTANCE = 1" in header
    assert "AFFINE_LOCK_INIT_VISIBLE_TARGET_TABLE = 2" in header
    assert "AFFINE_LOCK_INIT_SCRAMBLE" not in header
    assert "AFFINE_LOCK_INIT_RANDOM" not in header
    assert "AFFINE_LOCK_INIT_WCA_RANDOM_STATE" not in header
    assert "debug_log" not in header
    assert "short_solve_audit" not in header

    env_api_order = [
        "affine_lock_init_env",
        "affine_lock_add_log",
        "affine_lock_compute_observations",
        "compute_observations",
        "c_reset",
        "affine_lock_advance_curriculum",
        "affine_lock_finish_episode",
        "c_step",
        "c_close",
        "c_render",
    ]
    env_api_positions = {}
    for name in env_api_order:
        match = re.search(rf"\b{name}\s*\(", header)
        assert match is not None, name
        env_api_positions[name] = match.start()
    assert [env_api_positions[name] for name in env_api_order] == sorted(
        env_api_positions.values()
    )

    binding = (ROOT / "ocean" / "affine_lock" / "binding.c").read_text()
    assert "#define OBS_SIZE AFFINE_LOCK_OBS_SIZE" in binding
    assert "#define ACT_SIZES {AFFINE_LOCK_NUM_ACTIONS}" in binding
    assert "#define OBS_TENSOR_T FloatTensor" in binding
    assert "#define MY_THREAD_CLOSE" not in binding
    assert "my_thread_close" not in binding
    assert 'dict_get(env_kwargs, "rank")' not in binding
    assert "debug_log" not in binding
    assert "short_solve_audit" not in binding

    log_keys = re.findall(r'dict_set\(out,\s*"([^"]+)"', binding)
    assert log_keys == EXPECTED_MY_LOG_KEYS
    assert len(log_keys) + 1 <= 32  # static_vec_log appends "n".

    log_struct = re.search(r"typedef struct Log \{(?P<body>.*?)\} Log;", header, re.S)
    assert log_struct is not None
    log_fields = re.findall(r"^\s*float\s+([a-zA-Z0-9_]+);", log_struct.group("body"), re.M)
    derived_log_keys = {
        "conditional_solve_steps",
        "conditional_solve_efficiency",
        "min_win_moves",
        "solved_min_win_moves",
    }
    framework_log_fields = {"n"}
    renamed_log_fields = {"target_distance", "solved_target_distance"}
    internal_log_fields = {
        "score",
        "scramble_depth",
        "at_max_depth",
        "solve_steps",
        "start_mismatches",
        "final_mismatches",
        "one_action_target_rate",
        "two_action_target_rate",
        "short_solve_rate",
        "solve_efficiency",
        "reward_state_mismatch",
        "depth_2_rate",
        "depth_4_rate",
        "depth_8_rate",
        "depth_16_rate",
    }
    assert (
        set(log_fields)
        - framework_log_fields
        - renamed_log_fields
        - internal_log_fields
        <= set(log_keys) - derived_log_keys
    )


def check_shared_core_text():
    bindings = (ROOT / "src" / "bindings.cu").read_text()
    vecenv = (ROOT / "src" / "vecenv.h").read_text()

    assert 'dict_set(env_dict, "seed"' not in bindings
    assert 'dict_set(env_dict, "rank"' not in bindings
    assert "extra_capacity" not in bindings
    assert "MY_THREAD_CLOSE" not in vecenv
    assert "my_thread_close" not in vecenv


def float_buffer(ptr, count):
    return (ctypes.c_float * count).from_address(ptr)


def check_backend_metadata():
    from pufferlib import _C
    from pufferlib.pufferl import load_config, validate_config

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
    assert base_args["train"]["beta1"] == 0.5
    assert base_args["train"]["beta2"] == 0.9915
    assert base_args["train"]["eps"] == 0.0001
    assert base_args["train"]["vtrace_rho_clip"] == 1.4
    assert base_args["train"]["vtrace_c_clip"] == 3.75
    assert base_args["train"]["prio_alpha"] == 0.055
    assert base_args["train"]["prio_beta0"] == 0.161

    try:
        load_affine_args(["--config-profile", "affine_lock_highthroughput"])
        raise AssertionError("retired affine_lock_highthroughput profile loaded")
    except ValueError as exc:
        assert "No config profile affine_lock_highthroughput" in str(exc)

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
        assert logs["invalid_rate"] == 1.0
        assert logs["timeout_rate"] == 0.0
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
    check_binding_text()
    check_shared_core_text()
    if args.require_backend:
        check_backend_metadata()


if __name__ == "__main__":
    main()

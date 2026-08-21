#!/usr/bin/env python3
import glob
import json
import math
import os
import sys
import time
import copy
from collections import defaultdict

import rich
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pufferlib.pufferl
from pufferlib import _C
from pufferlib.torch_pufferl import PuffeRL, load_policy


PROFILE_DEFAULTS = {
    "fast": {
        "HIDDEN_SIZE": 128,
        "TOTAL_AGENTS": 256,
        "HORIZON": 32,
        "MINIBATCH_SIZE": 8192,
        "LEARNING_RATE": 0.005,
        "ENT_COEF": 0.1,
        "MASTERY_THRESHOLD": 0.95,
        "FINAL_MASTERY_THRESHOLD": 0.95,
    },
    "steady": {
        "HIDDEN_SIZE": 256,
        "TOTAL_AGENTS": 1024,
        "HORIZON": 128,
        "MINIBATCH_SIZE": 8192,
        "LEARNING_RATE": 0.002,
        "ENT_COEF": 0.001,
        "FIRST_STAGE_TIMESTEPS": 5_000_000,
        "FIRST_STAGE_MIN_TIMESTEPS": 500_000,
        "TIMESTEPS_PER_STAGE": 500_000,
        "MIN_TIMESTEPS_PER_STAGE": 500_000,
        "MASTERY_THRESHOLD": 0.95,
        "FINAL_MASTERY_THRESHOLD": 0.95,
    },
}


def curriculum_profile():
    profile = os.environ.get("CURRICULUM_PROFILE", "auto").strip().lower()
    if profile == "auto":
        n_qubits = int(os.environ.get("N_QUBITS", 3))
        return "fast" if n_qubits <= 3 else "steady"
    if profile not in PROFILE_DEFAULTS:
        names = ", ".join(["auto", *PROFILE_DEFAULTS])
        raise ValueError(f"CURRICULUM_PROFILE must be one of: {names}")
    return profile


def env_value(name, default):
    if name in os.environ:
        return os.environ[name]
    return PROFILE_DEFAULTS.get(curriculum_profile(), {}).get(name, default)


def env_int(name, default):
    return int(env_value(name, default))


def env_float(name, default):
    return float(env_value(name, default))


def env_str(name, default):
    return str(env_value(name, default))


def latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "clifford", "**", "*.bin")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getctime)


def stage_timesteps(difficulty, stage_idx=None):
    first_stage_override = env_value("FIRST_STAGE_TIMESTEPS", None)
    if stage_idx == 0 and first_stage_override is not None:
        return int(first_stage_override)

    override = env_value("TIMESTEPS_PER_STAGE", None)
    if override is not None:
        return int(override)
    if difficulty <= 3:
        return 8_000_000
    if difficulty <= 6:
        return 12_000_000
    return 16_000_000


def stage_min_timesteps(stage_idx):
    first_stage_override = env_value("FIRST_STAGE_MIN_TIMESTEPS", None)
    if stage_idx == 0 and first_stage_override is not None:
        return int(first_stage_override)
    return env_int("MIN_TIMESTEPS_PER_STAGE", 0)


def stage_max_steps(difficulty, success_step_stats=None):
    return stage_max_steps_with_source(difficulty, success_step_stats)[0]


def stage_max_steps_with_source(difficulty, success_step_stats=None):
    override = os.environ.get("MAX_STEPS")
    if override is not None:
        return int(override), f"MAX_STEPS={override}"

    min_steps = env_int("MIN_MAX_STEPS", 4)
    base_slack = env_float("MAX_STEPS_BASE_SLACK", 2)
    headroom = env_float("MAX_STEPS_HEADROOM", 0)
    max_steps = max(min_steps, math.ceil(difficulty + base_slack))
    source = f"ceil(difficulty + MAX_STEPS_BASE_SLACK)={max_steps}"
    if headroom > 0:
        headroom_steps = math.ceil(difficulty * headroom)
        if headroom_steps > max_steps:
            max_steps = headroom_steps
            source = f"ceil(difficulty * MAX_STEPS_HEADROOM)={max_steps}"

    stddevs = env_float("MAX_STEPS_STDDEVS", 0)
    stddevs_after = env_float("MAX_STEPS_STDDEVS_AFTER_DIFFICULTY", 0)
    if stddevs > 0 and difficulty >= stddevs_after and success_step_stats is not None:
        mean_steps, std_steps = success_step_stats
        stddev_steps = math.ceil(mean_steps + stddevs * std_steps)
        if stddev_steps > max_steps:
            max_steps = stddev_steps
            source = (
                f"prev_success_step_mean={mean_steps:.3f} + "
                f"{stddevs:g} * prev_success_step_std={std_steps:.3f} "
                f"=> {max_steps}"
            )

    force_after = env_float("FORCE_MAX_STEPS_AFTER_DIFFICULTY", -1)
    force_value = env_int("FORCE_MAX_STEPS_VALUE", 1000)
    if force_after >= 0 and difficulty >= force_after:
        return force_value, f"FORCE_MAX_STEPS_VALUE={force_value}"

    slack_after = env_float("MAX_STEPS_SLACK_AFTER_DIFFICULTY", 0)
    slack = env_int("MAX_STEPS_SLACK", 0) if difficulty >= slack_after else 0
    if slack:
        source = f"{source} + MAX_STEPS_SLACK={slack}"
    return max_steps + slack, source


def format_difficulty(difficulty):
    return f"{difficulty:g}"


def reached_mastery(perf, stop_threshold, stage_steps, min_timesteps):
    if not math.isfinite(perf):
        return False
    if perf >= 1.0:
        return True
    return stage_steps >= min_timesteps and perf >= stop_threshold


def reached_threshold(perf, stop_threshold):
    return math.isfinite(perf) and perf >= stop_threshold


def curriculum_difficulties(max_difficulty, stride):
    if max_difficulty <= 0:
        return []
    if stride <= 0:
        raise ValueError("CURRICULUM_STRIDE must be > 0")

    difficulties = []
    current = 1.0
    while current <= max_difficulty + 1e-9:
        difficulties.append(round(current, 10))
        current += stride

    if difficulties and difficulties[-1] > max_difficulty:
        difficulties[-1] = max_difficulty
    elif not difficulties or difficulties[-1] < max_difficulty - 1e-9:
        difficulties.append(max_difficulty)

    return difficulties


def expected_actions(n_qubits, use_shortcut_gates):
    single_qubit_actions = 5 if use_shortcut_gates else 2
    return single_qubit_actions * n_qubits + n_qubits * (n_qubits - 1) // 2


def default_run_name(n_qubits, hidden_size, use_shortcut_gates=True):
    action_suffix = "" if use_shortcut_gates else "_hs_cz"
    return f"clifford_{n_qubits}q{action_suffix}_mlp{hidden_size}_long"


def build_args(difficulty, max_steps, total_timesteps, load_model_path=None):
    n_qubits = env_int("N_QUBITS", 3)
    use_shortcut_gates = env_int("USE_SHORTCUT_GATES", 1)
    hidden_size = env_int("HIDDEN_SIZE", 128)
    run_name = default_run_name(
        n_qubits,
        hidden_size,
        use_shortcut_gates=bool(use_shortcut_gates),
    )
    return {
        "env_name": "clifford",
        "rank": 0,
        "world_size": 1,
        "gpu_id": 0,
        "profile": False,
        "checkpoint_dir": env_str(
            "CHECKPOINT_DIR", os.path.join("checkpoints", run_name)
        ),
        "log_dir": env_str("LOG_DIR", os.path.join("logs", run_name)),
        "checkpoint_interval": 1,
        "eval_episodes": env_int("TOTAL_AGENTS", 256),
        "reset_state": True,
        "load_model_path": load_model_path,
        "load_enemy_model_path": None,
        "load_id": None,
        "wandb": False,
        "slowly": True,
        "render_mode": "auto",
        "vec": {
            "total_agents": env_int("TOTAL_AGENTS", 256),
            "num_buffers": 1,
            "num_threads": env_int("NUM_THREADS", 1),
        },
        "env": {
            "n_qubits": n_qubits,
            "difficulty": difficulty,
            "max_steps": max_steps,
            "single_qubit_cost": env_float("SINGLE_QUBIT_COST", 0.001),
            "cz_cost": env_float("CZ_COST", 0.1),
            "goal_bonus": env_float("GOAL_BONUS", 1.0),
            "failure_penalty": env_float("FAILURE_PENALTY", -1.0),
            "use_shortcut_gates": use_shortcut_gates,
            "seed": env_int("SEED", 1),
        },
        "policy": {
            "hidden_size": hidden_size,
            "num_layers": env_int("NUM_LAYERS", 2),
            "expansion_factor": 1,
        },
        "torch": {
            "network": env_str("NETWORK", "MLP"),
            "encoder": "DefaultEncoder",
            "decoder": "DefaultDecoder",
        },
        "train": {
            "gpus": 1,
            "seed": env_int("SEED", 1),
            "total_timesteps": total_timesteps,
            "learning_rate": env_float("LEARNING_RATE", 0.005),
            "anneal_lr": 0,
            "min_lr_ratio": 0.0,
            "gamma": env_float("GAMMA", 0.995),
            "gae_lambda": env_float("GAE_LAMBDA", 0.90),
            "replay_ratio": env_float("REPLAY_RATIO", 1.0),
            "clip_coef": env_float("CLIP_COEF", 0.2),
            "vf_coef": env_float("VF_COEF", 2.0),
            "vf_clip_coef": 0.2,
            "max_grad_norm": env_float("MAX_GRAD_NORM", 1.5),
            "ent_coef": env_float("ENT_COEF", 0.1),
            "beta1": 0.95,
            "beta2": 0.999,
            "eps": 1e-12,
            "minibatch_size": env_int("MINIBATCH_SIZE", 8192),
            "horizon": env_int("HORIZON", 32),
            "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0,
            "prio_alpha": 0.8,
            "prio_beta0": 0.2,
        },
        "sweep": {
            "metric": "score",
            "downsample": 5,
        },
    }


def flatten_logs(logs):
    return dict(pufferlib.pufferl.unroll_nested_dict(logs))


def save_metrics(log_dir, run_id, args, logs):
    os.makedirs(os.path.join(log_dir, "clifford"), exist_ok=True)
    metrics = defaultdict(list)
    for log in logs:
        for key, value in log.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                pass
            metrics[key].append(value)

    path = os.path.join(log_dir, "clifford", f"{run_id}.json")
    with open(path, "w") as f:
        json.dump({**args, "metrics": dict(metrics)}, f)


def save_checkpoint(pufferl, checkpoint_dir, run_id, global_step):
    directory = os.path.join(checkpoint_dir, "clifford", run_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{global_step:016d}.bin")
    pufferl.save_weights(path)
    return path


def create_pufferl(args, policy, optimizer):
    vec = _C.create_vec(args, 0)
    n_qubits = args["env"]["n_qubits"]
    expected_obs_size = (2 * n_qubits) ** 2
    expected_act_sizes = [
        expected_actions(n_qubits, args["env"].get("use_shortcut_gates", 0))
    ]
    if vec.obs_size != expected_obs_size or vec.act_sizes != expected_act_sizes:
        vec.close()
        raise RuntimeError(
            f"Expected a {n_qubits}-qubit Clifford build with "
            f"obs_size={expected_obs_size} and act_sizes={expected_act_sizes}; "
            f"got obs_size={vec.obs_size}, act_sizes={vec.act_sizes}"
        )

    if policy is None:
        policy = load_policy(args, vec)

    pufferl = PuffeRL(args, vec, policy, verbose=False)
    optimizer_name = str(env_str("OPTIMIZER", "adamw")).lower()
    if optimizer is not None:
        pufferl.optimizer = optimizer
        pufferl.optimizer.param_groups[0]["lr"] = args["train"]["learning_rate"]
    elif optimizer_name == "adamw":
        pufferl.optimizer = torch.optim.AdamW(  # type: ignore[assignment]
            pufferl.policy.parameters(),
            lr=args["train"]["learning_rate"],
            betas=(args["train"].get("beta1", 0.9), args["train"].get("beta2", 0.999)),
            eps=args["train"].get("eps", 1e-8),
        )
    return pufferl


def dashboard_args(args, global_step_offset):
    args = {**args, "train": {**args["train"]}}
    args["train"]["total_timesteps"] = (
        global_step_offset + args["train"]["total_timesteps"]
    )
    return args


def display_logs(
    logs,
    global_step_offset,
    local_step,
    epoch_offset,
    run_start_time,
    previous_logs=None,
):
    logs = flatten_logs(logs)
    logs["agent_steps"] = global_step_offset + local_step
    logs["epoch"] = epoch_offset + logs.get("epoch", 0)
    logs["uptime"] = time.time() - run_start_time
    if logs.get("SPS", 0) == 0 and previous_logs is not None:
        logs["SPS"] = previous_logs.get("SPS", 0)
    return logs


def train_stage(
    args,
    policy,
    optimizer,
    run_id,
    global_step_offset,
    stop_threshold,
    min_timesteps,
    run_start_time,
):
    pufferlib.pufferl.validate_config(args)
    pufferl = create_pufferl(args, policy, optimizer)
    model_size = pufferl.num_params()
    flat_logs = {}
    stage_logs = []
    best_mastery_perf = -math.inf
    best_mastery_step = 0
    best_policy_state = None
    best_optimizer_state = None
    panel_args = dashboard_args(args, global_step_offset)
    batch_size = args["vec"]["total_agents"] * args["train"]["horizon"]
    epoch_offset = global_step_offset // batch_size

    try:
        while pufferl.global_step < args["train"]["total_timesteps"]:
            pufferl.rollouts()
            rollout_perf = float(getattr(pufferl, "env_logs", {}).get("perf", math.nan))
            flat_logs = display_logs(
                pufferl.log(),
                global_step_offset,
                pufferl.global_step,
                epoch_offset,
                run_start_time,
            )
            stage_logs.append(flat_logs)
            pufferlib.pufferl.print_dashboard(panel_args, model_size, flat_logs)

            if (
                reached_threshold(rollout_perf, stop_threshold)
                and rollout_perf > best_mastery_perf
            ):
                best_mastery_perf = rollout_perf
                best_mastery_step = pufferl.global_step
                best_policy_state = copy.deepcopy(pufferl.policy.state_dict())
                best_optimizer_state = copy.deepcopy(pufferl.optimizer.state_dict())

            if reached_mastery(
                rollout_perf,
                stop_threshold,
                pufferl.global_step,
                min_timesteps,
            ):
                rich.print(
                    f"Early stop: env/perf={rollout_perf:.3f} reached mastery "
                    f"(threshold={stop_threshold:.3f}, min_timesteps={min_timesteps})"
                )
                path = save_checkpoint(
                    pufferl,
                    args["checkpoint_dir"],
                    run_id,
                    global_step_offset + pufferl.global_step,
                )
                return (
                    pufferl.policy,
                    pufferl.optimizer,
                    pufferl.global_step,
                    rollout_perf,
                    path,
                    stage_logs,
                    True,
                )

            if best_policy_state is not None and pufferl.global_step >= min_timesteps:
                pufferl.policy.load_state_dict(best_policy_state)
                pufferl.optimizer.load_state_dict(best_optimizer_state)
                rich.print(
                    f"Restoring best stage policy: env/perf={best_mastery_perf:.3f} "
                    f"at step={best_mastery_step} after min_timesteps={min_timesteps}"
                )
                path = save_checkpoint(
                    pufferl,
                    args["checkpoint_dir"],
                    run_id,
                    global_step_offset + pufferl.global_step,
                )
                return (
                    pufferl.policy,
                    pufferl.optimizer,
                    pufferl.global_step,
                    best_mastery_perf,
                    path,
                    stage_logs,
                    True,
                )

            pufferl.train()

        flat_logs = display_logs(
            pufferl.log(),
            global_step_offset,
            pufferl.global_step,
            epoch_offset,
            run_start_time,
            stage_logs[-1] if stage_logs else None,
        )
        stage_logs.append(flat_logs)
        pufferlib.pufferl.print_dashboard(panel_args, model_size, flat_logs)
        perf = float(flat_logs.get("env/perf", math.nan))
        path = save_checkpoint(
            pufferl,
            args["checkpoint_dir"],
            run_id,
            global_step_offset + pufferl.global_step,
        )
        mastered = reached_mastery(
            perf,
            stop_threshold,
            pufferl.global_step,
            min_timesteps,
        )
        if not mastered and best_policy_state is not None:
            pufferl.policy.load_state_dict(best_policy_state)
            pufferl.optimizer.load_state_dict(best_optimizer_state)
            perf = best_mastery_perf
            mastered = True
            path = save_checkpoint(
                pufferl,
                args["checkpoint_dir"],
                run_id,
                global_step_offset + pufferl.global_step,
            )
        return (
            pufferl.policy,
            pufferl.optimizer,
            pufferl.global_step,
            perf,
            path,
            stage_logs,
            mastered,
        )
    finally:
        pufferl.close()


def main():
    if getattr(_C, "env_name", None) != "clifford" or getattr(_C, "gpu", None) != 0:
        raise RuntimeError(
            "Build the Clifford CPU backend before running curriculum training"
        )

    profile = curriculum_profile()
    n_qubits = env_int("N_QUBITS", 3)
    use_shortcut_gates = env_int("USE_SHORTCUT_GATES", 1)
    hidden_size = env_int("HIDDEN_SIZE", 128)
    run_name = default_run_name(
        n_qubits,
        hidden_size,
        use_shortcut_gates=bool(use_shortcut_gates),
    )
    checkpoint_dir = env_str("CHECKPOINT_DIR", os.path.join("checkpoints", run_name))
    log_dir = env_str("LOG_DIR", os.path.join("logs", run_name))
    run_id = str(int(1000 * time.time()))
    max_difficulty = env_float("MAX_DIFFICULTY", 64)
    curriculum_stride = env_float("CURRICULUM_STRIDE", 0.25)
    advance_threshold = env_float("MASTERY_THRESHOLD", 0.95)
    final_threshold = env_float("FINAL_MASTERY_THRESHOLD", 0.95)
    max_stage_attempts = env_int("MAX_STAGE_ATTEMPTS", 1)
    resume = env_int("RESUME", 1)
    load_path = latest_checkpoint(checkpoint_dir) if resume else None
    run_start_time = time.time()

    policy = None
    optimizer = None
    global_step_offset = 0
    all_logs = []
    latest_path = None
    difficulties = curriculum_difficulties(max_difficulty, curriculum_stride)
    success_step_stats = None
    keep_max_steps_floor = env_int("KEEP_MAX_STEPS_FLOOR", 0)
    max_steps_floor = 0

    for stage_idx, difficulty in enumerate(difficulties):
        max_steps, max_steps_source = stage_max_steps_with_source(
            difficulty,
            success_step_stats,
        )
        if keep_max_steps_floor:
            if max_steps_floor > max_steps:
                max_steps = max_steps_floor
                max_steps_source = f"KEEP_MAX_STEPS_FLOOR={max_steps_floor}"
        stop_threshold = (
            final_threshold if stage_idx == len(difficulties) - 1 else advance_threshold
        )
        min_timesteps = stage_min_timesteps(stage_idx)
        attempt = 1

        while True:
            args = build_args(
                difficulty=difficulty,
                max_steps=max_steps,
                total_timesteps=stage_timesteps(difficulty, stage_idx),
                load_model_path=load_path,
            )
            load_path = None

            rich.print(
                f"\n=== Clifford {n_qubits}q difficulty={format_difficulty(difficulty)} attempt={attempt} "
                f"profile={profile} "
                f"max_steps={max_steps} budget={args['train']['total_timesteps']} "
                f"agents={args['vec']['total_agents']} horizon={args['train']['horizon']} "
                f"minibatch={args['train']['minibatch_size']} "
                f"lr={args['train']['learning_rate']} ent={args['train']['ent_coef']} "
                f"max_steps_source={max_steps_source} "
                f"goal_bonus={args['env']['goal_bonus']} "
                f"failure_penalty={args['env']['failure_penalty']} "
                f"threshold={stop_threshold} "
                f"min_timesteps={min_timesteps} ==="
            )
            policy, optimizer, stage_steps, perf, latest_path, stage_logs, mastered = (
                train_stage(
                    args,
                    policy,
                    optimizer,
                    run_id,
                    global_step_offset,
                    stop_threshold,
                    min_timesteps,
                    run_start_time,
                )
            )
            global_step_offset += stage_steps
            all_logs.extend(stage_logs)

            rich.print(
                f"Stage difficulty={format_difficulty(difficulty)} attempt={attempt} env/perf={perf:.6f} "
                f"global_steps={global_step_offset}"
            )
            save_metrics(log_dir, run_id, args, all_logs)

            if mastered:
                if keep_max_steps_floor:
                    max_steps_floor = max(max_steps_floor, max_steps)
                if stage_logs:
                    mean_steps = stage_logs[-1].get("env/success_step_mean")
                    std_steps = stage_logs[-1].get("env/success_step_std")
                    if mean_steps is not None and std_steps is not None:
                        success_step_stats = (float(mean_steps), float(std_steps))
                break

            if max_stage_attempts > 0 and attempt >= max_stage_attempts:
                criteria = f"env/perf >= {stop_threshold}"
                if min_timesteps > 0:
                    criteria += f" after min_timesteps={min_timesteps}"
                criteria += " or env/perf >= 1.0"
                raise RuntimeError(
                    f"Difficulty {format_difficulty(difficulty)} did not meet advancement criteria "
                    f"({criteria}) within budget={args['train']['total_timesteps']} timesteps "
                    f"after {attempt} attempts"
                )

            rich.print(
                f"Retrying difficulty={format_difficulty(difficulty)}: env/perf={perf:.3f} "
                f"< {stop_threshold:.3f}"
            )
            attempt += 1

    rich.print(f"\nDone. Latest checkpoint: {latest_path}")


if __name__ == "__main__":
    main()

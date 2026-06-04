#!/usr/bin/env python3
import argparse
import json
import sys
import time
from copy import deepcopy

from pufferlib.pufferl import _resolve_backend, load_config, unroll_nested_dict


def base_args():
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        args = load_config("pathfinder")
    finally:
        sys.argv = saved_argv
    args["wandb"] = False
    args["render_mode"] = "None"
    args["checkpoint_interval"] = 10**12
    return args


def run_once(label, args, timesteps, eval_epochs_override):
    cfg = deepcopy(args)
    cfg["train"]["total_timesteps"] = timesteps
    cfg["train"]["gpus"] = 1
    cfg["nccl_id"] = b""
    backend = _resolve_backend(cfg)
    pufferl = backend.create_pufferl(cfg)
    train_epochs = max(1, timesteps // (cfg["vec"]["total_agents"] * cfg["train"]["horizon"]))
    flat = {}
    eval_flat = {}
    t0 = time.perf_counter()
    train_t0 = t0
    try:
        for _ in range(train_epochs):
            backend.rollouts(pufferl)
            backend.train(pufferl)
            flat = {**flat, **dict(unroll_nested_dict(backend.log(pufferl)))}
        train_elapsed = time.perf_counter() - train_t0
        train_steps = int(pufferl.global_step)

        eval_epochs = train_epochs // 2 if eval_epochs_override < 0 else eval_epochs_override
        for _ in range(eval_epochs):
            backend.rollouts(pufferl)
            eval_flat = {**eval_flat, **dict(unroll_nested_dict(backend.eval_log(pufferl)))}
    finally:
        actual_steps = int(pufferl.global_step)
        backend.close(pufferl)

    elapsed = time.perf_counter() - t0
    stats = eval_flat if eval_flat else flat
    summary = {
        "label": label,
        "timesteps_requested": timesteps,
        "train_timesteps_actual": train_steps,
        "timesteps_actual": actual_steps,
        "train_epochs": train_epochs,
        "eval_epochs": train_epochs // 2 if eval_epochs_override < 0 else eval_epochs_override,
        "train_seconds": train_elapsed,
        "seconds": elapsed,
        "wall_sps": train_steps / train_elapsed if train_elapsed > 0 else 0.0,
        "total_wall_sps": actual_steps / elapsed if elapsed > 0 else 0.0,
        "reported_sps": flat.get("SPS", 0.0),
        "success": stats.get("env/success", 0.0),
        "score": stats.get("env/score", 0.0),
        "episode_return": stats.get("env/episode_return", 0.0),
        "episode_length": stats.get("env/episode_length", 0.0),
        "wall_hits": stats.get("env/wall_hits", 0.0),
        "known_wall_deaths": stats.get("env/known_wall_deaths", 0.0),
        "revisits": stats.get("env/revisits", 0.0),
        "shortest_path_len": stats.get("env/shortest_path_len", 0.0),
        "agent_path_len": stats.get("env/agent_path_len", 0.0),
        "curriculum_level": stats.get("env/curriculum_level", 0.0),
        "curriculum_max_solution_len": stats.get("env/curriculum_max_solution_len", 0.0),
        "perf_rollout_sec": flat.get("perf/rollout", 0.0),
        "perf_eval_env_sec": flat.get("perf/eval_env", 0.0),
        "perf_train_sec": flat.get("perf/train", 0.0),
        "policy_hidden_size": cfg["policy"]["hidden_size"],
        "policy_num_layers": cfg["policy"]["num_layers"],
        "total_agents": cfg["vec"]["total_agents"],
        "horizon": cfg["train"]["horizon"],
        "minibatch_size": cfg["train"]["minibatch_size"],
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2097152)
    parser.add_argument("--eval-epochs", type=int, default=-1)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--total-agents", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--compare-hidden-size", type=int, default=None)
    parser.add_argument("--min-wall-sps", type=float, default=0.0)
    parser.add_argument("--min-success", type=float, default=0.0)
    parser.add_argument("--require-sps-gain", type=float, default=0.0)
    parser.add_argument("--require-success-ratio", type=float, default=0.0)
    opts = parser.parse_args()

    args = base_args()
    if opts.hidden_size is not None:
        args["policy"]["hidden_size"] = opts.hidden_size
    if opts.num_layers is not None:
        args["policy"]["num_layers"] = opts.num_layers
    if opts.total_agents is not None:
        args["vec"]["total_agents"] = opts.total_agents
    if opts.horizon is not None:
        args["train"]["horizon"] = opts.horizon
    if opts.minibatch_size is not None:
        args["train"]["minibatch_size"] = opts.minibatch_size

    if opts.compare_hidden_size is None:
        single = run_once("single", args, opts.timesteps, opts.eval_epochs)
        result = {
            "runs": [single],
            "checks": {
                "passes_min_wall_sps": single["wall_sps"] >= opts.min_wall_sps,
                "passes_min_success": single["success"] >= opts.min_success,
            },
        }
        if not result["checks"]["passes_min_wall_sps"] or not result["checks"]["passes_min_success"]:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
    else:
        baseline = run_once("baseline", args, opts.timesteps, opts.eval_epochs)
        candidate_args = deepcopy(args)
        candidate_args["policy"]["hidden_size"] = opts.compare_hidden_size
        candidate = run_once("candidate", candidate_args, opts.timesteps, opts.eval_epochs)
        sps_gain = (
            candidate["wall_sps"] / baseline["wall_sps"] - 1.0
            if baseline["wall_sps"] > 0.0 else 0.0
        )
        success_ratio = (
            candidate["success"] / baseline["success"]
            if baseline["success"] > 0.0 else 1.0
        )
        result = {
            "runs": [baseline, candidate],
            "comparison": {
                "sps_gain": sps_gain,
                "success_ratio": success_ratio,
                "passes_sps_gain": sps_gain >= opts.require_sps_gain,
                "passes_success_ratio": success_ratio >= opts.require_success_ratio,
            },
        }
        if not result["comparison"]["passes_sps_gain"] or not result["comparison"]["passes_success_ratio"]:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

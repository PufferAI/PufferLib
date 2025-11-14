import copy
import time
import random
import pickle

import numpy as np
import torch

import pufferlib.pufferl as pufferl
from pufferlib.sweep import Protein


def sweep(env_name, args):
    start_time = time.time()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    obs_file = f"sweep_observations_{env_name}_{timestamp}.pkl"
    print(f"Sweep observations will be saved to: {obs_file}")

    sweep_manager = Protein(args["sweep"], **args["sweep_extra"])
    points_per_run = args["sweep"]["downsample"]
    target_key = f"environment/{args['sweep']['metric']}"

    orig_arg = copy.deepcopy(args)
    suggest_history = []

    for i in range(args["max_runs"]):
        print(f"\n--- Starting sweep run {i + 1}/{args['max_runs']} ---")

        # Set a new random seed for each run
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Suggest new hyperparameters and update the arguments
        suggest_start_time = time.time()
        run_args, info = sweep_manager.suggest(args)
        suggest_time = time.time() - suggest_start_time
        print(f"sweep_manager.suggest() took {suggest_time:.4f} seconds")
        suggest_history.append(
            {
                "run_args": copy.deepcopy(run_args),
                "info": info,
                "suggest_time": suggest_time,
                "run_index": i,
            }
        )

        # Run training with the suggested hyperparameters
        all_logs = pufferl.train(env_name, args=run_args, should_stop_early=pufferl.stop_if_loss_nan)

        # Process the logs to get scores and costs for the sweep observation
        all_logs = [e for e in all_logs if target_key in e]
        if not all_logs:
            print(
                f"Warning: Target key '{target_key}' not found in logs for this run. Skipping observation."
            )
            continue

        scores = pufferl.downsample(
            [log[target_key] for log in all_logs], points_per_run
        )
        costs = pufferl.downsample([log["uptime"] for log in all_logs], points_per_run)
        timesteps = pufferl.downsample(
            [log["agent_steps"] for log in all_logs], points_per_run
        )

        # Observe the results of the run
        for score, cost, timestep in zip(scores, costs, timesteps):
            # Temporarily set total_timesteps to the observed value for accurate logging
            run_args["train"]["total_timesteps"] = timestep
            sweep_manager.observe(run_args, score, cost)

        # Save observations to a fixed file every 10 runs and at the end
        if (i + 1) % 10 == 0 or (i + 1) >= args["max_runs"]:
            print(f"\n--- Saving sweep observations to {obs_file} (run {i + 1}) ---")
            with open(obs_file, "wb") as f:
                pickle.dump(
                    {
                        "success": sweep_manager.success_observations,
                        "failure": sweep_manager.failure_observations,
                        "suggest_history": suggest_history,
                        "total_sweep_time": time.time() - start_time,
                        "args": orig_arg,
                    },
                    f,
                )

    total_sweep_time = time.time() - start_time
    print(f"\n--- Total sweep time: {total_sweep_time:.2f} seconds ---")
    total_suggest_time = sum(h["suggest_time"] for h in suggest_history)
    print(f"--- Total suggest time: {total_suggest_time:.2f} seconds ---")


if __name__ == "__main__":
    import sys

    env_name = sys.argv.pop(1) if len(sys.argv) > 1 else "puffer_pong"

    # parser = None
    parser = pufferl.make_parser()
    parser.add_argument('--gp-iter', type=int, default=None)
    parser.add_argument('--gp-lr', type=float, default=None)
    # parser.add_argument('--use-gpu', action="store_true")
    # parser.add_argument('--prune-pareto', action="store_true")
    # parser.add_argument('--ucb-beta', type=float, default=None)
    args = pufferl.load_config(env_name, parser)

    # Use wandb
    args["wandb"] = True
    args["no_model_upload"] = True
    # args["train"]["optimizer"] = "adam"

    # Custom sweep args
    args["sweep_extra"] = {}
    if args["gp_iter"] is not None:
        args["sweep_extra"]["gp_training_iter"] = args["gp_iter"]
    if args["gp_lr"] is not None:
        args["sweep_extra"]["gp_learning_rate"] = args["gp_lr"]
    # if args["use_gpu"]:
    #     args["sweep"]["use_gpu"] = True
    # if args["prune_pareto"]:
    #     args["sweep_extra"]["prune_pareto"] = args["prune_pareto"]
    # if args["ucb_beta"] is not None:
    #     args["sweep_extra"]["ucb_beta"] = args["ucb_beta"]

    sweep(env_name, args)

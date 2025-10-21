import os
import pickle

import numpy as np
import torch

import pandas as pd
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
from pufferlib import pufferl
from pufferlib.sweep import Protein


def evaluate_gp(gp_model, likelihood, test_x, test_y):
    """
    Evaluates a trained GPyTorch model on test data.

    Args:
        gp_model: The trained gpytorch.models.ExactGP model.
        likelihood: The gpytorch.likelihoods.GaussianLikelihood.
        test_x (torch.Tensor): Test input features.
        test_y (torch.Tensor): Test target values.

    Returns:
        dict: A dictionary of evaluation metrics.
    """
    gp_model.eval()
    likelihood.eval()

    # Move tensors to the same device as the model
    device = next(gp_model.parameters()).device
    test_x = test_x.float().to(device)
    test_y = test_y.float().to(device)

    with torch.no_grad():
        # Get predictions from the model
        predictions = likelihood(gp_model(test_x))
        pred_mean = predictions.mean
        pred_std = predictions.stddev

    # 1. Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(pred_mean - test_y)).item()

    # 2. Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(torch.mean((pred_mean - test_y) ** 2)).item()

    # 3. Negative Log-Likelihood (NLL)
    # This is the negative of the log probability of the test data under the model's predictions
    nll = (
        -torch.distributions.Normal(pred_mean, pred_std).log_prob(test_y).mean().item()
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "nll": nll,
    }


def _validate_gp_model(
    gp_model, likelihood, val_obs, train_obs, target_key, use_log=False
):
    """Helper to prepare validation data, evaluate a GP model, and print metrics."""
    val_x = torch.from_numpy(np.stack([e["input"] for e in val_obs])).to(torch.float64)
    val_y = torch.from_numpy(np.array([e[target_key] for e in val_obs])).to(
        torch.float64
    )
    train_y = np.array([e[target_key] for e in train_obs])

    if use_log:
        val_y = torch.log(val_y)
        train_y = np.log(train_y)

    min_y, max_y = np.min(train_y), np.max(train_y)
    val_y_norm = (val_y - min_y) / (max_y - min_y + 1e-6)

    metrics = evaluate_gp(gp_model, likelihood, val_x, val_y_norm)

    target_name = f"log-{target_key}" if use_log else target_key
    print(
        f"{target_key.capitalize()} GP Validation Metrics (on normalized {target_name}):"
    )
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def run_experiment(args, train_obs, val_obs, gp_iter, gp_lr, use_gpu=False):
    """
    Trains and validates the GP models from the Protein sweep.
    """
    print(f"\n--- Testing with gp_iter={gp_iter}, gp_lr={gp_lr} ---")

    # Initialize a Protein object to get access to its GP models and helpers
    sweep_manager = Protein(
        args["sweep"],
        gp_training_iter=gp_iter,
        gp_learning_rate=gp_lr,
        use_gpu=use_gpu,
    )

    # --- Train GPs using the observe/suggest loop ---
    print("Training GPs by observing data iteratively (10 obs per suggest)...")
    batch_size = 10
    score_loss_history = []
    cost_loss_history = []
    num_batches = len(train_obs) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_obs = train_obs[start_idx:end_idx]

        for obs in batch_obs:
            # The 'input' in the pickled file is a numpy array of normalized params.
            # We need to convert it back to a dict of unnormalized params for observe.
            hypers = sweep_manager.hyperparameters.to_dict(obs["input"])
            sweep_manager.observe(hypers, obs["output"], obs["cost"])

        # Skip training until some data is in
        if i < 10:
            continue

        score_loss, cost_loss = sweep_manager._train_gp_models()
        score_loss_history.append(score_loss)
        cost_loss_history.append(cost_loss)

    print(
        f"  Finished training on {len(train_obs)} observations over {num_batches} iterations."
    )

    # --- Evaluate Score GP ---
    score_metrics = _validate_gp_model(
        sweep_manager.gp_score,
        sweep_manager.likelihood_score,
        val_obs,
        train_obs,
        target_key="output",
    )

    # --- Evaluate Cost GP ---
    cost_metrics = _validate_gp_model(
        sweep_manager.gp_cost,
        sweep_manager.likelihood_cost,
        val_obs,
        train_obs,
        "cost",
        use_log=True,
    )

    return score_metrics, cost_metrics, score_loss_history, cost_loss_history


def visualize_results(results_file):
    """
    Loads GP evaluation results and creates visualizations.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("Install matplotlib to visualize results: pip install matplotlib")
        return

    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return

    with open(results_file, "rb") as f:
        results = pickle.load(f)

    if not results:
        print("No results to visualize.")
        return

    # --- Data Preparation ---
    # Unpack nested metric dictionaries
    flat_results = []
    for res in results:
        if not ("score_metrics" in res or res["success"]):
            continue

        flat_res = {
            "gp_iter": res["gp_iter"],
            "gp_lr": res["gp_lr"],
            "score_mae": res["score_metrics"]["mae"],
            "score_rmse": res["score_metrics"]["rmse"],
            "score_nll": res["score_metrics"]["nll"],
            "cost_mae": res["cost_metrics"]["mae"],
            "cost_rmse": res["cost_metrics"]["rmse"],
            "cost_nll": res["cost_metrics"]["nll"],
            "score_loss_history": res["score_loss_history"],
            "cost_loss_history": res["cost_loss_history"],
        }
        flat_results.append(flat_res)

    df = pd.DataFrame(flat_results)
    iters_list = sorted(df["gp_iter"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(iters_list)))
    iter_color_map = dict(zip(iters_list, colors))
 
    lrs_list = sorted(df["gp_lr"].unique())
    # Use a different colormap for the learning rates in the loss plot
    lr_colors = plt.cm.plasma(np.linspace(0, 1, len(lrs_list)))
    lr_color_map = dict(zip(lrs_list, lr_colors))

    # --- Plot 1: Metrics vs. Learning Rate ---
    for gp_type in ["score", "cost"]:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
        fig.suptitle(f"{gp_type.capitalize()} GP Validation Metrics vs. Learning Rate", fontsize=16)

        for i, metric in enumerate(["mae", "rmse", "nll"]):
            ax = axes[i]
            for iters in iters_list:
                subset = df[df["gp_iter"] == iters]
                ax.plot(
                    subset["gp_lr"],
                    subset[f"{gp_type}_{metric}"],
                    marker="o",
                    linestyle="-",
                    color=iter_color_map[iters],
                    label=f"{iters} iters",
                )
            ax.set_xscale("log")
            ax.set_xlabel("GP Learning Rate (log scale)")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} vs. LR")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.show()

    # --- Plot 2: Loss Curves ---
    fig, axes = plt.subplots(
        2, len(iters_list), figsize=(6 * len(iters_list), 10), sharey="row"
    )
    fig.suptitle("GP Training Loss Curves", fontsize=16)
    for i, iters in enumerate(iters_list):
        # Score Loss
        ax_score = axes[0, i]
        subset = df[df["gp_iter"] == iters]
        for _, row in subset.sort_values("gp_lr").iterrows():
            ax_score.plot(
                row["score_loss_history"],
                alpha=0.7,
                color=lr_color_map[row["gp_lr"]],
                label=f'lr={row["gp_lr"]:.1e}',
            )
        ax_score.set_title(f"Score Loss (iters={iters})")
        ax_score.set_xlabel("Training Batch")
        if i == 0:
            ax_score.set_ylabel("Loss (MLL)")
        ax_score.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_score.grid(True, linestyle="--", linewidth=0.5)
        ax_score.legend(fontsize="small")

        # Cost Loss
        ax_cost = axes[1, i]
        for _, row in subset.sort_values("gp_lr").iterrows():
            ax_cost.plot(
                row["cost_loss_history"],
                alpha=0.7,
                color=lr_color_map[row["gp_lr"]],
                label=f'lr={row["gp_lr"]:.1e}',
            )
        ax_cost.set_title(f"Cost Loss (iters={iters})")
        ax_cost.set_xlabel("Training Batch")
        if i == 0:
            ax_cost.set_ylabel("Loss (MLL)")
        ax_cost.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_cost.grid(True, linestyle="--", linewidth=0.5)
        ax_cost.legend(fontsize="small")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.show()

    print("\nVisualizations generated. Press Enter in the terminal to exit.")
    input()


if __name__ == "__main__":
    parser = pufferl.make_parser()
    # Use tests/test_custom_sweep.py to collect sweep obs pkl
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        default="sweep_observations.pkl",
        help="Path to the input sweep observations pickle file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="gp_evaluation_results.pkl",
        help="Path to save the output evaluation results pickle file.",
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (the rest is for validation).",
    )
    parser.add_argument(
        "--visualize-only",
        # default=True,
        action="store_true",
        help="Skip experiments and only visualize results from the output file.",
    )

    env_name = "puffer_breakout"
    args = pufferl.load_config(env_name, parser)

    output_file = args["output_file"]

    if args["visualize_only"]:
        print(f"--- Visualization Mode: Loading results from {output_file} ---")
        visualize_results(output_file)
        exit()

    all_results = []
    completed_runs = set()

    # Load existing results to resume, or rename old file to start fresh
    if os.path.exists(output_file):
        print(f"Found existing results file: {output_file}. Attempting to resume.")
        with open(output_file, "rb") as f:
            all_results = pickle.load(f)
        for res in all_results:
            completed_runs.add((res["gp_iter"], res["gp_lr"]))
        print(f"Loaded {len(all_results)} completed runs. Skipping them.")

    with open(args["input_file"], "rb") as f:
        data = pickle.load(f)

    # We only use successful observations for training the GP
    success_observations = data.get("success", [])
    print(f"Loaded {len(success_observations)} successful observations.")

    # Split data based on the provided ratio
    split_idx = int(len(success_observations) * args["train_split_ratio"])
    train_observations = success_observations[:split_idx]
    validation_observations = success_observations[split_idx:]

    if not train_observations or not validation_observations:
        raise ValueError(
            "Data split resulted in empty training or validation set. Check data and split ratio."
        )

    print(
        f"Using {len(train_observations)} for training and {len(validation_observations)} for validation."
    )

    # --- Define Hyperparameter Grid ---
    gp_iters_to_test = [50, 100, 200]
    gp_lrs_to_test = np.logspace(-2, -4, 20).tolist()  # 20 rates from 0.01 to 0.0001

    for iters in gp_iters_to_test:
        for lr in gp_lrs_to_test:
            if (iters, lr) in completed_runs:
                print(f"Skipping already completed run: gp_iter={iters}, gp_lr={lr}")
                continue

            try:
                score_metrics, cost_metrics, score_loss_hist, cost_loss_hist = (
                    run_experiment(
                        args,
                        train_observations,
                        validation_observations,
                        gp_iter=iters,
                        gp_lr=lr,
                        use_gpu=True,
                    )
                )
                all_results.append(
                    {
                        "gp_iter": iters,
                        "gp_lr": lr,
                        "score_metrics": score_metrics,
                        "cost_metrics": cost_metrics,
                        "score_loss_history": score_loss_hist,
                        "cost_loss_history": cost_loss_hist,
                        "success": True,
                    }
                )
            except:
                all_results.append({"gp_iter": iters, "gp_lr": lr, "success": False})

            # Save results after each experiment to avoid data loss on interruption
            with open(args["output_file"], "wb") as f:
                pickle.dump(all_results, f)
            print(
                f"Saved {len(all_results)} experiment results to {args['output_file']}"
            )

    print(f"\nFinished. All evaluation results saved to {args['output_file']}")
    visualize_results(output_file)

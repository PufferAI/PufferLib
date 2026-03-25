#!/usr/bin/env python3

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pufferlib.ocean.clifford import Clifford
from pufferlib.pytorch import layer_init

N_QUBITS = 4
START_DIFFICULTY = 1.0
MAX_DIFFICULTY = 1000.0
SUCCESS_THRESHOLD = 0.90
DIFFICULTY_STEP = 0.25
SINGLE_QUBIT_COST = 0.01
GOAL_BONUS = 0.0
REWARD_MODE = "hamming_left"
HAMMING_LEFT_SCALE = 0.5

NUM_ENVS = 2048
TOTAL_TIMESTEPS = 50_000_000
LEARNING_RATE = 2.5e-4
HIDDEN_SIZE = 256
HIDDEN_LAYERS = 2
N_STEPS = 128
MINIBATCH_SIZE = 8192
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.0
CLIP_COEF = 0.15
VF_CLIP_COEF = 0.2
TARGET_KL = 0.0

EVAL_FREQ_STEPS = 98_304
PROGRESSION_EVAL_EPISODES = 50
GREEDY_EVAL_DIFFICULTY_CUTOFF = 20.0
MIN_ADVANCE_EVALS = 1
EMA_SR_ALPHA = 0.9
MAX_STEPS_SLACK = 64
MIN_MAX_STEPS = 32
FORCE_MAX_STEPS_AFTER_DIFFICULTY = 12.0
FORCE_MAX_STEPS_VALUE = 1000

TRAIN_CONFIG_DEFAULTS = {
    "env": "puffer_clifford",
    "torch_deterministic": True,
    "cpu_offload": False,
    "optimizer": "muon",
    "precision": "float32",
    "anneal_lr": True,
    "min_lr_ratio": 0.0,
    "vf_coef": 2.0,
    "max_grad_norm": 1.5,
    "adam_beta1": 0.95,
    "adam_beta2": 0.999,
    "adam_eps": 1e-12,
    "data_dir": "experiments",
    "checkpoint_interval": 200,
    "max_minibatch_size": 32768,
    "compile": False,
    "use_rnn": False,
    "vtrace_rho_clip": 1.0,
    "vtrace_c_clip": 1.0,
    "prio_alpha": 0.8,
    "prio_beta0": 0.2,
}

PANEL_STAT_KEYS = {
    "curriculum/difficulty",
    "curriculum/rollout_success_rate",
    "curriculum/decision_success_rate",
    "curriculum/ema_success_rate",
    "curriculum/mean_reward",
    "curriculum/mean_cz",
    "curriculum/evals_since_advance",
    "curriculum/consecutive_above_threshold",
    "curriculum/ent_coef",
    "perf",
    "score",
    "episode_return",
    "episode_length",
    "mean_cz",
    "success_rate",
}


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Policy(torch.nn.Module):
    def __init__(self, env, hidden_size=HIDDEN_SIZE, hidden_layers=HIDDEN_LAYERS):
        super().__init__()
        obs_size = int(np.prod(env.single_observation_space.shape))
        num_actions = int(env.single_action_space.n)
        hidden_layers = max(int(hidden_layers), 1)

        backbone = [torch.nn.LayerNorm(obs_size)]
        in_features = obs_size
        for _ in range(hidden_layers):
            backbone.append(layer_init(torch.nn.Linear(in_features, int(hidden_size))))
            backbone.append(torch.nn.GELU())
            in_features = int(hidden_size)
        self.backbone = torch.nn.Sequential(*backbone)
        self.action_head = torch.nn.Linear(in_features, num_actions)
        self.value_head = torch.nn.Linear(in_features, 1)
        layer_init(self.action_head, std=0.01)
        layer_init(self.value_head, std=1.0)

    def forward_eval(self, observations, state=None):
        hidden = self.backbone(observations.float().view(observations.shape[0], -1))
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

def make_train_config(args):
    return {
        **TRAIN_CONFIG_DEFAULTS,
        "total_timesteps": int(args.total_timesteps),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.num_envs) * N_STEPS,
        "bptt_horizon": N_STEPS,
        "minibatch_size": max(MINIBATCH_SIZE, N_STEPS),
        "update_epochs": UPDATE_EPOCHS,
        "device": args.device,
        "seed": int(args.seed),
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "ent_coef": float(args.ent_coef),
        "clip_coef": CLIP_COEF,
        "vf_clip_coef": VF_CLIP_COEF,
        "target_kl": TARGET_KL,
    }


def make_env(num_envs, difficulty, max_steps, seed):
    return Clifford(
        num_envs=int(num_envs),
        n_qubits=N_QUBITS,
        difficulty=float(difficulty),
        max_steps=int(max_steps),
        single_qubit_cost=SINGLE_QUBIT_COST,
        goal_bonus=GOAL_BONUS,
        reward_mode=REWARD_MODE,
        hamming_left_scale=HAMMING_LEFT_SCALE,
        use_reset_pool=True,
        seed=int(seed),
    )


def run_greedy_eval(policy, env, device, episodes, seed):
    completed = 0
    successes = 0
    rewards = []
    completed_cz_counts = []

    obs, _ = env.reset(seed=seed)
    episode_rewards = np.zeros(env.num_agents, dtype=np.float32)
    episode_cz_counts = np.zeros(env.num_agents, dtype=np.int32)

    policy_was_training = policy.training
    policy.eval()

    with torch.no_grad():
        while completed < episodes:
            obs_t = torch.as_tensor(obs, device=device)
            logits, _values = policy.forward_eval(obs_t)
            actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32, copy=False)
            obs, step_rewards, terminals, truncations, _info = env.step(actions)
            episode_rewards += step_rewards
            for idx, action in enumerate(actions):
                if env._actions[int(action)][0] == "cz":
                    episode_cz_counts[idx] += 1

            done_mask = np.logical_or(terminals, truncations)
            done_indices = np.flatnonzero(done_mask)
            for idx in done_indices:
                completed += 1
                rewards.append(float(episode_rewards[idx]))
                completed_cz_counts.append(int(episode_cz_counts[idx]))
                if terminals[idx]:
                    successes += 1
                episode_rewards[idx] = 0.0
                episode_cz_counts[idx] = 0
                if completed >= episodes:
                    break

    if policy_was_training:
        policy.train()

    return {
        "success_rate": successes / max(completed, 1),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_cz": float(np.mean(completed_cz_counts)) if completed_cz_counts else 0.0,
    }


def _metric_values(rollout_stats, key):
    values = rollout_stats.get(key, [])
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    if values is None:
        return []
    return [float(values)]


def merge_rollout_stats(rollout_stats, extra_metrics):
    merged = defaultdict(list)
    for source in (rollout_stats, extra_metrics):
        if not source:
            continue
        for key, value in source.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                merged[key].extend(value)
            else:
                merged[key].append(value)
    return merged


def rollout_progression_metrics(rollout_stats):
    if not rollout_stats:
        return None
    episode_count = sum(_metric_values(rollout_stats, "episode_count"))
    if episode_count <= 0.0:
        return None

    success_count = sum(_metric_values(rollout_stats, "success_count"))
    reward_sum = sum(_metric_values(rollout_stats, "episode_return_sum"))
    episode_cz_sum = sum(_metric_values(rollout_stats, "episode_cz_sum"))
    return {
        "success_rate": success_count / episode_count,
        "mean_reward": reward_sum / episode_count,
        "mean_cz": episode_cz_sum / episode_count,
    }


def compute_max_steps(difficulty):
    max_steps = int(math.ceil(max(float(difficulty), 0.0) - 1e-12)) + MAX_STEPS_SLACK
    if (
        FORCE_MAX_STEPS_AFTER_DIFFICULTY is not None
        and FORCE_MAX_STEPS_VALUE > 0
        and difficulty >= FORCE_MAX_STEPS_AFTER_DIFFICULTY
    ):
        max_steps = FORCE_MAX_STEPS_VALUE
    return max(max_steps, MIN_MAX_STEPS, 1)


def update_curriculum_stats(trainer, values):
    for key, value in values.items():
        trainer.stats[key] = value
        trainer.last_stats[key] = value


def curriculum_stats(
    difficulty,
    rollout_success_rate,
    decision_success_rate,
    ema_success_rate,
    mean_reward,
    mean_cz,
    evals_since_advance,
    consecutive_above_threshold,
    ent_coef,
):
    return {
        "curriculum/difficulty": float(difficulty),
        "curriculum/mean_reward": float(mean_reward),
        "curriculum/decision_success_rate": float(decision_success_rate),
        "curriculum/ema_success_rate": float(ema_success_rate),
        "curriculum/rollout_success_rate": float(rollout_success_rate),
        "curriculum/mean_cz": float(mean_cz),
        "curriculum/evals_since_advance": float(evals_since_advance),
        "curriculum/consecutive_above_threshold": float(consecutive_above_threshold),
        "curriculum/ent_coef": float(ent_coef),
    }


def prune_panel_stats(stats):
    if not stats:
        return
    for key in list(stats.keys()):
        if key not in PANEL_STAT_KEYS:
            del stats[key]


def install_panel_filter(trainer):
    original_print_dashboard = trainer.print_dashboard

    def filtered_print_dashboard(*args, **kwargs):
        if trainer.stats:
            trainer.raw_stats = dict(trainer.stats)
        elif trainer.last_stats:
            trainer.raw_stats = dict(trainer.last_stats)
        prune_panel_stats(trainer.stats)
        prune_panel_stats(trainer.last_stats)
        return original_print_dashboard(*args, **kwargs)

    trainer.print_dashboard = filtered_print_dashboard


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal Ocean Clifford trainer focused on the current working 4-qubit regime"
    )
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--ent-coef", type=float, default=ENT_COEF)
    parser.add_argument("--difficulty-step", type=float, default=DIFFICULTY_STEP)
    return parser.parse_args()


def validate_args(args):
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if args.total_timesteps < int(args.num_envs) * N_STEPS:
        raise ValueError(f"total_timesteps must be at least {int(args.num_envs) * N_STEPS}")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.ent_coef < 0.0:
        raise ValueError("ent_coef must be non-negative")
    if args.difficulty_step <= 0.0:
        raise ValueError("difficulty_step must be positive")


def main():
    args = parse_args()
    validate_args(args)
    from pufferlib import pufferl

    current_difficulty = START_DIFFICULTY
    current_max_steps = compute_max_steps(current_difficulty)

    train_env = make_env(args.num_envs, current_difficulty, current_max_steps, args.seed)
    eval_env = make_env(
        min(PROGRESSION_EVAL_EPISODES, int(args.num_envs)),
        current_difficulty,
        current_max_steps,
        args.seed + 10_000,
    )

    policy = Policy(train_env).to(args.device)
    trainer = pufferl.PuffeRL(make_train_config(args), train_env, policy)
    install_panel_filter(trainer)

    last_progress_step = 0
    ema_success_rate = 0.0
    ema_initialized = False
    consecutive_above_threshold = 0
    evals_since_advance = 0
    last_rollout_success_rate = 0.0
    last_decision_success_rate = 0.0
    last_mean_reward = 0.0
    last_mean_cz = 0.0

    update_curriculum_stats(
        trainer,
        curriculum_stats(
            current_difficulty,
            last_rollout_success_rate,
            last_decision_success_rate,
            ema_success_rate,
            last_mean_reward,
            last_mean_cz,
            evals_since_advance,
            consecutive_above_threshold,
            trainer.config["ent_coef"],
        ),
    )
    trainer.print_dashboard(clear=True)

    try:
        while trainer.epoch < trainer.total_epochs:
            update_curriculum_stats(
                trainer,
                curriculum_stats(
                    current_difficulty,
                    last_rollout_success_rate,
                    last_decision_success_rate,
                    ema_success_rate,
                    last_mean_reward,
                    last_mean_cz,
                    evals_since_advance,
                    consecutive_above_threshold,
                    trainer.config["ent_coef"],
                ),
            )
            trainer.evaluate()
            trainer.train()

            if trainer.global_step - last_progress_step < EVAL_FREQ_STEPS:
                continue
            last_progress_step = trainer.global_step

            rollout_stats = merge_rollout_stats(getattr(trainer, "raw_stats", None), train_env.flush_logs())
            rollout_metrics = rollout_progression_metrics(rollout_stats)
            if rollout_metrics is None:
                continue

            decision_metrics = rollout_metrics
            if current_difficulty < GREEDY_EVAL_DIFFICULTY_CUTOFF:
                eval_env.set_difficulty(current_difficulty)
                eval_env.set_max_steps(current_max_steps)
                decision_metrics = run_greedy_eval(
                    policy=policy,
                    env=eval_env,
                    device=args.device,
                    episodes=PROGRESSION_EVAL_EPISODES,
                    seed=args.seed + trainer.epoch,
                )

            raw_success_rate = float(rollout_metrics["success_rate"])
            if not ema_initialized:
                ema_success_rate = raw_success_rate
                ema_initialized = True
            else:
                ema_success_rate = (
                    float(EMA_SR_ALPHA) * ema_success_rate
                    + (1.0 - float(EMA_SR_ALPHA)) * raw_success_rate
                )

            decision_success_rate = float(decision_metrics["success_rate"])
            advancement_success_rate = (
                decision_success_rate
                if current_difficulty < GREEDY_EVAL_DIFFICULTY_CUTOFF
                else ema_success_rate
            )

            if advancement_success_rate >= SUCCESS_THRESHOLD:
                consecutive_above_threshold += 1
            else:
                consecutive_above_threshold = 0

            if (
                consecutive_above_threshold >= MIN_ADVANCE_EVALS
                and current_difficulty < MAX_DIFFICULTY
            ):
                current_difficulty = min(MAX_DIFFICULTY, current_difficulty + args.difficulty_step)
                current_max_steps = compute_max_steps(current_difficulty)
                train_env.set_difficulty(current_difficulty)
                train_env.set_max_steps(current_max_steps)
                eval_env.set_difficulty(current_difficulty)
                eval_env.set_max_steps(current_max_steps)
                consecutive_above_threshold = 0
                evals_since_advance = 0
            else:
                evals_since_advance += 1

            last_rollout_success_rate = raw_success_rate
            last_decision_success_rate = decision_success_rate
            last_mean_reward = float(rollout_metrics["mean_reward"])
            last_mean_cz = float(rollout_metrics["mean_cz"])

            update_curriculum_stats(
                trainer,
                curriculum_stats(
                    current_difficulty,
                    last_rollout_success_rate,
                    last_decision_success_rate,
                    ema_success_rate,
                    last_mean_reward,
                    last_mean_cz,
                    evals_since_advance,
                    consecutive_above_threshold,
                    trainer.config["ent_coef"],
                ),
            )

        trainer.print_dashboard()
    finally:
        trainer.close()
        eval_env.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Elo Evaluation System for Dogfight Self-Play.

Evaluates trained models against fixed reference opponents to compute
a comparable Elo rating. Designed for post-training sweep optimization.

Reference opponents:
- Autopilot stages (no .pt files needed, runs in C code)
- Neural checkpoints (saved via bootstrap command)

Usage:
    # Evaluate a model against references
    python -m pufferlib.ocean.dogfight.elo_eval eval --model experiments/model.pt

    # Bootstrap: save model as reference
    python -m pufferlib.ocean.dogfight.elo_eval bootstrap --model experiments/model.pt \\
        --tag ref_v1_strong --elo 1200

    # Post-sweep round-robin tournament
    python -m pufferlib.ocean.dogfight.elo_eval tournament \\
        --models experiments/puffer_dogfight_*.pt --games 20
"""
import argparse
import copy
import json
import math
import multiprocessing
import os
import shutil
import time

import numpy as np
import torch

from pufferlib.ocean.dogfight.dogfight import Dogfight, OBS_SIZES
from pufferlib.ocean.dogfight.dogfight_log import init_log, log


def load_policy_from_path(path, env, device='cuda', hidden_size=128):
    """Load policy from .pt file, handling both PuffeRL and CheckpointQueue formats.

    Args:
        path: Path to .pt checkpoint file.
        env: Dogfight environment instance (for policy construction).
        device: Torch device string.
        hidden_size: Hidden layer size for policy network.

    Returns:
        Policy in eval mode with frozen parameters.
    """
    from pufferlib.models import LSTMWrapper, Default as Policy

    inner_policy = Policy(env, hidden_size=hidden_size)
    policy = LSTMWrapper(env, inner_policy, input_size=hidden_size, hidden_size=hidden_size)
    policy = policy.to(device)

    state_dict = torch.load(path, map_location=device, weights_only=True)

    if isinstance(state_dict, dict) and 'policy_state_dict' in state_dict:
        policy.load_state_dict(state_dict['policy_state_dict'])
    else:
        try:
            policy.load_state_dict(state_dict)
        except RuntimeError:
            cleaned = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '')
                cleaned[new_k] = v
            policy.load_state_dict(cleaned)

    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False

    return policy


def run_matches(env, player_policy, opponent_policy, num_games, device='cuda',
                hidden_size=128, player_hidden_size=None, opponent_hidden_size=None):
    """Run vectorized matches between player and opponent policies.

    For autopilot opponents, opponent_policy is None and the C code handles
    opponent behavior based on the curriculum stage.

    Args:
        env: Dogfight environment (already configured with correct stage/override).
        player_policy: Player neural network policy.
        opponent_policy: Opponent neural network policy, or None for autopilot.
        num_games: Number of games to play.
        device: Torch device string.
        hidden_size: Default hidden size for both player and opponent LSTM.
        player_hidden_size: Override hidden size for player LSTM (defaults to hidden_size).
        opponent_hidden_size: Override hidden size for opponent LSTM (defaults to hidden_size).

    Returns:
        dict with keys: wins, losses, draws (from player perspective).
    """
    from pufferlib.ocean.dogfight import binding

    p_hs = player_hidden_size or hidden_size
    o_hs = opponent_hidden_size or hidden_size

    results = {'wins': 0, 'losses': 0, 'draws': 0, 'clean_fights': 0}
    games_played = 0
    max_ticks = 6000  # 2 minutes at 50Hz

    while games_played < num_games:
        # Use different seed per game to avoid deterministic spawn positions
        # (vec_reset calls srand(seed), so seed=0 gives identical spawns every game)
        obs, _ = env.reset(seed=games_played + 1)
        done = False
        tick = 0

        # Init LSTM state for new episode
        state_p = {'lstm_h': torch.zeros(1, p_hs, device=device),
                    'lstm_c': torch.zeros(1, p_hs, device=device)}
        state_o = {'lstm_h': torch.zeros(1, o_hs, device=device),
                    'lstm_c': torch.zeros(1, o_hs, device=device)}

        while not done and tick < max_ticks:
            obs_tensor = torch.as_tensor(obs, device=device)

            with torch.no_grad():
                logits_p, _ = player_policy.forward_eval(obs_tensor, state=state_p)
                action_p = logits_p.sample()
                action_p_np = action_p.cpu().numpy().astype(np.float32)
                action_p_np = np.clip(action_p_np, -1, 1)

            if opponent_policy is not None:
                obs_opp = binding.vec_get_opponent_observations(env.c_envs)
                obs_opp = torch.as_tensor(obs_opp[:1], device=device)
                if torch.isnan(obs_opp).any():
                    obs_opp = torch.nan_to_num(obs_opp, nan=0.0)

                with torch.no_grad():
                    logits_o, _ = opponent_policy.forward_eval(obs_opp, state=state_o)
                    action_o = logits_o.sample()
                    action_o_np = action_o.cpu().numpy().astype(np.float32)
                    action_o_np = np.clip(action_o_np, -1, 1)

                binding.vec_set_opponent_actions(env.c_envs, action_o_np)

            obs, reward, terminal, truncation, info = env.step(action_p_np.reshape(1, -1))
            done = terminal[0] or truncation[0]
            tick += 1

        # Determine outcome from final reward
        # Kill: ±1.0, Crash survival: ±0.25, Timeout: -0.5
        final_reward = reward[0]
        if final_reward > 0.1:
            results['wins'] += 1
        elif final_reward < -0.1:
            results['losses'] += 1
        else:
            results['draws'] += 1

        games_played += 1

    return results


def run_matches_vectorized(player_policy, opponent_policy, num_games,
                           obs_scheme=0, hidden_size=128, num_envs=64,
                           device='cuda', max_ticks=6000):
    """Run many games in parallel using vectorized envs.

    Creates a temporary env with num_envs parallel environments,
    runs batches of num_envs games simultaneously.

    Args:
        player_policy: Player neural network policy.
        opponent_policy: Opponent neural network policy.
        num_games: Total number of games to play.
        obs_scheme: Observation scheme for the environment.
        hidden_size: Hidden size for LSTM state.
        num_envs: Number of parallel environments.
        device: Torch device string.
        max_ticks: Maximum ticks per episode before forced draw.

    Returns:
        dict with keys: wins, losses, draws (from player perspective).
    """
    from pufferlib.ocean.dogfight import binding

    env = Dogfight(
        num_envs=num_envs,
        render_mode=None,
        obs_scheme=obs_scheme,
        curriculum_enabled=1,
        curriculum_randomize=1,
        eval_spawn_mode=2,
        fixed_stage=20,
        max_steps=max_ticks,
    )
    binding.vec_enable_opponent_override(env.c_envs, 1)

    results = {'wins': 0, 'losses': 0, 'draws': 0}
    games_completed = 0

    # Per-env tick counters
    env_ticks = np.zeros(num_envs, dtype=np.int32)

    # Init batched LSTM state
    state_p = {'lstm_h': torch.zeros(num_envs, hidden_size, device=device),
                'lstm_c': torch.zeros(num_envs, hidden_size, device=device)}
    state_o = {'lstm_h': torch.zeros(num_envs, hidden_size, device=device),
                'lstm_c': torch.zeros(num_envs, hidden_size, device=device)}

    obs, _ = env.reset()

    while games_completed < num_games:
        obs_tensor = torch.as_tensor(obs, device=device)

        with torch.no_grad():
            logits_p, _ = player_policy.forward_eval(obs_tensor, state=state_p)
            action_p = logits_p.sample()
            action_p_np = action_p.cpu().numpy().astype(np.float32)
            action_p_np = np.clip(action_p_np, -1, 1)

        # Opponent forward pass
        obs_opp = binding.vec_get_opponent_observations(env.c_envs)
        obs_opp = torch.as_tensor(obs_opp, device=device)
        if torch.isnan(obs_opp).any():
            obs_opp = torch.nan_to_num(obs_opp, nan=0.0)

        with torch.no_grad():
            logits_o, _ = opponent_policy.forward_eval(obs_opp, state=state_o)
            action_o = logits_o.sample()
            action_o_np = action_o.cpu().numpy().astype(np.float32)
            action_o_np = np.clip(action_o_np, -1, 1)

        binding.vec_set_opponent_actions(env.c_envs, action_o_np)
        obs, reward, terminal, truncation, info = env.step(action_p_np)
        env_ticks += 1

        # Check each env for episode completion
        done_mask = terminal | truncation
        for i in range(num_envs):
            if done_mask[i] and games_completed < num_games:
                r = reward[i]
                if r > 0.1:
                    results['wins'] += 1
                elif r < -0.1:
                    results['losses'] += 1
                else:
                    results['draws'] += 1
                games_completed += 1
                env_ticks[i] = 0
                # Reset LSTM state for this env
                state_p['lstm_h'][i] = 0
                state_p['lstm_c'][i] = 0
                state_o['lstm_h'][i] = 0
                state_o['lstm_c'][i] = 0

    env.close()
    return results


def run_matches_cross_scheme(player_policy, opponent_policy, num_games,
                              player_obs_scheme=0, opponent_obs_scheme=0,
                              player_hidden_size=128, opponent_hidden_size=128,
                              num_envs=16, device='cuda', max_ticks=6000):
    """Run games between policies with DIFFERENT observation schemes.

    The env is created with the player's obs_scheme. The C code is told to
    compute opponent observations using opponent_obs_scheme via
    vec_set_opponent_obs_scheme(). This is slower than same-scheme matches
    because the opponent observation buffer has a different size.

    Args:
        player_policy: Player neural network policy.
        opponent_policy: Opponent neural network policy.
        num_games: Total games to play.
        player_obs_scheme: Observation scheme for the player.
        opponent_obs_scheme: Observation scheme for the opponent.
        player_hidden_size: Hidden size for player LSTM.
        opponent_hidden_size: Hidden size for opponent LSTM.
        num_envs: Number of parallel environments.
        device: Torch device string.
        max_ticks: Maximum ticks per episode.

    Returns:
        dict with keys: wins, losses, draws (from player perspective).
    """
    from pufferlib.ocean.dogfight import binding

    env = Dogfight(
        num_envs=num_envs,
        render_mode=None,
        obs_scheme=player_obs_scheme,
        curriculum_enabled=1,
        curriculum_randomize=1,
        eval_spawn_mode=2,
        fixed_stage=20,
        max_steps=max_ticks,
    )
    binding.vec_enable_opponent_override(env.c_envs, 1)
    # Tell C to compute opponent obs with opponent's scheme
    binding.vec_set_opponent_obs_scheme(env.c_envs, opponent_obs_scheme)

    results = {'wins': 0, 'losses': 0, 'draws': 0}
    games_completed = 0

    state_p = {'lstm_h': torch.zeros(num_envs, player_hidden_size, device=device),
                'lstm_c': torch.zeros(num_envs, player_hidden_size, device=device)}
    state_o = {'lstm_h': torch.zeros(num_envs, opponent_hidden_size, device=device),
                'lstm_c': torch.zeros(num_envs, opponent_hidden_size, device=device)}

    obs, _ = env.reset()

    while games_completed < num_games:
        obs_tensor = torch.as_tensor(obs, device=device)

        with torch.no_grad():
            logits_p, _ = player_policy.forward_eval(obs_tensor, state=state_p)
            action_p = logits_p.sample()
            action_p_np = action_p.cpu().numpy().astype(np.float32)
            action_p_np = np.clip(action_p_np, -1, 1)

        # Opponent observations (computed with opponent_obs_scheme by C)
        obs_opp = binding.vec_get_opponent_observations(env.c_envs)
        obs_opp = torch.as_tensor(obs_opp, device=device)
        if torch.isnan(obs_opp).any():
            obs_opp = torch.nan_to_num(obs_opp, nan=0.0)

        with torch.no_grad():
            logits_o, _ = opponent_policy.forward_eval(obs_opp, state=state_o)
            action_o = logits_o.sample()
            action_o_np = action_o.cpu().numpy().astype(np.float32)
            action_o_np = np.clip(action_o_np, -1, 1)

        binding.vec_set_opponent_actions(env.c_envs, action_o_np)
        obs, reward, terminal, truncation, info = env.step(action_p_np)

        done_mask = terminal | truncation
        for i in range(num_envs):
            if done_mask[i] and games_completed < num_games:
                r = reward[i]
                if r > 0.1:
                    results['wins'] += 1
                elif r < -0.1:
                    results['losses'] += 1
                else:
                    results['draws'] += 1
                games_completed += 1
                state_p['lstm_h'][i] = 0
                state_p['lstm_c'][i] = 0
                state_o['lstm_h'][i] = 0
                state_o['lstm_c'][i] = 0

    env.close()
    return results


_worker_policies = {}  # Global dict for pool workers: idx -> loaded policy


def _init_worker(policy_infos, device):
    """Pool initializer: load all policies once per worker process.

    Args:
        policy_infos: List of (idx, model_path, hidden_size, obs_scheme) tuples.
        device: Torch device string.
    """
    global _worker_policies
    from pufferlib.ocean.dogfight.dogfight import Dogfight

    # Group by obs_scheme to share tmp_env
    by_scheme = {}
    for idx, path, hs, scheme in policy_infos:
        by_scheme.setdefault(scheme, []).append((idx, path, hs))

    for scheme, entries in by_scheme.items():
        tmp_env = Dogfight(
            num_envs=1, render_mode=None, obs_scheme=scheme,
            curriculum_enabled=1, fixed_stage=20, max_steps=6000,
        )
        for idx, path, hs in entries:
            _worker_policies[idx] = load_policy_from_path(
                path, tmp_env, device, hidden_size=hs)
        tmp_env.close()


def _run_pair_worker(args):
    """Worker function for multiprocessing pool.

    Uses pre-loaded policies from _worker_policies (set by _init_worker).

    Args:
        args: Tuple of (i, j, obs_scheme, num_envs, games, device)

    Returns:
        Tuple of (i, j, result_dict) where result_dict has wins/losses/draws.
    """
    i, j, obs_scheme, num_envs, games, device = args
    hs = _worker_policies[i].hidden_size

    result = run_matches_vectorized(
        _worker_policies[i], _worker_policies[j], games,
        obs_scheme=obs_scheme, hidden_size=hs, num_envs=num_envs,
        device=device)

    return (i, j, result)


def run_league_tournament(policies, league_dir, games_per_pair=30, num_envs=64,
                          device='cuda', num_workers=None):
    """Round-robin tournament for league policies.

    Groups policies by obs_scheme, runs within-scheme round-robins using
    vectorized match running with multiprocessing across pairs, assembles
    combined win matrix, computes ratings via iterative MAP estimation.

    Args:
        policies: List of PolicyEntry objects (from league_manifest).
        league_dir: Base directory for resolving relative model paths.
        games_per_pair: Games per ordered pair (A vs B, then B vs A).
        num_envs: Number of parallel environments for vectorized eval.
        device: Torch device string.
        num_workers: Number of parallel worker processes (default 1 = serial).
            Values >1 use multiprocessing, which helps with multi-GPU or
            very large leagues but adds overhead on single-GPU.

    Returns:
        (win_matrix_dict, ratings_dict) where:
            win_matrix_dict = {labels: [...], data: [[...]]}
            ratings_dict = {policy_id: rating}
    """
    n = len(policies)
    if n < 2:
        log('[EVAL] error=need_at_least_2_policies')
        return {'labels': [], 'data': []}, {}

    labels = [p.id for p in policies]
    # Win rate matrix: wins[i][j] = fraction of games i won against j
    wins = [[0] * n for _ in range(n)]
    draws = [[0] * n for _ in range(n)]

    # Group by obs_scheme
    scheme_groups = {}
    for idx, p in enumerate(policies):
        scheme_groups.setdefault(p.obs_scheme, []).append(idx)

    # Determine worker count (default serial — single GPU can't parallelize well)
    if num_workers is None:
        num_workers = 1

    # Build policy info list for worker initialization and pair list
    policy_infos = []  # (idx, model_path, hidden_size, obs_scheme)
    all_pairs = []
    for scheme, indices in scheme_groups.items():
        if len(indices) < 2:
            continue

        num_pairs = len(indices) * (len(indices) - 1)
        log(f'[EVAL] scheme={scheme} policies={len(indices)} pairs={num_pairs}')

        for idx in indices:
            p = policies[idx]
            policy_infos.append((
                idx, os.path.join(league_dir, p.model_path),
                p.hidden_size, scheme,
            ))

        for i in indices:
            for j in indices:
                if i == j:
                    continue
                all_pairs.append((i, j, scheme, num_envs, games_per_pair, device))

    # Run pairs: multiprocessing if workers > 1, else serial
    if num_workers > 1 and len(all_pairs) > 1:
        log(f'[EVAL] mode=parallel pairs={len(all_pairs)} workers={num_workers}')
        # Use spawn context (safe with CUDA, but has import overhead).
        # Note: multiprocessing helps mainly with multi-GPU setups or
        # very large leagues. For single-GPU, serial is usually faster
        # because workers compete for GPU and each needs CUDA init.
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(num_workers, initializer=_init_worker,
                       initargs=(policy_infos, device)) as pool:
            results_list = pool.map(_run_pair_worker, all_pairs)
    else:
        # Serial: load policies in-process and run directly
        log(f'[EVAL] mode=serial pairs={len(all_pairs)}')
        _init_worker(policy_infos, device)
        results_list = [_run_pair_worker(args) for args in all_pairs]

    # Collect results into win/draw matrices
    for i, j, result in results_list:
        wins[i][j] = result['wins']
        draws[i][j] = result['draws']

        total = result['wins'] + result['losses'] + result['draws']
        wr = result['wins'] / max(total, 1) * 100
        log(f'[MATCH] p1={policies[i].id} p2={policies[j].id} w={result["wins"]} l={result["losses"]} d={result["draws"]} wr={wr:.1f}')

    # Cross-scheme pairs get neutral entries (0.5/0.5 placeholder)
    # They simply won't affect ratings since wins=losses

    # Compute MAP ratings via iterative BT (anchor alphabetically-first policy at 1000)
    # Using a stable anchor (sorted by ID) prevents rating shifts when policy order changes
    anchor_idx = min(range(n), key=lambda i: labels[i])
    elos = [1000.0] * n
    for iteration in range(200):
        max_change = 0.0
        for i in range(n):
            if i == anchor_idx:
                continue  # Skip anchor
            matchups = []
            ref_elos = {}
            has_games = False
            for j in range(n):
                if i == j:
                    continue
                total_games = wins[i][j] + wins[j][i] + draws[i][j]
                if total_games == 0:
                    continue  # No data for cross-scheme pairs
                has_games = True
                tag = f'model_{j}'
                ref_elos[tag] = elos[j]
                matchups.append({
                    'opponent_tag': tag,
                    'wins': wins[i][j],
                    'losses': wins[j][i],
                    'draws': draws[i][j],
                })
            if has_games:
                old_elo = elos[i]
                elos[i] = compute_elo_mle(matchups, ref_elos)
                max_change = max(max_change, abs(elos[i] - old_elo))
        if max_change < 1.0:
            break

    # Build results
    ratings = {labels[i]: elos[i] for i in range(n)}
    win_data = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0.5)
            else:
                total = wins[i][j] + wins[j][i] + draws[i][j]
                if total > 0:
                    row.append((wins[i][j] + 0.5 * draws[i][j]) / total)
                else:
                    row.append(0.5)  # No data
        win_data.append(row)

    win_matrix = {'labels': labels, 'data': win_data}

    # Log ratings
    for rank, (pid, rating) in enumerate(sorted(ratings.items(), key=lambda x: -x[1])):
        log(f'[RATING] policy={pid} rating={rating:.0f} rank={rank+1}')

    return win_matrix, ratings


def compute_elo_mle(matchup_results, reference_elos, prior_rating=1000.0,
                    prior_strength=2.0):
    """Compute MAP Elo rating given results against known-rated opponents.

    Uses bisection search to find the rating R that maximizes the
    log-posterior (log-likelihood + Gaussian prior) under the standard
    logistic Elo model:
        P(win) = 1 / (1 + 10^((R_opp - R) / 400))

    The Gaussian prior with strength=2.0 acts as 2 virtual games at 50%
    against a 1000-rated opponent, preventing extreme divergence when
    real game data is sparse.

    Args:
        matchup_results: list of dicts, each with keys:
            - opponent_tag: str
            - wins: int
            - losses: int
            - draws: int (counted as 0.5 win + 0.5 loss)
        reference_elos: dict mapping opponent_tag -> Elo rating.
        prior_rating: Center of Gaussian prior (default 1000).
        prior_strength: Strength of prior in virtual games (default 2.0).

    Returns:
        float: MAP Elo rating for the candidate.
    """
    def log_posterior(r_candidate):
        ll = 0.0
        for m in matchup_results:
            r_opp = reference_elos[m['opponent_tag']]
            expected = 1.0 / (1.0 + 10.0 ** ((r_opp - r_candidate) / 400.0))
            # Clamp to avoid log(0)
            expected = max(min(expected, 0.9999), 0.0001)

            w = m['wins'] + 0.5 * m['draws']
            l = m['losses'] + 0.5 * m['draws']
            if w > 0:
                ll += w * math.log(expected)
            if l > 0:
                ll += l * math.log(1.0 - expected)

        # Gaussian prior: -strength * ((R - prior) / 400)^2
        # This pulls ratings toward prior_rating, preventing divergence
        ll -= prior_strength * ((r_candidate - prior_rating) / 400.0) ** 2
        return ll

    # Bisection search over candidate rating
    lo, hi = 100.0, 2000.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        eps = 0.5
        if log_posterior(mid + eps) > log_posterior(mid - eps):
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def run_benchmark_eval(model_path, reference_opponents, games_per_matchup=20,
                       obs_scheme=0, device='cuda'):
    """Run full benchmark evaluation against reference opponents.

    Args:
        model_path: Path to the model to evaluate.
        reference_opponents: List of opponent dicts from manifest.json.
        games_per_matchup: Number of games per opponent.
        obs_scheme: Observation scheme (must match training).
        device: Torch device string.

    Returns:
        dict with keys: elo, matchups, total_wins, total_losses, total_draws,
                        eval_time_seconds.
    """
    from pufferlib.ocean.dogfight import binding

    start_time = time.time()

    matchup_results = []
    reference_elos = {}
    total_wins = 0
    total_losses = 0
    total_draws = 0

    for opp in reference_opponents:
        tag = opp['tag']
        opp_type = opp['type']
        opp_elo = opp['elo']
        reference_elos[tag] = opp_elo

        if opp_type == 'autopilot':
            # Create env with fixed stage, no opponent override (autopilot controls opponent)
            stage = opp.get('stage', 20)
            env = Dogfight(
                num_envs=1,
                render_mode=None,
                obs_scheme=obs_scheme,
                curriculum_enabled=1,
                fixed_stage=stage,
                max_steps=6000,
            )
            player_policy = load_policy_from_path(model_path, env, device)
            opponent_policy = None

        elif opp_type == 'neural':
            # Create env with opponent override enabled
            opp_obs_scheme = opp.get('obs_scheme', obs_scheme)
            env = Dogfight(
                num_envs=1,
                render_mode=None,
                obs_scheme=opp_obs_scheme,
                curriculum_enabled=1,
                curriculum_randomize=1,
                eval_spawn_mode=2,
                fixed_stage=20,
                max_steps=6000,
            )
            binding.vec_enable_opponent_override(env.c_envs, 1)
            player_policy = load_policy_from_path(model_path, env, device)
            # Load neural opponent from reference directory
            ref_dir = os.path.dirname(model_path)  # Will be overridden by caller
            opp_path = opp.get('path', '')
            if not os.path.isabs(opp_path):
                # Relative to manifest directory — caller should resolve this
                pass
            opponent_policy = load_policy_from_path(opp_path, env, device)

        else:
            log(f'[ERROR] phase=eval msg="Unknown opponent type: {opp_type}, skipping {tag}"')
            continue

        # Run matches
        results = run_matches(env, player_policy, opponent_policy, games_per_matchup, device)
        env.close()

        matchup_results.append({
            'opponent_tag': tag,
            'wins': results['wins'],
            'losses': results['losses'],
            'draws': results['draws'],
        })

        total_wins += results['wins']
        total_losses += results['losses']
        total_draws += results['draws']

        wr = 100 * results['wins'] / max(games_per_matchup, 1)
        log(f'[MATCH] p1=candidate p2={tag} w={results["wins"]} l={results["losses"]} d={results["draws"]} wr={wr:.1f} elo_opp={opp_elo}')

    # Compute MLE Elo
    elo = compute_elo_mle(matchup_results, reference_elos)
    eval_time = time.time() - start_time

    return {
        'elo': elo,
        'matchups': matchup_results,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'total_draws': total_draws,
        'eval_time_seconds': eval_time,
    }


def load_manifest(reference_dir):
    """Load reference opponent manifest from directory.

    Args:
        reference_dir: Path to directory containing manifest.json.

    Returns:
        dict: Manifest data, or None if not found.
    """
    manifest_path = os.path.join(reference_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        log(f'[ERROR] phase=eval msg="No manifest found at {manifest_path}"')
        return None

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Resolve relative neural opponent paths
    for opp in manifest.get('opponents', []):
        if opp.get('type') == 'neural' and 'path' in opp:
            if not os.path.isabs(opp['path']):
                opp['path'] = os.path.join(reference_dir, opp['path'])

    return manifest


def save_as_reference(model_path, reference_dir, tag, elo, obs_scheme=0):
    """Save a model as a reference opponent.

    Copies the model file to reference_dir and updates manifest.json.

    Args:
        model_path: Path to the source model .pt file.
        reference_dir: Path to reference opponents directory.
        tag: Tag for this reference (e.g., 'ref_v1_strong').
        elo: Elo rating to assign.
        obs_scheme: Observation scheme the model was trained with.
    """
    os.makedirs(reference_dir, exist_ok=True)

    # Copy model file
    dest_filename = f'{tag}.pt'
    dest_path = os.path.join(reference_dir, dest_filename)
    shutil.copy2(model_path, dest_path)
    log(f'[CHECKPOINT] event=bootstrap_copy path={dest_path}')

    # Update manifest
    manifest = load_manifest(reference_dir)
    if manifest is None:
        manifest = {'opponents': [], 'anchor': 'autopilot_s20', 'version': 1}

    # Remove existing entry with same tag
    manifest['opponents'] = [o for o in manifest['opponents'] if o['tag'] != tag]

    # Add new entry
    manifest['opponents'].append({
        'tag': tag,
        'type': 'neural',
        'path': dest_filename,
        'obs_scheme': obs_scheme,
        'elo': elo,
    })

    manifest['version'] = manifest.get('version', 0) + 1

    manifest_path = os.path.join(reference_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    log(f'[CHECKPOINT] event=bootstrap_manifest tag={tag} elo={elo}')


def run_tournament(model_paths, games_per_matchup=20, obs_scheme=0, device='cuda'):
    """Run round-robin tournament between multiple models.

    Each pair plays games_per_matchup games from each side.
    Computes MLE Elo ratings with the first model anchored at 1000.

    Args:
        model_paths: List of paths to model .pt files.
        games_per_matchup: Number of games per ordered pair.
        obs_scheme: Observation scheme.
        device: Torch device string.

    Returns:
        dict mapping model_path -> Elo rating.
    """
    from pufferlib.ocean.dogfight import binding

    n = len(model_paths)
    if n < 2:
        log('[EVAL] error=need_at_least_2_models')
        return {}

    # Win matrix: wins[i][j] = number of times model i beat model j
    wins = [[0] * n for _ in range(n)]
    draws = [[0] * n for _ in range(n)]

    # Create env once (with opponent override)
    env = Dogfight(
        num_envs=1,
        render_mode=None,
        obs_scheme=obs_scheme,
        curriculum_enabled=1,
        curriculum_randomize=1,
        eval_spawn_mode=2,
        fixed_stage=20,
        max_steps=6000,
    )
    binding.vec_enable_opponent_override(env.c_envs, 1)

    # Load all policies
    policies = []
    for path in model_paths:
        policy = load_policy_from_path(path, env, device)
        policies.append(policy)

    # Round-robin
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            results = run_matches(env, policies[i], policies[j], games_per_matchup, device)
            wins[i][j] = results['wins']
            draws[i][j] = results['draws']

            name_i = os.path.basename(model_paths[i])
            name_j = os.path.basename(model_paths[j])
            total = results['wins'] + results['losses'] + results['draws']
            wr = results['wins'] / max(total, 1) * 100
            log(f'[MATCH] p1={name_i} p2={name_j} w={results["wins"]} l={results["losses"]} d={results["draws"]} wr={wr:.1f}')

    env.close()

    # Compute MAP ratings via iterative BT (anchor model 0 at 1000)
    elos = [1000.0] * n
    for iteration in range(200):
        max_change = 0.0
        for i in range(1, n):  # Skip anchor
            matchups = []
            ref_elos = {}
            for j in range(n):
                if i == j:
                    continue
                tag = f'model_{j}'
                ref_elos[tag] = elos[j]
                matchups.append({
                    'opponent_tag': tag,
                    'wins': wins[i][j],
                    'losses': wins[j][i],
                    'draws': draws[i][j],
                })
            old_elo = elos[i]
            elos[i] = compute_elo_mle(matchups, ref_elos)
            max_change = max(max_change, abs(elos[i] - old_elo))
        if max_change < 1.0:
            break

    # Build result
    result = {}
    for i, path in enumerate(model_paths):
        result[path] = elos[i]
        name = os.path.basename(path)
        log(f'[RATING] policy={name} rating={elos[i]:.0f}')

    return result


def main():
    parser = argparse.ArgumentParser(description='Dogfight Elo Evaluation System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # eval subcommand
    eval_parser = subparsers.add_parser('eval', help='Evaluate model against references')
    eval_parser.add_argument('--model', type=str, required=True,
                             help='Path to model .pt file')
    eval_parser.add_argument('--reference-dir', type=str,
                             default='pufferlib/ocean/dogfight/reference_opponents',
                             help='Path to reference opponents directory')
    eval_parser.add_argument('--games', type=int, default=20,
                             help='Games per matchup')
    eval_parser.add_argument('--obs-scheme', type=int, default=0,
                             help='Observation scheme')
    eval_parser.add_argument('--device', type=str, default='cuda',
                             help='Torch device')

    # bootstrap subcommand
    boot_parser = subparsers.add_parser('bootstrap', help='Save model as reference opponent')
    boot_parser.add_argument('--model', type=str, required=True,
                             help='Path to model .pt file')
    boot_parser.add_argument('--tag', type=str, required=True,
                             help='Tag for reference (e.g., ref_v1_strong)')
    boot_parser.add_argument('--elo', type=float, required=True,
                             help='Elo rating to assign')
    boot_parser.add_argument('--obs-scheme', type=int, default=0,
                             help='Observation scheme')
    boot_parser.add_argument('--reference-dir', type=str,
                             default='pufferlib/ocean/dogfight/reference_opponents',
                             help='Path to reference opponents directory')

    # tournament subcommand
    tourn_parser = subparsers.add_parser('tournament', help='Round-robin tournament')
    tourn_parser.add_argument('--models', type=str, nargs='+', required=True,
                              help='Paths to model .pt files')
    tourn_parser.add_argument('--games', type=int, default=20,
                              help='Games per matchup')
    tourn_parser.add_argument('--obs-scheme', type=int, default=0,
                              help='Observation scheme')
    tourn_parser.add_argument('--device', type=str, default='cuda',
                              help='Torch device')

    args = parser.parse_args()

    if args.command in ('eval', 'tournament'):
        init_log('league/logs', f'elo_{args.command}')

    if args.command == 'eval':
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        manifest = load_manifest(args.reference_dir)
        if manifest is None:
            print('No manifest found. Create reference opponents first.')
            return

        result = run_benchmark_eval(
            model_path=args.model,
            reference_opponents=manifest['opponents'],
            games_per_matchup=args.games,
            obs_scheme=args.obs_scheme,
            device=device,
        )
        print(f'\nElo Rating: {result["elo"]:.0f}')
        print(f'Total: {result["total_wins"]}W/{result["total_losses"]}L/{result["total_draws"]}D')
        print(f'Eval time: {result["eval_time_seconds"]:.1f}s')

    elif args.command == 'bootstrap':
        save_as_reference(
            model_path=args.model,
            reference_dir=args.reference_dir,
            tag=args.tag,
            elo=args.elo,
            obs_scheme=args.obs_scheme,
        )
        print(f'Saved {args.model} as reference: {args.tag} (Elo {args.elo})')

    elif args.command == 'tournament':
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        ratings = run_tournament(
            model_paths=args.models,
            games_per_matchup=args.games,
            obs_scheme=args.obs_scheme,
            device=device,
        )
        print('\nFinal Ratings:')
        for path, elo in sorted(ratings.items(), key=lambda x: -x[1]):
            print(f'  {os.path.basename(path)}: {elo:.0f}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

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
import os
import shutil
import time

import numpy as np
import torch

from pufferlib.ocean.dogfight.dogfight import Dogfight, OBS_SIZES


def load_policy_from_path(path, env, device='cuda'):
    """Load policy from .pt file, handling both PuffeRL and CheckpointQueue formats.

    Args:
        path: Path to .pt checkpoint file.
        env: Dogfight environment instance (for policy construction).
        device: Torch device string.

    Returns:
        Policy in eval mode with frozen parameters.
    """
    from pufferlib.models import Default as Policy

    policy = Policy(env, hidden_size=128)
    policy = policy.to(device)

    state_dict = torch.load(path, map_location=device, weights_only=True)

    if isinstance(state_dict, dict) and 'policy_state_dict' in state_dict:
        # CheckpointQueue format
        policy.load_state_dict(state_dict['policy_state_dict'])
    else:
        # PuffeRL raw state_dict format
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith('lstm.') or k.startswith('cell.'):
                continue
            new_k = k.replace('module.', '').replace('policy.', '')
            cleaned[new_k] = v
        policy.load_state_dict(cleaned)

    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False

    return policy


def run_matches(env, player_policy, opponent_policy, num_games, device='cuda'):
    """Run vectorized matches between player and opponent policies.

    For autopilot opponents, opponent_policy is None and the C code handles
    opponent behavior based on the curriculum stage.

    Args:
        env: Dogfight environment (already configured with correct stage/override).
        player_policy: Player neural network policy.
        opponent_policy: Opponent neural network policy, or None for autopilot.
        num_games: Number of games to play.
        device: Torch device string.

    Returns:
        dict with keys: wins, losses, draws (from player perspective).
    """
    from pufferlib.ocean.dogfight import binding

    results = {'wins': 0, 'losses': 0, 'draws': 0}
    games_played = 0
    max_ticks = 6000  # 2 minutes at 50Hz

    while games_played < num_games:
        obs, _ = env.reset()
        done = False
        tick = 0

        while not done and tick < max_ticks:
            obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)

            with torch.no_grad():
                logits_p, _ = player_policy.forward_eval(obs_tensor, state=None)
                action_p = logits_p.sample()
                action_p_np = action_p.cpu().numpy().astype(np.float32)
                action_p_np = np.clip(action_p_np, -1, 1)

            if opponent_policy is not None:
                obs_opp = binding.vec_get_opponent_observations(env.c_envs)
                obs_opp = torch.as_tensor(obs_opp[:1], device=device)
                if torch.isnan(obs_opp).any():
                    obs_opp = torch.nan_to_num(obs_opp, nan=0.0)

                with torch.no_grad():
                    logits_o, _ = opponent_policy.forward_eval(obs_opp, state=None)
                    action_o = logits_o.sample()
                    action_o_np = action_o.cpu().numpy().astype(np.float32)
                    action_o_np = np.clip(action_o_np, -1, 1)

                binding.vec_set_opponent_actions(env.c_envs, action_o_np)

            obs, reward, terminal, truncation, info = env.step(action_p_np.reshape(1, -1))
            done = terminal[0] or truncation[0]
            tick += 1

        # Determine outcome from final reward
        final_reward = reward[0]
        if final_reward > 0.5:
            results['wins'] += 1
        elif final_reward < -0.5:
            results['losses'] += 1
        else:
            results['draws'] += 1

        games_played += 1

    return results


def compute_elo_mle(matchup_results, reference_elos):
    """Compute MLE Elo rating given results against known-rated opponents.

    Uses bisection search to find the rating R that maximizes the
    log-likelihood of observed results under the standard logistic Elo model:
        P(win) = 1 / (1 + 10^((R_opp - R) / 400))

    Args:
        matchup_results: list of dicts, each with keys:
            - opponent_tag: str
            - wins: int
            - losses: int
            - draws: int (counted as 0.5 win + 0.5 loss)
        reference_elos: dict mapping opponent_tag -> Elo rating.

    Returns:
        float: MLE Elo rating for the candidate.
    """
    def log_likelihood(r_candidate):
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
        return ll

    # Bisection search over candidate rating
    lo, hi = 0.0, 3000.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        # Check gradient direction: if increasing R improves likelihood, search higher
        eps = 0.5
        if log_likelihood(mid + eps) > log_likelihood(mid - eps):
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
            print(f'[ELO-EVAL] Unknown opponent type: {opp_type}, skipping {tag}')
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

        win_pct = 100 * results['wins'] / games_per_matchup
        print(f'[ELO-EVAL] vs {tag} (Elo {opp_elo}): '
              f'{results["wins"]}W/{results["losses"]}L/{results["draws"]}D '
              f'({win_pct:.0f}% win rate)')

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
        print(f'[ELO-EVAL] No manifest found at {manifest_path}')
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
    print(f'[ELO-EVAL] Copied model to {dest_path}')

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
    print(f'[ELO-EVAL] Updated manifest: {tag} (Elo {elo})')


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
        print('[ELO-EVAL] Need at least 2 models for tournament')
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
            print(f'[TOURNAMENT] {name_i} vs {name_j}: '
                  f'{results["wins"]}W/{results["losses"]}L/{results["draws"]}D')

    env.close()

    # Compute Elo ratings via iterative MLE
    # Anchor model 0 at 1000
    elos = [1000.0] * n
    for iteration in range(200):
        for i in range(1, n):  # Skip anchor
            # Compute log-likelihood gradient for model i
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
            elos[i] = compute_elo_mle(matchups, ref_elos)

    # Build result
    result = {}
    for i, path in enumerate(model_paths):
        result[path] = elos[i]
        name = os.path.basename(path)
        print(f'[TOURNAMENT] {name}: Elo {elos[i]:.0f}')

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

#!/usr/bin/env python
"""Anchor Evaluation: Evaluate models against a fixed set of reference opponents.

Provides a comparable rating across sweeps by measuring win rates against
permanent anchors (autopilot stages + neural checkpoints). Unlike the internal
"strength" metric, anchor_rating is measured against the same opponents
regardless of which sweep produced the model.

Usage:
    # Evaluate a single model
    python pufferlib/ocean/dogfight/anchor_eval.py \
        --model experiments/puffer_dogfight_abc123.pt \
        --obs-scheme 0

    # Evaluate with more games for tighter confidence intervals
    python pufferlib/ocean/dogfight/anchor_eval.py \
        --model experiments/model.pt --games 50

    # Add a neural anchor from a trained model
    python pufferlib/ocean/dogfight/anchor_eval.py add-anchor \
        --model experiments/best_model.pt \
        --tag df29_best_scheme0 --obs-scheme 0
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from pufferlib.ocean.dogfight.dogfight import Dogfight, OBS_SIZES
from pufferlib.ocean.dogfight.elo_eval import (
    load_policy_from_path,
    run_matches_vectorized,
    run_matches_cross_scheme,
    compute_elo_mle,
)
from pufferlib.ocean.dogfight.collect_from_wandb import (
    infer_hidden_size_from_checkpoint,
    infer_obs_scheme_from_checkpoint,
)

DEFAULT_ANCHOR_DIR = 'pufferlib/ocean/dogfight/reference_opponents'
DEFAULT_GAMES_PER_ANCHOR = 50
DEFAULT_NUM_ENVS = 16


def load_anchor_manifest(anchor_dir):
    """Load anchor manifest from directory.

    Returns list of anchor dicts with resolved paths.
    """
    manifest_path = os.path.join(anchor_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        print(f'[ANCHOR] No manifest at {manifest_path}')
        return []

    with open(manifest_path, 'r') as f:
        data = json.load(f)

    anchors = data.get('anchors', [])

    # Resolve relative paths for neural anchors
    for a in anchors:
        if a['type'] == 'neural' and 'path' in a:
            if not os.path.isabs(a['path']):
                a['path'] = os.path.join(anchor_dir, a['path'])

    return anchors


def _run_vs_autopilot(player_policy, stage, obs_scheme, hidden_size,
                      num_games, num_envs, device, env=None):
    """Run games against an autopilot opponent at a fixed curriculum stage.

    Autopilot opponents don't need a neural network — the C code handles
    their behavior based on the curriculum stage.

    If env is provided, it will be reused (stage set via binding). Otherwise
    a temporary env is created and closed after evaluation.
    """
    from pufferlib.ocean.dogfight import binding

    owns_env = env is None
    if owns_env:
        env = Dogfight(
            num_envs=num_envs,
            render_mode=None,
            obs_scheme=obs_scheme,
            curriculum_enabled=1,
            curriculum_randomize=1,
            eval_spawn_mode=2,
            fixed_stage=stage,
            max_steps=6000,
        )
    else:
        # Reuse pre-created env: update stage
        binding.vec_set_curriculum_target(env.c_envs, float(stage))
        num_envs = env.num_agents

    results = {'wins': 0, 'losses': 0, 'draws': 0}
    games_completed = 0
    env_ticks = np.zeros(num_envs, dtype=np.int32)

    state_p = {
        'lstm_h': torch.zeros(num_envs, hidden_size, device=device),
        'lstm_c': torch.zeros(num_envs, hidden_size, device=device),
    }

    obs, _ = env.reset()

    while games_completed < num_games:
        obs_tensor = torch.as_tensor(obs, device=device)

        with torch.no_grad():
            logits_p, _ = player_policy.forward_eval(obs_tensor, state=state_p)
            action_p = logits_p.sample()
            action_p_np = action_p.cpu().numpy().astype(np.float32)
            action_p_np = np.clip(action_p_np, -1, 1)

        obs, reward, terminal, truncation, info = env.step(action_p_np)
        env_ticks += 1

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
                state_p['lstm_h'][i] = 0
                state_p['lstm_c'][i] = 0

    if owns_env:
        env.close()
    return results


def likelihood_of_superiority(wins, losses):
    """Compute LOS = probability that true win rate > 0.5.

    Uses the normal approximation: LOS = 0.5 * (1 + erf((W-L) / sqrt(2W+2L)))
    """
    n = wins + losses
    if n == 0:
        return 0.5
    return 0.5 * (1.0 + math.erf((wins - losses) / math.sqrt(2.0 * n)))


def evaluate_against_anchors(model_path, obs_scheme=0, anchor_dir=DEFAULT_ANCHOR_DIR,
                              games_per_anchor=DEFAULT_GAMES_PER_ANCHOR,
                              num_envs=DEFAULT_NUM_ENVS, device='cuda',
                              hidden_size=None, eval_env=None):
    """Evaluate a model against the fixed anchor set.

    Args:
        model_path: Path to .pt checkpoint.
        obs_scheme: Observation scheme the model was trained with.
        anchor_dir: Path to directory containing manifest.json and anchor .pt files.
        games_per_anchor: Number of games to play against each anchor.
        num_envs: Number of parallel envs for vectorized evaluation.
        device: Torch device string.
        hidden_size: Hidden size override (inferred from checkpoint if None).
        eval_env: Optional pre-created Dogfight env to reuse for autopilot and
                  neural anchor evaluation. Avoids creating/destroying envs
                  mid-training which can corrupt GPU memory or C-side state.

    Returns:
        dict with keys:
            - Per-anchor entries: {anchor_tag: {'win_rate': float, 'wins': int,
              'losses': int, 'draws': int, 'los': float}}
            - 'anchor_rating': float (Bradley-Terry rating against anchors)
            - 'eval_time': float (seconds)
    """
    start = time.time()

    # Infer hidden_size from checkpoint if not provided
    if hidden_size is None:
        hidden_size = infer_hidden_size_from_checkpoint(model_path)

    # Load anchors
    anchors = load_anchor_manifest(anchor_dir)
    if not anchors:
        print('[ANCHOR] No anchors found')
        return {'anchor_rating': 1000.0, 'eval_time': 0.0}

    # Filter anchors by obs_scheme compatibility
    # Autopilot anchors work with any obs_scheme (obs_scheme=null means universal)
    # Neural anchors can cross-scheme eval (slower) or same-scheme (faster)
    compatible = []
    for a in anchors:
        a_scheme = a.get('obs_scheme')
        if a_scheme is None:
            compatible.append(a)  # Autopilot: universal
        else:
            compatible.append(a)  # Neural: cross-scheme supported via opponent_obs_scheme

    if not compatible:
        print(f'[ANCHOR] No compatible anchors for obs_scheme={obs_scheme}')
        return {'anchor_rating': 1000.0, 'eval_time': 0.0}

    # Load player policy once (use a temporary env for policy construction)
    tmp_env = Dogfight(
        num_envs=1, render_mode=None, obs_scheme=obs_scheme,
        curriculum_enabled=1, fixed_stage=20, max_steps=6000,
    )
    player_policy = load_policy_from_path(model_path, tmp_env, device,
                                           hidden_size=hidden_size)
    tmp_env.close()

    results = {}
    matchup_data = []
    reference_elos = {}

    for anchor in compatible:
        tag = anchor['tag']
        expected_rating = anchor.get('expected_rating', 1000.0)
        reference_elos[tag] = expected_rating

        if anchor['type'] == 'autopilot':
            stage = anchor['stage']
            match_result = _run_vs_autopilot(
                player_policy, stage, obs_scheme, hidden_size,
                games_per_anchor, num_envs, device, env=eval_env)

        elif anchor['type'] == 'neural':
            anchor_path = anchor['path']
            if not os.path.exists(anchor_path):
                print(f'[ANCHOR] Missing neural anchor: {anchor_path}, skipping')
                continue

            anchor_hs = anchor.get('hidden_size', hidden_size)
            anchor_obs_scheme = anchor.get('obs_scheme', obs_scheme)

            # Load opponent policy (needs env with opponent's obs_scheme for construction)
            tmp_env2 = Dogfight(
                num_envs=1, render_mode=None, obs_scheme=anchor_obs_scheme,
                curriculum_enabled=1, fixed_stage=20, max_steps=6000,
            )
            opponent_policy = load_policy_from_path(
                anchor_path, tmp_env2, device, hidden_size=anchor_hs)
            tmp_env2.close()

            if anchor_obs_scheme == obs_scheme:
                # Same scheme: fast path
                match_result = run_matches_vectorized(
                    player_policy, opponent_policy, games_per_anchor,
                    obs_scheme=obs_scheme, hidden_size=hidden_size,
                    num_envs=num_envs, device=device, env=eval_env)
            else:
                # Cross-scheme: uses separate opponent obs computation
                match_result = run_matches_cross_scheme(
                    player_policy, opponent_policy, games_per_anchor,
                    player_obs_scheme=obs_scheme,
                    opponent_obs_scheme=anchor_obs_scheme,
                    player_hidden_size=hidden_size,
                    opponent_hidden_size=anchor_hs,
                    num_envs=num_envs, device=device)
        else:
            continue

        total = match_result['wins'] + match_result['losses'] + match_result['draws']
        win_rate = match_result['wins'] / max(total, 1)
        los = likelihood_of_superiority(match_result['wins'], match_result['losses'])

        results[tag] = {
            'win_rate': win_rate,
            'wins': match_result['wins'],
            'losses': match_result['losses'],
            'draws': match_result['draws'],
            'los': los,
        }

        matchup_data.append({
            'opponent_tag': tag,
            'wins': match_result['wins'],
            'losses': match_result['losses'],
            'draws': match_result['draws'],
        })

        print(f'[ANCHOR] vs {tag}: {match_result["wins"]}W/{match_result["losses"]}L/'
              f'{match_result["draws"]}D  wr={win_rate:.2f}  LOS={los:.3f}')

    # Compute Bradley-Terry rating against anchors
    if matchup_data:
        anchor_rating = compute_elo_mle(matchup_data, reference_elos)
    else:
        anchor_rating = 1000.0

    elapsed = time.time() - start
    results['anchor_rating'] = anchor_rating
    results['eval_time'] = elapsed

    print(f'[ANCHOR] Rating: {anchor_rating:.0f}  ({elapsed:.1f}s)')
    return results


def add_neural_anchor(model_path, tag, obs_scheme, anchor_dir=DEFAULT_ANCHOR_DIR,
                      expected_rating=None, hidden_size=None):
    """Add a neural network model as a permanent anchor.

    Copies the model to the anchors/ subdirectory and updates manifest.json.

    Args:
        model_path: Source .pt file path.
        tag: Unique tag for this anchor (e.g., 'df29_best_scheme0').
        obs_scheme: Observation scheme the model was trained with.
        anchor_dir: Path to reference opponents directory.
        expected_rating: Estimated rating (will be calibrated on first tournament).
        hidden_size: Hidden size override (inferred from checkpoint if None).
    """
    import shutil

    if hidden_size is None:
        hidden_size = infer_hidden_size_from_checkpoint(model_path)

    if expected_rating is None:
        expected_rating = 1000.0

    # Copy model to anchors directory
    anchors_subdir = os.path.join(anchor_dir, 'anchors')
    os.makedirs(anchors_subdir, exist_ok=True)
    dest = os.path.join(anchors_subdir, f'{tag}.pt')
    shutil.copy2(model_path, dest)

    # Update manifest
    manifest_path = os.path.join(anchor_dir, 'manifest.json')
    with open(manifest_path, 'r') as f:
        data = json.load(f)

    # Remove existing entry with same tag
    data['anchors'] = [a for a in data.get('anchors', []) if a['tag'] != tag]

    data['anchors'].append({
        'tag': tag,
        'type': 'neural',
        'path': f'anchors/{tag}.pt',
        'obs_scheme': obs_scheme,
        'hidden_size': hidden_size,
        'expected_rating': expected_rating,
    })

    data['version'] = data.get('version', 1) + 1

    with open(manifest_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f'[ANCHOR] Added neural anchor: {tag} (obs_scheme={obs_scheme}, '
          f'hidden_size={hidden_size}, expected_rating={expected_rating})')
    print(f'[ANCHOR] Saved to {dest}')


def main():
    parser = argparse.ArgumentParser(description='Anchor Evaluation System')
    subparsers = parser.add_subparsers(dest='command')

    # Default: evaluate
    eval_parser = subparsers.add_parser('eval', help='Evaluate model against anchors')
    eval_parser.add_argument('--model', type=str, required=True)
    eval_parser.add_argument('--obs-scheme', type=int, default=0)
    eval_parser.add_argument('--anchor-dir', type=str, default=DEFAULT_ANCHOR_DIR)
    eval_parser.add_argument('--games', type=int, default=DEFAULT_GAMES_PER_ANCHOR)
    eval_parser.add_argument('--num-envs', type=int, default=DEFAULT_NUM_ENVS)
    eval_parser.add_argument('--device', type=str, default='cuda')

    # Add neural anchor
    add_parser = subparsers.add_parser('add-anchor', help='Add a neural anchor')
    add_parser.add_argument('--model', type=str, required=True)
    add_parser.add_argument('--tag', type=str, required=True)
    add_parser.add_argument('--obs-scheme', type=int, required=True)
    add_parser.add_argument('--anchor-dir', type=str, default=DEFAULT_ANCHOR_DIR)
    add_parser.add_argument('--expected-rating', type=float, default=None)

    # List anchors
    subparsers.add_parser('list', help='List all anchors')

    args = parser.parse_args()

    if args.command == 'eval' or args.command is None:
        if args.command is None:
            # Support running without subcommand for backward compat
            parser.print_help()
            return

        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        results = evaluate_against_anchors(
            model_path=args.model,
            obs_scheme=args.obs_scheme,
            anchor_dir=args.anchor_dir,
            games_per_anchor=args.games,
            num_envs=args.num_envs,
            device=device,
        )

        print(f'\nAnchor Rating: {results["anchor_rating"]:.0f}')
        print(f'Eval time: {results["eval_time"]:.1f}s')
        print('\nPer-anchor results:')
        for tag, data in results.items():
            if isinstance(data, dict) and 'win_rate' in data:
                print(f'  {tag:30s}  {data["wins"]:3d}W/{data["losses"]:3d}L/{data["draws"]:3d}D'
                      f'  wr={data["win_rate"]:.2f}  LOS={data["los"]:.3f}')

    elif args.command == 'add-anchor':
        add_neural_anchor(
            model_path=args.model,
            tag=args.tag,
            obs_scheme=args.obs_scheme,
            anchor_dir=args.anchor_dir,
            expected_rating=args.expected_rating,
        )

    elif args.command == 'list':
        anchors = load_anchor_manifest(DEFAULT_ANCHOR_DIR)
        if not anchors:
            print('No anchors configured.')
            return
        print(f'{"Tag":30s}  {"Type":10s}  {"Scheme":8s}  {"Rating":8s}')
        print('-' * 60)
        for a in anchors:
            scheme = str(a.get('obs_scheme', 'any'))
            rating = f'{a.get("expected_rating", "?")}'
            print(f'{a["tag"]:30s}  {a["type"]:10s}  {scheme:8s}  {rating:8s}')


if __name__ == '__main__':
    main()

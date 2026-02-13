#!/usr/bin/env python
"""Collect top-N policies from a W&B project to build a league population.

Downloads model checkpoints, extracts configs, creates frozen gen-0 anchors,
and builds the initial league manifest.

Usage:
    python pufferlib/ocean/dogfight/collect_from_wandb.py \
        --project df27 --top-n 20 --output-dir league/ \
        --metric environment/strength
"""
import argparse
import os
import shutil
from datetime import datetime

import torch


def _get_encoder_shape(path):
    """Get (hidden_size, obs_size) from checkpoint encoder weight.

    The encoder weight has shape [hidden_size, obs_size].
    """
    state_dict = torch.load(path, map_location='cpu', weights_only=True)

    # Handle CheckpointQueue format
    if isinstance(state_dict, dict) and 'policy_state_dict' in state_dict:
        state_dict = state_dict['policy_state_dict']

    # Look for encoder weight (both with and without prefix)
    for key in ['encoder.weight', 'policy.encoder.weight', 'module.encoder.weight']:
        if key in state_dict:
            return state_dict[key].shape[0], state_dict[key].shape[1]

    # Fallback: look for any weight that could be the encoder
    for key, tensor in state_dict.items():
        if 'encoder' in key and 'weight' in key and len(tensor.shape) == 2:
            return tensor.shape[0], tensor.shape[1]

    return 128, None


# Reverse map: obs_size -> obs_scheme
_OBS_SIZE_TO_SCHEME = {17: 0, 23: 1, 26: 2, 22: 3}


def infer_hidden_size_from_checkpoint(path):
    """Infer hidden_size from checkpoint tensor shapes."""
    return _get_encoder_shape(path)[0]


def infer_obs_scheme_from_checkpoint(path):
    """Infer obs_scheme from checkpoint encoder input dimension.

    Returns the obs_scheme int, or None if unrecognized.
    """
    _, obs_size = _get_encoder_shape(path)
    if obs_size is None:
        return None
    return _OBS_SIZE_TO_SCHEME.get(obs_size)


def collect_from_wandb(project, top_n, output_dir, metric='environment/strength',
                       entity=None, min_steps_fraction=0.7):
    """Collect top-N runs from a W&B project.

    Args:
        project: W&B project name (e.g., 'df27')
        top_n: Number of top runs to collect
        output_dir: Directory to save models, anchors, and manifest
        metric: W&B metric key to sort by (descending)
        entity: W&B entity (None = default)
        min_steps_fraction: Only include runs that completed at least this fraction of total_timesteps
    """
    import wandb
    from pufferlib.ocean.dogfight.league_manifest import LeagueManifest, PolicyEntry

    api = wandb.Api()

    # Query runs sorted by metric
    path = f'{entity}/{project}' if entity else project
    print(f'[COLLECT] Querying W&B project: {path}')
    runs = api.runs(path, order=f'-summary_metrics.{metric}')

    # Filter and collect top N
    collected = []
    for run in runs:
        if len(collected) >= top_n:
            break

        # Check run completed reasonably
        summary = run.summary
        if metric not in summary:
            continue

        metric_val = summary[metric]
        if metric_val is None or metric_val == 0:
            continue

        # Check run length
        config = run.config
        total_steps = config.get('train', {}).get('total_timesteps',
                     config.get('total_timesteps', 0))
        agent_steps = summary.get('agent_steps', 0)
        if total_steps > 0 and agent_steps < min_steps_fraction * total_steps:
            print(f'  Skipping {run.name} ({run.id}): only {agent_steps}/{total_steps} steps')
            continue

        collected.append(run)
        print(f'  [{len(collected)}/{top_n}] {run.name} ({run.id}): {metric}={metric_val:.4f}')

    if not collected:
        print('[COLLECT] No qualifying runs found!')
        return None

    print(f'[COLLECT] Collected {len(collected)} runs')

    # Create output directories
    models_dir = os.path.join(output_dir, 'models')
    anchors_dir = os.path.join(output_dir, 'anchors')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(anchors_dir, exist_ok=True)

    # Build manifest
    manifest = LeagueManifest()
    manifest.source_project = project
    manifest.created = datetime.now().isoformat()

    for run in collected:
        config = run.config
        obs_scheme = config.get('env', {}).get('obs_scheme',
                    config.get('obs_scheme', 0))

        # Download model artifact
        model_path = None
        for artifact in run.logged_artifacts():
            if artifact.type == 'model':
                artifact_dir = artifact.download(root=os.path.join(output_dir, 'artifacts'))
                # Find .pt file in artifact
                for f in os.listdir(artifact_dir):
                    if f.endswith('.pt'):
                        model_path = os.path.join(artifact_dir, f)
                        break
                break

        # Fallback: try to find model file from run files
        if model_path is None:
            for f in run.files():
                if f.name.endswith('.pt'):
                    f.download(root=os.path.join(output_dir, 'downloads'), replace=True)
                    model_path = os.path.join(output_dir, 'downloads', f.name)
                    break

        if model_path is None:
            print(f'  WARNING: No model found for {run.name} ({run.id}), skipping')
            continue

        # Infer hidden_size from checkpoint
        hidden_size = config.get('policy', {}).get('hidden_size',
                     config.get('hidden_size', None))
        if hidden_size is None:
            hidden_size = infer_hidden_size_from_checkpoint(model_path)
            print(f'  Inferred hidden_size={hidden_size} from checkpoint')

        # Verify obs_scheme from checkpoint weights (wandb config can lie)
        actual_scheme = infer_obs_scheme_from_checkpoint(model_path)
        if actual_scheme is not None and actual_scheme != obs_scheme:
            print(f'  WARNING: {run.name} ({run.id}): wandb config says obs_scheme={obs_scheme} '
                  f'but checkpoint weights are scheme {actual_scheme} — using {actual_scheme}')
            obs_scheme = actual_scheme

        # Create policy ID
        policy_id = f'scheme{obs_scheme}_run_{run.id}'

        # Copy model to models dir
        model_filename = f'{policy_id}_gen0.pt'
        dest_model = os.path.join(models_dir, model_filename)
        shutil.copy2(model_path, dest_model)

        # Create frozen anchor copy
        anchor_id = f'anchor_{policy_id}_gen0'
        anchor_filename = f'{anchor_id}.pt'
        dest_anchor = os.path.join(anchors_dir, anchor_filename)
        shutil.copy2(model_path, dest_anchor)

        # Extract training config (subset of interesting params)
        train_config = {}
        if 'train' in config:
            for k in ['learning_rate', 'gamma', 'gae_lambda', 'vf_coef',
                       'ent_coef', 'clip_coef', 'max_grad_norm',
                       'prio_alpha', 'prio_beta0',
                       'vtrace_rho_clip', 'vtrace_c_clip',
                       'vf_clip_coef',
                       'adam_beta1', 'adam_beta2', 'adam_eps']:
                if k in config['train']:
                    train_config[k] = config['train'][k]
        if 'env' in config:
            for k in ['reward_aim_scale', 'reward_closing_scale', 'penalty_neg_g',
                       'control_rate_penalty', 'max_steps',
                       'low_altitude_threshold', 'low_altitude_penalty',
                       'recovery_trigger_prob']:
                if k in config['env']:
                    train_config[k] = config['env'][k]

        # Add active policy
        active_entry = PolicyEntry(
            id=policy_id,
            model_path=os.path.relpath(dest_model, output_dir),
            obs_scheme=obs_scheme,
            hidden_size=hidden_size,
            config=train_config,
            status='active',
            generation=0,
            rating=1000.0,
            rating_rd=350.0,
            created_round=0,
            best_checkpoint_path=os.path.relpath(dest_model, output_dir),
            best_checkpoint_generation=0,
            consecutive_rejections=0,
            flagged_for_review=False,
            wandb_run_id=run.id,
            wandb_run_name=run.name,
        )
        manifest.add_policy(active_entry)

        # Add frozen anchor
        anchor_entry = PolicyEntry(
            id=anchor_id,
            model_path=os.path.relpath(dest_anchor, output_dir),
            obs_scheme=obs_scheme,
            hidden_size=hidden_size,
            config=train_config,
            status='frozen',
            generation=0,
            rating=1000.0,
            rating_rd=350.0,
            created_round=0,
            best_checkpoint_path=os.path.relpath(dest_anchor, output_dir),
            best_checkpoint_generation=0,
            source_policy=policy_id,
            frozen_at_round=0,
        )
        manifest.add_policy(anchor_entry)

        print(f'  Saved: {policy_id} (obs_scheme={obs_scheme}, hidden_size={hidden_size})')

    # Save manifest
    manifest_path = os.path.join(output_dir, 'manifest.json')
    manifest.save(manifest_path)
    print(f'[COLLECT] Saved manifest to {manifest_path}')
    print(f'[COLLECT] Active policies: {len(manifest.get_active_policies())}')
    print(f'[COLLECT] Frozen anchors: {len(manifest.get_frozen_anchors())}')

    return manifest


def main():
    parser = argparse.ArgumentParser(description='Collect top-N policies from W&B for league')
    parser.add_argument('--project', type=str, required=True, help='W&B project name')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top runs to collect')
    parser.add_argument('--output-dir', type=str, default='league/', help='Output directory')
    parser.add_argument('--metric', type=str, default='environment/strength',
                        help='W&B metric to sort by (descending)')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--min-steps', type=float, default=0.7,
                        help='Min fraction of total_timesteps completed')
    args = parser.parse_args()

    collect_from_wandb(
        project=args.project,
        top_n=args.top_n,
        output_dir=args.output_dir,
        metric=args.metric,
        entity=args.entity,
        min_steps_fraction=args.min_steps,
    )


if __name__ == '__main__':
    main()

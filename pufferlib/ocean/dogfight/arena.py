#!/usr/bin/env python
"""Arena: Head-to-head policy comparison for Dogfight.

Pits two policies against each other over many episodes to evaluate relative strength.

Usage:
    # Compare two checkpoints
    python pufferlib/ocean/dogfight/arena.py \\
        --policy-a experiments/model_a.pt \\
        --policy-b experiments/model_b.pt \\
        --episodes 100

    # Compare policy against autopilot (no policy-b = use autopilot)
    python pufferlib/ocean/dogfight/arena.py \\
        --policy-a experiments/model.pt \\
        --episodes 100

    # With rendering (slow)
    python pufferlib/ocean/dogfight/arena.py \\
        --policy-a model_a.pt \\
        --policy-b model_b.pt \\
        --render --fps 30

    # At specific curriculum stage
    python pufferlib/ocean/dogfight/arena.py \\
        --policy-a model_a.pt \\
        --policy-b model_b.pt \\
        --stage 15
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from pufferlib.ocean.dogfight.dogfight import Dogfight
from pufferlib.models import Default as Policy


def load_policy(path, env, device='cuda'):
    """Load a policy checkpoint."""
    policy = Policy(env, hidden_size=128)
    policy = policy.to(device)

    state_dict = torch.load(path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        # Skip LSTM keys
        if k.startswith('lstm.') or k.startswith('cell.'):
            continue
        # Strip prefixes
        new_k = k.replace('module.', '').replace('policy.', '')
        cleaned_state_dict[new_k] = v

    policy.load_state_dict(cleaned_state_dict)
    policy.eval()

    return policy


def run_episode(env, policy_a, policy_b, device='cuda', render=False, fps=30):
    """Run a single episode, return winner ('a', 'b', or 'draw')."""
    obs, _ = env.reset()

    from pufferlib.ocean.dogfight import binding

    done = False
    tick = 0
    max_ticks = 6000  # 2 minutes at 50Hz

    while not done and tick < max_ticks:
        # Get observations for both perspectives
        obs_a = torch.as_tensor(obs, device=device).unsqueeze(0)
        obs_b = binding.vec_get_opponent_observations(env.c_envs)
        obs_b = torch.as_tensor(obs_b, device=device)

        # Policy A action (player)
        with torch.no_grad():
            logits_a, _ = policy_a.forward_eval(obs_a, state=None)
            action_a = logits_a.sample()
            action_a = action_a.cpu().numpy().astype(np.float32)
            action_a = np.clip(action_a, -1, 1)

        # Policy B action (opponent)
        if policy_b is not None:
            with torch.no_grad():
                logits_b, _ = policy_b.forward_eval(obs_b, state=None)
                action_b = logits_b.sample()
                action_b = action_b.cpu().numpy().astype(np.float32)
                action_b = np.clip(action_b, -1, 1)

            # Set opponent actions in C code
            binding.vec_set_opponent_actions(env.c_envs, action_b)

        # Step environment
        obs, reward, terminal, truncation, info = env.step(action_a.reshape(1, -1))

        if render:
            env.render()
            time.sleep(1.0 / fps)

        done = terminal[0] or truncation[0]
        tick += 1

    # Determine winner based on reward
    final_reward = reward[0]
    if final_reward > 0.5:
        return 'a'  # Policy A killed opponent
    elif final_reward < -0.5:
        return 'b'  # Policy A was killed (opponent wins)
    else:
        return 'draw'  # Timeout or other


def main():
    parser = argparse.ArgumentParser(description='Dogfight Arena: Head-to-head policy comparison')
    parser.add_argument('--policy-a', type=str, required=True,
                        help='Path to first policy checkpoint')
    parser.add_argument('--policy-b', type=str, default=None,
                        help='Path to second policy checkpoint (omit for autopilot)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to run')
    parser.add_argument('--stage', type=int, default=-1,
                        help='Curriculum stage (-1 for stage 20 AutoAce)')
    parser.add_argument('--obs-scheme', type=int, default=0,
                        help='Observation scheme (must match training)')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes')
    parser.add_argument('--fps', type=int, default=30,
                        help='Render FPS')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for policy inference')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA not available, using CPU')

    # Create environment
    render_mode = 'human' if args.render else None
    env = Dogfight(
        num_envs=1,
        render_mode=render_mode,
        render_fps=args.fps if args.render else None,
        obs_scheme=args.obs_scheme,
        curriculum_enabled=1,
        fixed_stage=args.stage if args.stage >= 0 else 20,  # Default to AutoAce stage
        max_steps=6000,  # 2 minutes per episode
    )

    # Load policies
    print(f'Loading policy A: {args.policy_a}')
    policy_a = load_policy(args.policy_a, env, device)

    if args.policy_b:
        print(f'Loading policy B: {args.policy_b}')
        policy_b = load_policy(args.policy_b, env, device)

        # Enable opponent override for policy-controlled opponent
        from pufferlib.ocean.dogfight import binding
        binding.vec_enable_opponent_override(env.c_envs, 1)
    else:
        print('Policy B: Autopilot (C code)')
        policy_b = None

    # Run matches
    results = {'a': 0, 'b': 0, 'draw': 0}
    episode_times = []

    print(f'\nRunning {args.episodes} episodes...\n')

    for ep in range(args.episodes):
        start_time = time.time()
        winner = run_episode(env, policy_a, policy_b, device, args.render, args.fps)
        elapsed = time.time() - start_time
        episode_times.append(elapsed)

        results[winner] += 1

        # Progress update
        if (ep + 1) % 10 == 0 or (ep + 1) == args.episodes:
            a_wins = results['a']
            b_wins = results['b']
            draws = results['draw']
            total = a_wins + b_wins + draws
            a_pct = 100 * a_wins / total if total > 0 else 0
            b_pct = 100 * b_wins / total if total > 0 else 0
            draw_pct = 100 * draws / total if total > 0 else 0
            avg_time = np.mean(episode_times)

            print(f'Episode {ep+1}/{args.episodes}: '
                  f'A={a_wins} ({a_pct:.1f}%) | '
                  f'B={b_wins} ({b_pct:.1f}%) | '
                  f'Draw={draws} ({draw_pct:.1f}%) | '
                  f'Avg time: {avg_time:.2f}s')

    # Final summary
    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)

    total = args.episodes
    a_wins = results['a']
    b_wins = results['b']
    draws = results['draw']

    print(f'Policy A ({args.policy_a}):')
    print(f'  Wins: {a_wins} ({100*a_wins/total:.1f}%)')

    if args.policy_b:
        print(f'Policy B ({args.policy_b}):')
    else:
        print('Autopilot:')
    print(f'  Wins: {b_wins} ({100*b_wins/total:.1f}%)')

    print(f'Draws: {draws} ({100*draws/total:.1f}%)')

    # Win rate comparison
    if a_wins + b_wins > 0:
        a_vs_b = a_wins / (a_wins + b_wins)
        print(f'\nA vs B win rate: {100*a_vs_b:.1f}%')

        # Confidence interval (Wilson score)
        n = a_wins + b_wins
        p = a_vs_b
        z = 1.96  # 95% confidence
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        spread = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        ci_low = max(0, center - spread)
        ci_high = min(1, center + spread)
        print(f'95% CI: [{100*ci_low:.1f}%, {100*ci_high:.1f}%]')

    env.close()


if __name__ == '__main__':
    main()

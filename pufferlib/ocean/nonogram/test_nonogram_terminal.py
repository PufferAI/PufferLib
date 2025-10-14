#!/usr/bin/env python3
"""Terminal-based Nonogram testing script for AI evaluation"""

import argparse
import glob
import os
import sys
import time
import numpy as np
import random
import torch
from pufferlib.ocean.nonogram.nonogram import Nonogram

MAX_SIZE = 8
MIN_SIZE = 4
MAX_CLUES = 4

def print_board(env, size, show_solution=False):
    """Print the nonogram board to terminal"""
    grid_size = MAX_SIZE * MAX_SIZE
    clue_size = MAX_SIZE * MAX_CLUES

    # Get current board state
    obs = env.observations[0]
    board = obs[:grid_size].reshape(MAX_SIZE, MAX_SIZE)
    row_clues = obs[grid_size:grid_size + clue_size].reshape(MAX_SIZE, MAX_CLUES)
    col_clues = obs[grid_size + clue_size:grid_size + 2*clue_size].reshape(MAX_SIZE, MAX_CLUES)

    if show_solution:
        solutions = env.get_solutions()
        solution = solutions[0].reshape(MAX_SIZE, MAX_SIZE)

    print("\n" + "="*60)

    # Print column clues
    print("   ", end="")
    for c in range(size):
        clue_str = ""
        for clue_idx in range(MAX_CLUES):
            clue = col_clues[c, clue_idx]
            if clue > 0:
                clue_str += str(int(clue))
        print(f"{clue_str:^3}", end=" ")
    print()

    print("   " + "─" * (size * 4))

    # Print rows with row clues
    for r in range(size):
        # Print row clues
        clue_str = ""
        for clue_idx in range(MAX_CLUES):
            clue = row_clues[r, clue_idx]
            if clue > 0:
                clue_str += str(int(clue)) + " "
        print(f"{clue_str:>3}│", end="")

        # Print board cells
        for c in range(size):
            cell = board[r, c]
            if show_solution:
                sol_cell = solution[r, c]
                if cell == 1 and sol_cell == 1:
                    print(" ■ ", end=" ")  # Correct filled
                elif cell == 1 and sol_cell == 0:
                    print(" ✗ ", end=" ")  # Wrong filled
                elif cell == 0 and sol_cell == 1:
                    print(" ? ", end=" ")  # Should be filled
                else:
                    print(" · ", end=" ")  # Correct empty
            else:
                if cell == 1:  # FILLED
                    print(" ■ ", end=" ")
                elif cell == 2:  # PADDING
                    print(" X ", end=" ")
                else:  # EMPTY
                    print(" · ", end=" ")
        print()

    print("="*60)

def print_stats(steps, max_steps, filled, target, reward, total_reward, size):
    """Print game statistics"""
    print(f"Size: {size}x{size} | Steps: {steps}/{max_steps} | Filled: {filled}/{target}")
    print(f"Last Reward: {reward:.3f} | Total Return: {total_reward:.3f}")

def run_evaluation(env, policy, device, use_rnn, num_episodes=100, show_boards=True, delay=0.0):
    """Run multiple episodes and collect statistics"""
    stats = {
        'wins': 0,
        'losses': 0,
        'total_steps': [],
        'total_rewards': [],
        'size_performance': {i: {'wins': 0, 'total': 0} for i in range(MIN_SIZE, MAX_SIZE+1)}
    }

    lstm_h = None
    lstm_c = None
    if use_rnn:
        lstm_h = torch.zeros(1, policy.hidden_size, device=device)
        lstm_c = torch.zeros(1, policy.hidden_size, device=device)

    for episode in range(num_episodes):
        env.reset(seed=random.randint(0, 2**31 - 1))
        size = env.get_size()

        if use_rnn:
            lstm_h.zero_()
            lstm_c.zero_()

        done = False
        steps = 0
        total_reward = 0.0
        max_steps = 4 * MAX_SIZE * MAX_SIZE

        if show_boards:
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes} - Board Size: {size}x{size}")
            print_board(env, size)

        while not done and steps < max_steps:
            # Get observation
            obs_tensor = torch.from_numpy(env.observations[0:1]).float().to(device)

            # Get action from policy
            with torch.no_grad():
                if use_rnn:
                    state = {'lstm_h': lstm_h, 'lstm_c': lstm_c}
                    logits, value = policy.forward_eval(obs_tensor, state)
                    lstm_h = state['lstm_h']
                    lstm_c = state['lstm_c']
                else:
                    logits, value = policy.forward_eval(obs_tensor, None)

                # Sample action
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).cpu().numpy()[0]

            # Take step
            obs, rewards, terminals, truncations, info = env.step(np.array([action]))

            reward = rewards[0]
            total_reward += reward
            steps += 1

            if terminals[0]:
                done = True
                is_win = reward > 0

                if is_win:
                    stats['wins'] += 1
                    stats['size_performance'][size]['wins'] += 1
                else:
                    stats['losses'] += 1

                stats['size_performance'][size]['total'] += 1
                stats['total_steps'].append(steps)
                stats['total_rewards'].append(total_reward)

                if show_boards:
                    print_board(env, size, show_solution=True)
                    result = "✓ SOLVED" if is_win else "✗ FAILED"
                    print(f"\n{result}")
                    print_stats(steps, max_steps, 0, 0, reward, total_reward, size)

                break

            if delay > 0:
                time.sleep(delay)

        if not show_boards and (episode + 1) % 10 == 0:
            win_rate = stats['wins'] / (episode + 1) * 100
            print(f"Episode {episode + 1}/{num_episodes} - Win Rate: {win_rate:.1f}%")

    return stats

def print_summary(stats):
    """Print summary statistics"""
    total_episodes = stats['wins'] + stats['losses']
    win_rate = stats['wins'] / total_episodes * 100 if total_episodes > 0 else 0
    avg_steps = np.mean(stats['total_steps']) if stats['total_steps'] else 0
    avg_reward = np.mean(stats['total_rewards']) if stats['total_rewards'] else 0

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Episodes: {total_episodes}")
    print(f"Wins: {stats['wins']}")
    print(f"Losses: {stats['losses']}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.3f}")

    print("\nPerformance by Board Size:")
    for size in sorted(stats['size_performance'].keys()):
        perf = stats['size_performance'][size]
        if perf['total'] > 0:
            size_win_rate = perf['wins'] / perf['total'] * 100
            print(f"  {size}x{size}: {perf['wins']}/{perf['total']} ({size_win_rate:.1f}%)")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Test Nonogram AI agent (terminal version)')
    parser.add_argument('--model', type=str, default='latest',
                        help='Path to model checkpoint, or "latest" to auto-select')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run model on')
    parser.add_argument('--use-rnn', action='store_true', help='Use RNN policy')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run')
    parser.add_argument('--show-boards', action='store_true', help='Show board visualization')
    parser.add_argument('--delay', type=float, default=0.0, help='Delay between moves (seconds)')
    args = parser.parse_args()

    # Create environment
    env = Nonogram(num_envs=1, min_size=MIN_SIZE, max_size=MAX_SIZE)
    if args.seed is not None:
        env.reset(seed=args.seed)

    # Load model
    print(f"Loading model from {args.model}...")

    # Handle 'latest' keyword
    if args.model == 'latest':
        pattern = "experiments/puffer_nonogram_*/*.pt"
        models = glob.glob(pattern)
        models = [m for m in models if 'trainer_state' not in m]
        if not models:
            raise FileNotFoundError(f"No model files found matching {pattern}")
        args.model = max(models, key=os.path.getctime)
        print(f"Auto-selected latest model: {args.model}")

    # Import policy class
    from pufferlib.ocean.torch import Policy, Recurrent

    # Create policy
    base_policy = Policy(env, hidden_size=128)
    if args.use_rnn:
        policy = Recurrent(env, base_policy, input_size=128, hidden_size=128)
    else:
        policy = base_policy

    # Load weights and auto-detect RNN
    state_dict = torch.load(args.model, map_location=args.device, weights_only=True)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Auto-detect if model uses RNN
    has_rnn = any('lstm' in k or 'cell' in k for k in state_dict.keys())
    if has_rnn and not args.use_rnn:
        print("Auto-detected RNN in model, switching to RNN mode...")
        args.use_rnn = True
        policy = Recurrent(env, base_policy, input_size=128, hidden_size=128)
    elif not has_rnn and args.use_rnn:
        print("Warning: --use-rnn specified but model doesn't have RNN weights, using base policy")
        args.use_rnn = False
        policy = base_policy

    policy.load_state_dict(state_dict)
    policy = policy.to(args.device)
    policy.eval()

    print(f"Model loaded successfully!")
    print(f"Device: {args.device}")
    print(f"Using RNN: {args.use_rnn}")
    print(f"Running {args.episodes} episodes...\n")

    # Run evaluation
    try:
        stats = run_evaluation(
            env, policy, args.device, args.use_rnn,
            num_episodes=args.episodes,
            show_boards=args.show_boards,
            delay=args.delay
        )
        print_summary(stats)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    finally:
        env.close()

if __name__ == '__main__':
    main()

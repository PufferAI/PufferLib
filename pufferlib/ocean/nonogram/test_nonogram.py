#!/usr/bin/env python3
"""Interactive Nonogram testing script with raylib - C version"""

import argparse
import glob
import os
import numpy as np
import random
import torch
from pufferlib.ocean.nonogram.nonogram import Nonogram
from raylib import rl, colors

CELL_SIZE = 40
CLUE_AREA = 120
BOARD_SPACING = 60
FONT_SIZE = 20
MAX_SIZE = 8
MIN_SIZE = 4
MAX_CLUES = 4

def draw_board(board, clues, size, offset_x, offset_y, show_result=False, is_win=False):
    """Draw a nonogram board at the given offset - shows ALL cells including padding"""
    row_clues = clues[0]
    col_clues = clues[1]

    # Draw column clues - show ALL columns including padding
    for clue_row in range(MAX_CLUES):
        for c in range(MAX_SIZE):
            clue = col_clues[c, clue_row] if c < len(col_clues) else 0
            x = offset_x + CLUE_AREA + c * CELL_SIZE + CELL_SIZE // 2
            y = offset_y + clue_row * 20 + 10
            if clue > 0:
                text = str(int(clue)).encode()
                text_width = rl.MeasureText(text, FONT_SIZE)
                rl.DrawText(text, x - text_width // 2, y, FONT_SIZE, colors.RAYWHITE)
            elif c >= size:
                # Show 'P' for padding clue columns
                rl.DrawText(b"P", x - 5, y, FONT_SIZE - 4, colors.GRAY)

    # Draw row clues - show ALL rows including padding
    for r in range(MAX_SIZE):
        clue_x = offset_x + 10
        for clue_idx in range(MAX_CLUES):
            clue = row_clues[r, clue_idx] if r < len(row_clues) else 0
            y = offset_y + CLUE_AREA + r * CELL_SIZE + CELL_SIZE // 2 - FONT_SIZE // 2
            if clue > 0:
                text = str(int(clue)).encode()
                rl.DrawText(text, clue_x, y, FONT_SIZE, colors.RAYWHITE)
                clue_x += rl.MeasureText(text, FONT_SIZE) + 5
            elif r >= size and clue_idx == 0:
                # Show 'P' for padding clue rows
                rl.DrawText(b"P", clue_x, y, FONT_SIZE - 4, colors.GRAY)
                break

    # Draw grid - board uses MAX_SIZE stride, show ALL cells
    grid = board.reshape(MAX_SIZE, MAX_SIZE)

    for r in range(MAX_SIZE):
        for c in range(MAX_SIZE):
            x = offset_x + CLUE_AREA + c * CELL_SIZE
            y = offset_y + CLUE_AREA + r * CELL_SIZE

            # Draw cell background
            if grid[r, c] == 2:  # PADDING
                rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.PURPLE)
            elif grid[r, c] == 1:  # FILLED
                if show_result:
                    # Show result coloring
                    if is_win:
                        rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.GREEN)
                    else:
                        rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.RED)
                else:
                    rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.WHITE)
            else:  # EMPTY
                rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.DARKGRAY)

            # Draw cell border
            rl.DrawRectangleLines(x, y, CELL_SIZE, CELL_SIZE, colors.LIGHTGRAY)

def draw_solution_board(env, size, offset_x, offset_y):
    """Draw the solution board at the given offset - shows ALL cells including padding"""
    grid_size = MAX_SIZE * MAX_SIZE

    # Get solution
    solutions = env.get_solutions()
    solution = solutions[0].reshape(MAX_SIZE, MAX_SIZE)

    # Extract clues from observation
    obs = env.observations[0]
    clue_size = MAX_SIZE * MAX_CLUES
    row_clues = obs[grid_size:grid_size + clue_size].reshape(MAX_SIZE, MAX_CLUES)
    col_clues = obs[grid_size + clue_size:].reshape(MAX_SIZE, MAX_CLUES)

    # Draw column clues - show ALL columns
    for clue_row in range(MAX_CLUES):
        for c in range(MAX_SIZE):
            clue = col_clues[c, clue_row]
            x = offset_x + CLUE_AREA + c * CELL_SIZE + CELL_SIZE // 2
            y = offset_y + clue_row * 20 + 10
            if clue > 0:
                text = str(int(clue)).encode()
                text_width = rl.MeasureText(text, FONT_SIZE)
                rl.DrawText(text, x - text_width // 2, y, FONT_SIZE, colors.RAYWHITE)
            elif c >= size:
                # Show 'P' for padding clue columns
                rl.DrawText(b"P", x - 5, y, FONT_SIZE - 4, colors.GRAY)

    # Draw row clues - show ALL rows
    for r in range(MAX_SIZE):
        clue_x = offset_x + 10
        for clue_idx in range(MAX_CLUES):
            clue = row_clues[r, clue_idx]
            y = offset_y + CLUE_AREA + r * CELL_SIZE + CELL_SIZE // 2 - FONT_SIZE // 2
            if clue > 0:
                text = str(int(clue)).encode()
                rl.DrawText(text, clue_x, y, FONT_SIZE, colors.RAYWHITE)
                clue_x += rl.MeasureText(text, FONT_SIZE) + 5
            elif r >= size and clue_idx == 0:
                # Show 'P' for padding clue rows
                rl.DrawText(b"P", clue_x, y, FONT_SIZE - 4, colors.GRAY)
                break

    # Draw solution grid - show ALL cells
    for r in range(MAX_SIZE):
        for c in range(MAX_SIZE):
            x = offset_x + CLUE_AREA + c * CELL_SIZE
            y = offset_y + CLUE_AREA + r * CELL_SIZE

            # Draw cell background
            if r >= size or c >= size:
                # Padding area
                rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.PURPLE)
            elif solution[r, c] == 1:  # FILLED
                rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.GREEN)
            else:  # EMPTY
                rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.DARKGRAY)

            # Draw cell border
            rl.DrawRectangleLines(x, y, CELL_SIZE, CELL_SIZE, colors.LIGHTGRAY)

def main():
    parser = argparse.ArgumentParser(description='Test Nonogram environment')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint, or "latest" to auto-select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run model on')
    parser.add_argument('--use-rnn', action='store_true', help='Use RNN policy')
    args = parser.parse_args()

    # Create environment with single instance for interactive play
    env = Nonogram(num_envs=1, min_size=MIN_SIZE, max_size=MAX_SIZE)
    env.reset(seed=args.seed)

    # Load model if specified
    policy = None
    lstm_h = None
    lstm_c = None
    if args.model:
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
            lstm_h = torch.zeros(1, policy.hidden_size, device=args.device)
            lstm_c = torch.zeros(1, policy.hidden_size, device=args.device)
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
            lstm_h = torch.zeros(1, policy.hidden_size, device=args.device)
            lstm_c = torch.zeros(1, policy.hidden_size, device=args.device)
        elif not has_rnn and args.use_rnn:
            print("Warning: --use-rnn specified but model doesn't have RNN weights, using base policy")
            args.use_rnn = False
            policy = base_policy

        policy.load_state_dict(state_dict)
        policy = policy.to(args.device)
        policy.eval()
        print(f"Model loaded successfully!")
        print(f"Using device: {args.device}")
        print(f"Using RNN: {args.use_rnn}")

    # Always size the window for maximum grid size
    board_width = CLUE_AREA + MAX_SIZE * CELL_SIZE
    board_height = CLUE_AREA + MAX_SIZE * CELL_SIZE

    screen_width = board_width * 2 + BOARD_SPACING + 40
    screen_height = board_height + 140

    rl.InitWindow(screen_width, screen_height, b"Nonogram - Variable Difficulty")
    rl.SetTargetFPS(60)

    game_over = False
    is_win = False
    message = ""
    steps_taken = 0
    total_reward = 0.0
    last_reward = 0.0

    # Get actual board size from environment
    size = env.get_size()
    max_steps = 4 * size * size

    # Maintain a display board that we control - always full grid
    grid_size = MAX_SIZE * MAX_SIZE
    display_board = env.observations[0, :grid_size].copy()

    # Extract clues
    obs = env.observations[0]
    clue_size = MAX_SIZE * MAX_CLUES
    row_clues = obs[grid_size:grid_size + clue_size].reshape(MAX_SIZE, MAX_CLUES)
    col_clues = obs[grid_size + clue_size:].reshape(MAX_SIZE, MAX_CLUES)
    clues = (row_clues, col_clues)

    # Auto-play mode settings
    auto_play = policy is not None
    auto_play_delay = 0.05  # seconds between AI moves
    last_auto_play_time = 0

    while not rl.WindowShouldClose():
        # Update display board from current env state when game is active
        if not game_over:
            # Always copy full grid
            display_board[:] = env.observations[0, :grid_size]

        # Toggle auto-play with SPACE (only if model is loaded)
        if policy is not None and rl.IsKeyPressed(rl.KEY_SPACE):
            auto_play = not auto_play
            message = "Auto-play: " + ("ON" if auto_play else "OFF")

        # Auto-play with AI
        if auto_play and policy is not None and not game_over:
            import time
            current_time = time.time()
            if current_time - last_auto_play_time >= auto_play_delay:
                last_auto_play_time = current_time

                # Get observation and convert to tensor
                obs_tensor = torch.from_numpy(env.observations[0:1]).float().to(args.device)

                # Get action from policy
                with torch.no_grad():
                    if args.use_rnn:
                        state = {'lstm_h': lstm_h, 'lstm_c': lstm_c}
                        logits, value = policy.forward_eval(obs_tensor, state)
                        lstm_h = state['lstm_h']
                        lstm_c = state['lstm_c']
                    else:
                        logits, value = policy.forward_eval(obs_tensor, None)

                    # Sample action
                    if isinstance(logits, torch.distributions.Normal):
                        action = logits.mean
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        action = torch.argmax(probs, dim=-1)

                    action = action.cpu().numpy()[0]

                # Take step
                obs, rewards, terminals, truncations, info = env.step(np.array([action]))

                # Update rewards
                last_reward = rewards[0]
                total_reward += rewards[0]

                # Check for game end
                if terminals[0]:
                    game_over = True
                    is_win = rewards[0] > 0
                    if is_win:
                        message = "AI Solved! Press R to play again"
                    else:
                        message = "AI Failed - Timeout! Press R to play again"
                    # Reset LSTM state
                    if args.use_rnn:
                        lstm_h.zero_()
                        lstm_c.zero_()
                elif rewards[0] < 0:
                    message = f"AI Invalid move at action {action}!"
                    steps_taken += 1
                else:
                    message = ""
                    steps_taken += 1

        # Handle mouse clicks (manual play)
        if not auto_play and not game_over and rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
            mouse_x = rl.GetMouseX()
            mouse_y = rl.GetMouseY()

            # Check if click is in the game board area
            board_x = 20 + CLUE_AREA
            board_y = 60 + CLUE_AREA

            if (board_x <= mouse_x < board_x + size * CELL_SIZE and
                board_y <= mouse_y < board_y + size * CELL_SIZE):

                # Convert mouse position to grid coordinates
                col = (mouse_x - board_x) // CELL_SIZE
                row = (mouse_y - board_y) // CELL_SIZE

                if 0 <= col < size and 0 <= row < size:
                    # Convert to action using MAX_SIZE stride
                    action = row * MAX_SIZE + col

                    # Take step
                    obs, rewards, terminals, truncations, info = env.step(np.array([action]))

                    # Update rewards
                    last_reward = rewards[0]
                    total_reward += rewards[0]

                    # Check for game end
                    if terminals[0]:
                        game_over = True
                        is_win = rewards[0] > 0
                        if is_win:
                            message = "Congratulations! Puzzle Solved! Press R to play again"
                        else:
                            message = "Game Over - Timeout! Press R to play again"
                        # display_board will keep the last state since we stop updating it
                    elif rewards[0] < 0:
                        message = "Invalid move!"
                        steps_taken += 1
                    else:
                        message = ""
                        steps_taken += 1

        # Handle reset
        if rl.IsKeyPressed(rl.KEY_R):
            env.reset(seed=random.randint(0, 2**31 - 1))
            game_over = False
            is_win = False
            message = ""
            steps_taken = 0
            total_reward = 0.0
            last_reward = 0.0

            # Reset LSTM state if using RNN
            if args.use_rnn and lstm_h is not None:
                lstm_h.zero_()
                lstm_c.zero_()

            # Get new board size
            size = env.get_size()
            max_steps = 4 * size * size
            display_board = env.observations[0, :grid_size].copy()

            # Extract new clues
            obs = env.observations[0]
            row_clues = obs[grid_size:grid_size + clue_size].reshape(MAX_SIZE, MAX_CLUES)
            col_clues = obs[grid_size + clue_size:].reshape(MAX_SIZE, MAX_CLUES)
            clues = (row_clues, col_clues)

        # Drawing
        rl.BeginDrawing()
        rl.ClearBackground(colors.BLACK)

        # Draw titles
        rl.DrawText(b"CURRENT BOARD", 20, 20, 24, colors.RAYWHITE)
        solution_x = board_width + BOARD_SPACING + 20
        rl.DrawText(b"SOLUTION", solution_x, 20, 24, colors.RAYWHITE)

        # Draw boards
        draw_board(display_board, clues, size, 20, 60, show_result=game_over, is_win=is_win)
        draw_solution_board(env, size, solution_x, 60)

        # Count filled cells - only count valid area using MAX_SIZE stride
        filled_total = 0
        grid = display_board.reshape(MAX_SIZE, MAX_SIZE)
        for r in range(size):
            for c in range(size):
                if grid[r, c] == 1:
                    filled_total += 1
        target_total = int(row_clues.sum())

        # Draw status
        status_y = board_height + 80
        status = f"Steps: {steps_taken}/{max_steps} | Filled: {filled_total}/{target_total} | Size: {size}x{size}".encode()
        rl.DrawText(status, 20, status_y, 20, colors.RAYWHITE)

        # Draw reward info
        reward_info = f"Last Reward: {last_reward:.3f} | Episode Return: {total_reward:.3f}".encode()
        rl.DrawText(reward_info, 20, status_y + 25, 20, colors.RAYWHITE)

        # Draw message
        if message:
            if "Congratulations" in message:
                color = colors.GREEN
            elif "Invalid" in message:
                color = colors.RED
            elif "Game Over" in message:
                color = colors.ORANGE
            else:
                color = colors.YELLOW
            rl.DrawText(message.encode(), 20, status_y + 55, 20, color)

        # Draw instructions
        if policy is not None:
            mode_text = f"Mode: {'AI' if auto_play else 'MANUAL'} | Press SPACE to toggle | Press R to reset | ESC to quit".encode()
            rl.DrawText(mode_text, 20, status_y + 85, 16, colors.LIGHTGRAY)
        else:
            rl.DrawText(b"Click cells to toggle | Press R to reset | ESC to quit", 20, status_y + 85, 16, colors.LIGHTGRAY)

        rl.EndDrawing()

    rl.CloseWindow()
    env.close()

if __name__ == '__main__':
    main()

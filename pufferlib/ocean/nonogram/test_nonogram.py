#!/usr/bin/env python3
"""Interactive Nonogram testing script with raylib - C version"""

import numpy as np
import random
from pufferlib.ocean.nonogram.nonogram import Nonogram
from raylib import rl, colors

CELL_SIZE = 40
CLUE_AREA = 120
BOARD_SPACING = 60
FONT_SIZE = 20
MAX_SIZE = 8
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
    # Create environment with single instance for interactive play
    env = Nonogram(num_envs=1, min_size=2, max_size=MAX_SIZE)
    env.reset(seed=42)

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

    while not rl.WindowShouldClose():
        # Update display board from current env state when game is active
        if not game_over:
            # Always copy full grid
            display_board[:] = env.observations[0, :grid_size]

        # Handle mouse clicks
        if not game_over and rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
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
        rl.DrawText(b"Click cells to toggle | Press R to reset | ESC to quit", 20, status_y + 85, 16, colors.LIGHTGRAY)

        rl.EndDrawing()

    rl.CloseWindow()
    env.close()

if __name__ == '__main__':
    main()

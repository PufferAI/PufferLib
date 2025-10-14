#!/usr/bin/env python3
"""Interactive Nonogram testing script with raylib - C version"""

import numpy as np
from pufferlib.ocean.nonogram.nonogram import Nonogram
from raylib import rl, colors

CELL_SIZE = 40
CLUE_AREA = 120
BOARD_SPACING = 60
FONT_SIZE = 20

def draw_board(board, clues, size, offset_x, offset_y, show_result=False, is_win=False):
    """Draw a nonogram board at the given offset"""
    max_clues = size // 2

    row_clues = clues[0]
    col_clues = clues[1]

    # Draw column clues
    for clue_row in range(max_clues):
        for c in range(size):
            clue = col_clues[c, clue_row]
            if clue > 0:
                x = offset_x + CLUE_AREA + c * CELL_SIZE + CELL_SIZE // 2
                y = offset_y + clue_row * 20 + 10
                text = str(int(clue)).encode()
                text_width = rl.MeasureText(text, FONT_SIZE)
                rl.DrawText(text, x - text_width // 2, y, FONT_SIZE, colors.RAYWHITE)

    # Draw row clues
    for r in range(size):
        clue_x = offset_x + 10
        for clue_idx in range(max_clues):
            clue = row_clues[r, clue_idx]
            if clue > 0:
                y = offset_y + CLUE_AREA + r * CELL_SIZE + CELL_SIZE // 2 - FONT_SIZE // 2
                text = str(int(clue)).encode()
                rl.DrawText(text, clue_x, y, FONT_SIZE, colors.RAYWHITE)
                clue_x += rl.MeasureText(text, FONT_SIZE) + 5

    # Draw grid
    grid = board.reshape(size, size)

    for r in range(size):
        for c in range(size):
            x = offset_x + CLUE_AREA + c * CELL_SIZE
            y = offset_y + CLUE_AREA + r * CELL_SIZE

            # Draw cell background
            if grid[r, c] == 1:  # FILLED
                if show_result:
                    # Show result coloring
                    if is_win:
                        rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.GREEN)
                    else:
                        rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.RED)
                else:
                    rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.WHITE)
            else:
                rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.DARKGRAY)

            # Draw cell border
            rl.DrawRectangleLines(x, y, CELL_SIZE, CELL_SIZE, colors.LIGHTGRAY)

def draw_solution_board(env, size, offset_x, offset_y):
    """Draw the solution board at the given offset"""
    max_clues = size // 2
    grid_size = 64  # MAX_SIZE * MAX_SIZE

    # Get solution
    solutions = env.get_solutions()
    solution = solutions[0].reshape(8, 8)[:size, :size]

    # Extract clues from observation
    obs = env.observations[0]
    row_clues = obs[grid_size:grid_size + 32].reshape(8, 4)[:size, :max_clues]
    col_clues = obs[grid_size + 32:].reshape(8, 4)[:size, :max_clues]

    # Draw column clues
    for clue_row in range(max_clues):
        for c in range(size):
            clue = col_clues[c, clue_row]
            if clue > 0:
                x = offset_x + CLUE_AREA + c * CELL_SIZE + CELL_SIZE // 2
                y = offset_y + clue_row * 20 + 10
                text = str(int(clue)).encode()
                text_width = rl.MeasureText(text, FONT_SIZE)
                rl.DrawText(text, x - text_width // 2, y, FONT_SIZE, colors.RAYWHITE)

    # Draw row clues
    for r in range(size):
        clue_x = offset_x + 10
        for clue_idx in range(max_clues):
            clue = row_clues[r, clue_idx]
            if clue > 0:
                y = offset_y + CLUE_AREA + r * CELL_SIZE + CELL_SIZE // 2 - FONT_SIZE // 2
                text = str(int(clue)).encode()
                rl.DrawText(text, clue_x, y, FONT_SIZE, colors.RAYWHITE)
                clue_x += rl.MeasureText(text, FONT_SIZE) + 5

    # Draw solution grid
    for r in range(size):
        for c in range(size):
            x = offset_x + CLUE_AREA + c * CELL_SIZE
            y = offset_y + CLUE_AREA + r * CELL_SIZE

            # Draw cell background
            if solution[r, c] == 1:  # FILLED
                rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.GREEN)
            else:
                rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.DARKGRAY)

            # Draw cell border
            rl.DrawRectangleLines(x, y, CELL_SIZE, CELL_SIZE, colors.LIGHTGRAY)

def main():
    # Create environment with single instance for interactive play
    env = Nonogram(num_envs=1, min_size=2, max_size=8)
    env.reset(seed=42)

    max_size = 8
    board_width = CLUE_AREA + max_size * CELL_SIZE
    board_height = CLUE_AREA + max_size * CELL_SIZE

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
    grid_size = size * size
    max_clues = size // 2

    # Maintain a display board that we control
    display_board = env.observations[0, :grid_size].copy()

    # Extract clues
    obs = env.observations[0]
    row_clues = obs[64:64 + 32].reshape(8, 4)[:size, :max_clues]
    col_clues = obs[64 + 32:].reshape(8, 4)[:size, :max_clues]
    clues = (row_clues, col_clues)

    while not rl.WindowShouldClose():
        # Update display board from current env state when game is active
        if not game_over:
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
                    # Convert to action
                    action = row * size + col

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
            env.reset()
            game_over = False
            is_win = False
            message = ""
            steps_taken = 0
            total_reward = 0.0
            last_reward = 0.0

            # Get new board size
            size = env.get_size()
            max_steps = 4 * size * size
            grid_size = size * size
            max_clues = size // 2
            display_board = env.observations[0, :grid_size].copy()

            # Extract new clues
            obs = env.observations[0]
            row_clues = obs[64:64 + 32].reshape(8, 4)[:size, :max_clues]
            col_clues = obs[64 + 32:].reshape(8, 4)[:size, :max_clues]
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

        # Count filled cells
        filled_total = int(display_board.sum())
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

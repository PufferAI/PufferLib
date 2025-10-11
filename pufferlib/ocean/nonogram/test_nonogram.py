#!/usr/bin/env python3
"""Interactive Nonogram testing script with raylib"""

import numpy as np
from nonogram import Nonogram
from raylib import rl, colors

CELL_SIZE = 40
CLUE_AREA = 120
BOARD_SPACING = 60
FONT_SIZE = 20

def draw_board(env, offset_x, offset_y, is_solution=False):
    """Draw a nonogram board at the given offset"""
    max_clues = env.size // 2

    # Draw column clues
    for clue_row in range(max_clues):
        for c in range(env.size):
            clue = env.cols_clues[c, clue_row]
            if clue > 0:
                x = offset_x + CLUE_AREA + c * CELL_SIZE + CELL_SIZE // 2
                y = offset_y + clue_row * 20 + 10
                text = str(clue).encode()
                text_width = rl.MeasureText(text, FONT_SIZE)
                rl.DrawText(text, x - text_width // 2, y, FONT_SIZE, colors.RAYWHITE)

    # Draw row clues
    for r in range(env.size):
        clue_x = offset_x + 10
        for clue_idx in range(max_clues):
            clue = env.rows_clues[r, clue_idx]
            if clue > 0:
                y = offset_y + CLUE_AREA + r * CELL_SIZE + CELL_SIZE // 2 - FONT_SIZE // 2
                text = str(clue).encode()
                rl.DrawText(text, clue_x, y, FONT_SIZE, colors.RAYWHITE)
                clue_x += rl.MeasureText(text, FONT_SIZE) + 5

    # Draw grid
    grid_size = env.size * env.size
    if is_solution:
        grid = env.solution
    else:
        grid = env.observations[0, :grid_size].reshape(env.size, env.size)

    for r in range(env.size):
        for c in range(env.size):
            x = offset_x + CLUE_AREA + c * CELL_SIZE
            y = offset_y + CLUE_AREA + r * CELL_SIZE

            # Draw cell background
            if grid[r, c] == 1:  # FILLED
                if is_solution:
                    rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.GREEN)
                else:
                    rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.WHITE)
            else:
                rl.DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, colors.DARKGRAY)

            # Draw cell border
            rl.DrawRectangleLines(x, y, CELL_SIZE, CELL_SIZE, colors.LIGHTGRAY)

def main():
    env = Nonogram(size=8, seed=42)
    env.reset()

    board_width = CLUE_AREA + env.size * CELL_SIZE
    board_height = CLUE_AREA + env.size * CELL_SIZE

    screen_width = board_width * 2 + BOARD_SPACING + 40
    screen_height = board_height + 140

    rl.InitWindow(screen_width, screen_height, b"Nonogram - Click to Toggle")
    rl.SetTargetFPS(60)

    game_over = False
    message = ""

    while not rl.WindowShouldClose():
        # Handle mouse clicks
        if not game_over and rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
            mouse_x = rl.GetMouseX()
            mouse_y = rl.GetMouseY()

            # Check if click is in the game board area
            board_x = 20 + CLUE_AREA
            board_y = 60 + CLUE_AREA

            if (board_x <= mouse_x < board_x + env.size * CELL_SIZE and
                board_y <= mouse_y < board_y + env.size * CELL_SIZE):

                # Convert mouse position to grid coordinates
                col = (mouse_x - board_x) // CELL_SIZE
                row = (mouse_y - board_y) // CELL_SIZE

                if 0 <= col < env.size and 0 <= row < env.size:
                    # Convert to action
                    action = row * env.size + col

                    # Take step
                    obs, reward, term, trunc, info = env.step([action])

                    # Check for game end
                    if term[0]:
                        game_over = True
                        if env.filled_total == env.target_total:
                            message = "Congratulations! Puzzle Solved!"
                        else:
                            message = "Game Over - Timeout!"
                    elif reward[0] < 0:
                        message = "Invalid move!"
                    else:
                        message = ""

        # Handle reset
        if rl.IsKeyPressed(rl.KEY_R):
            env.reset()
            game_over = False
            message = ""

        # Drawing
        rl.BeginDrawing()
        rl.ClearBackground(colors.BLACK)

        # Draw title
        rl.DrawText(b"CURRENT BOARD", 20, 20, 24, colors.RAYWHITE)
        solution_x = board_width + BOARD_SPACING + 20
        rl.DrawText(b"SOLUTION", solution_x, 20, 24, colors.RAYWHITE)

        # Draw boards
        draw_board(env, 20, 60, is_solution=False)
        draw_board(env, solution_x, 60, is_solution=True)

        # Draw status
        status_y = board_height + 80
        status = f"Steps: {env.steps_taken}/{env.max_steps} | Filled: {env.filled_total}/{env.target_total}".encode()
        rl.DrawText(status, 20, status_y, 20, colors.RAYWHITE)

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
            rl.DrawText(message.encode(), 20, status_y + 30, 20, color)

        # Draw instructions
        rl.DrawText(b"Click cells to toggle | Press R to reset | ESC to quit", 20, status_y + 60, 16, colors.LIGHTGRAY)

        rl.EndDrawing()

    rl.CloseWindow()

if __name__ == '__main__':
    main()

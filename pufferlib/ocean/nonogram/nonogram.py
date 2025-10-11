'''Pure python version of Nonogram, a simple single-agent sample environment.
   Use this as a template for your own envs.
'''

# We only use Gymnasium for their spaces API for compatibility with other libraries.
import gymnasium
import numpy as np
import pufferlib

EMPTY = 0
FILLED = 1

# Rewards
REWARD_WIN = 1.0
REWARD_INVALID_MOVE = -0.01
REWARD_TIMEOUT = -1 
# Inherit from PufferEnv
class Nonogram(pufferlib.PufferEnv):
    # Required keyword arguments: render_mode, buf, seed
    def __init__(self, render_mode=None, size=8, buf=None, seed=0):
        # Required attributes below
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=2,
            shape=(size*size+size*size,), dtype=np.uint8) # grid + clues for rows and cols
        self.single_action_space = gymnasium.spaces.Discrete(size*size)
        self.render_mode = render_mode
        self.num_agents = 1
        # Call super after initializing attributes
        super().__init__(buf)

        # Add anything else you want
        self.size = size
        self.rng = np.random.RandomState(seed)
        max_clues = size // 2
        self.rows_clues = np.zeros((size, max_clues), dtype=np.uint8)
        self.cols_clues = np.zeros((size, max_clues), dtype=np.uint8)
        self.rows_totals = np.zeros(size, dtype=np.uint8)
        self.cols_totals = np.zeros(size, dtype=np.uint8)
        self.rows_target_sum = np.zeros(size, dtype=np.uint8)
        self.cols_target_sum = np.zeros(size, dtype=np.uint8)
        self.rows_max_clue = np.zeros(size, dtype=np.uint8)
        self.cols_max_clue = np.zeros(size, dtype=np.uint8)
        self.rows_num_runs = np.zeros(size, dtype=np.uint8)
        self.cols_num_runs = np.zeros(size, dtype=np.uint8)
        self.filled_total = 0
        self.target_total = 0
        self.steps_taken = 0
        self.max_steps = 5 * size * size

    # All methods below are required with the signatures shown
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Initialize player grid as all EMPTY
        grid_size = self.size * self.size
        self.observations[0, :grid_size] = EMPTY

        # Generate random solution
        self.solution = self.rng.choice(
            [EMPTY, FILLED],
            (self.size, self.size),
            p=[0.5, 0.5]
        ).astype(np.uint8)

        # Calculate clues directly into arrays
        max_clues = self.size // 2

        # Fill clues for rows and columns
        for i in range(self.size):
            # Row clues
            clue_idx = 0
            count = 0
            for j in range(self.size):
                if self.solution[i, j] == FILLED:
                    count += 1
                elif count > 0:
                    self.rows_clues[i, clue_idx] = count
                    clue_idx += 1
                    count = 0
            if count > 0:
                self.rows_clues[i, clue_idx] = count
                clue_idx += 1
            self.rows_num_runs[i] = clue_idx

        for j in range(self.size):
            # Column clues
            clue_idx = 0
            count = 0
            for i in range(self.size):
                if self.solution[i, j] == FILLED:
                    count += 1
                elif count > 0:
                    self.cols_clues[j, clue_idx] = count
                    clue_idx += 1
                    count = 0
            if count > 0:
                self.cols_clues[j, clue_idx] = count
                clue_idx += 1
            self.cols_num_runs[j] = clue_idx

        # Store clues in observation
        clue_size = self.size * max_clues
        self.observations[0, grid_size:grid_size + clue_size] = self.rows_clues.flatten()
        self.observations[0, grid_size + clue_size:] = self.cols_clues.flatten()

        # Calculate max clues, target sums, and reset totals for constraint checking
        self.rows_totals[:] = 0
        self.cols_totals[:] = 0
        self.filled_total = 0
        self.rows_max_clue[:] = np.max(self.rows_clues, axis=1)
        self.cols_max_clue[:] = np.max(self.cols_clues, axis=1)
        self.rows_target_sum[:] = np.sum(self.rows_clues, axis=1)
        self.cols_target_sum[:] = np.sum(self.cols_clues, axis=1)
        self.target_total = np.sum(self.rows_target_sum)

        self.steps_taken = 0
        self.terminals[0] = False
        self.truncations[0] = False

        return self.observations, []

    def get_row_run_length(self, row, col):
        """Get the run length if we fill the cell at (row, col)"""
        row_start = row * self.size
        run_length = 1

        # Count left
        for c in range(col - 1, -1, -1):
            if self.observations[0, row_start + c] == FILLED:
                run_length += 1
            else:
                break

        # Count right
        for c in range(col + 1, self.size):
            if self.observations[0, row_start + c] == FILLED:
                run_length += 1
            else:
                break

        return run_length

    def get_col_run_length(self, row, col):
        """Get the run length if we fill the cell at (row, col)"""
        run_length = 1

        # Count up
        for r in range(row - 1, -1, -1):
            if self.observations[0, r * self.size + col] == FILLED:
                run_length += 1
            else:
                break

        # Count down
        for r in range(row + 1, self.size):
            if self.observations[0, r * self.size + col] == FILLED:
                run_length += 1
            else:
                break

        return run_length

    def check_line_matches(self, line_data, clues, num_runs):
        """Check if runs in line match clues exactly"""
        run_idx = 0
        count = 0

        for val in line_data:
            count += val
            if val == 0 and count > 0:
                if clues[run_idx] != count:
                    return False
                run_idx += 1
                count = 0

        # Check final run
        if count > 0:
            if clues[run_idx] != count:
                return False
            run_idx += 1

        return run_idx == num_runs

    def step(self, actions):
        pos = int(actions[0])
        row = pos // self.size
        col = pos % self.size

        self.terminals[0] = False
        self.rewards[0] = 0

        current = self.observations[0, pos]

        # If toggling on (EMPTY -> FILLED)
        if current == EMPTY:
            # First check: totals equal target - invalid move
            if self.rows_totals[row] == self.rows_target_sum[row] or self.cols_totals[col] == self.cols_target_sum[col]:
                self.rewards[0] = REWARD_INVALID_MOVE
                self.steps_taken += 1
                return self.observations, self.rewards, self.terminals, self.truncations, [{}]

            # Check if filling this cell would create a run longer than max allowed
            if self.get_row_run_length(row, col) > self.rows_max_clue[row]:
                self.rewards[0] = REWARD_INVALID_MOVE
                self.steps_taken += 1
                return self.observations, self.rewards, self.terminals, self.truncations, [{}]

            if self.get_col_run_length(row, col) > self.cols_max_clue[col]:
                self.rewards[0] = REWARD_INVALID_MOVE
                self.steps_taken += 1
                return self.observations, self.rewards, self.terminals, self.truncations, [{}]

            # Second check: if completing row/col, check runs match
            if self.rows_totals[row] == self.rows_target_sum[row] - 1:
                # Temporarily fill to check
                self.observations[0, pos] = FILLED
                row_start = row * self.size
                row_data = self.observations[0, row_start:row_start + self.size]
                if not self.check_line_matches(row_data, self.rows_clues[row], self.rows_num_runs[row]):
                    # Runs don't match - invalid move
                    self.observations[0, pos] = EMPTY
                    self.rewards[0] = REWARD_INVALID_MOVE
                    self.steps_taken += 1
                    return self.observations, self.rewards, self.terminals, self.truncations, [{}]
                self.observations[0, pos] = EMPTY

            if self.cols_totals[col] == self.cols_target_sum[col] - 1:
                # Temporarily fill to check
                self.observations[0, pos] = FILLED
                col_data = self.observations[0, col::self.size][:self.size]
                if not self.check_line_matches(col_data, self.cols_clues[col], self.cols_num_runs[col]):
                    # Runs don't match - invalid move
                    self.observations[0, pos] = EMPTY
                    self.rewards[0] = REWARD_INVALID_MOVE
                    self.steps_taken += 1
                    return self.observations, self.rewards, self.terminals, self.truncations, [{}]
                self.observations[0, pos] = EMPTY

            # Apply toggle
            self.observations[0, pos] = FILLED
            self.rows_totals[row] += 1
            self.cols_totals[col] += 1
            self.filled_total += 1
        else:
            # Toggling off (FILLED -> EMPTY) - always allowed
            self.observations[0, pos] = EMPTY
            self.rows_totals[row] -= 1
            self.cols_totals[col] -= 1
            self.filled_total -= 1

        self.steps_taken += 1

        # Check if solved
        if self.filled_total == self.target_total:
            self.terminals[0] = True
            self.rewards[0] = REWARD_WIN
            return self.observations, self.rewards, self.terminals, self.truncations, [{'solved': True}]

        # Check if ran out of steps
        if self.steps_taken >= self.max_steps:
            self.terminals[0] = True
            self.rewards[0] = REWARD_TIMEOUT
            return self.observations, self.rewards, self.terminals, self.truncations, [{'solved': False}]

        return self.observations, self.rewards, self.terminals, self.truncations, [{}]

    def render(self):
        from raylib import rl, colors

        if not hasattr(self, 'window_initialized'):
            CELL_SIZE = 40
            CLUE_AREA = 120
            BOARD_SPACING = 60

            board_width = CLUE_AREA + self.size * CELL_SIZE
            board_height = CLUE_AREA + self.size * CELL_SIZE

            screen_width = board_width * 2 + BOARD_SPACING + 40
            screen_height = board_height + 140

            rl.InitWindow(screen_width, screen_height, b"Nonogram")
            rl.SetTargetFPS(60)

            self.window_initialized = True
            self.CELL_SIZE = CELL_SIZE
            self.CLUE_AREA = CLUE_AREA
            self.BOARD_SPACING = BOARD_SPACING
            self.FONT_SIZE = 20

        rl.BeginDrawing()
        rl.ClearBackground(colors.BLACK)

        # Draw titles
        rl.DrawText(b"CURRENT BOARD", 20, 20, 24, colors.RAYWHITE)
        board_width = self.CLUE_AREA + self.size * self.CELL_SIZE
        solution_x = board_width + self.BOARD_SPACING + 20
        rl.DrawText(b"SOLUTION", solution_x, 20, 24, colors.RAYWHITE)

        # Draw both boards
        self._draw_board(20, 60, is_solution=False)
        self._draw_board(solution_x, 60, is_solution=True)

        # Draw status
        status_y = board_width + 80
        status = f"Steps: {self.steps_taken}/{self.max_steps} | Filled: {self.filled_total}/{self.target_total}".encode()
        rl.DrawText(status, 20, status_y, 20, colors.RAYWHITE)

        rl.EndDrawing()

    def _draw_board(self, offset_x, offset_y, is_solution=False):
        from raylib import rl, colors

        max_clues = self.size // 2

        # Draw column clues
        for clue_row in range(max_clues):
            for c in range(self.size):
                clue = self.cols_clues[c, clue_row]
                if clue > 0:
                    x = offset_x + self.CLUE_AREA + c * self.CELL_SIZE + self.CELL_SIZE // 2
                    y = offset_y + clue_row * 20 + 10
                    text = str(clue).encode()
                    text_width = rl.MeasureText(text, self.FONT_SIZE)
                    rl.DrawText(text, x - text_width // 2, y, self.FONT_SIZE, colors.RAYWHITE)

        # Draw row clues
        for r in range(self.size):
            clue_x = offset_x + 10
            for clue_idx in range(max_clues):
                clue = self.rows_clues[r, clue_idx]
                if clue > 0:
                    y = offset_y + self.CLUE_AREA + r * self.CELL_SIZE + self.CELL_SIZE // 2 - self.FONT_SIZE // 2
                    text = str(clue).encode()
                    rl.DrawText(text, clue_x, y, self.FONT_SIZE, colors.RAYWHITE)
                    clue_x += rl.MeasureText(text, self.FONT_SIZE) + 5

        # Draw grid
        grid_size = self.size * self.size
        if is_solution:
            grid = self.solution
        else:
            grid = self.observations[0, :grid_size].reshape(self.size, self.size)

        for r in range(self.size):
            for c in range(self.size):
                x = offset_x + self.CLUE_AREA + c * self.CELL_SIZE
                y = offset_y + self.CLUE_AREA + r * self.CELL_SIZE

                # Draw cell background
                if grid[r, c] == FILLED:
                    if is_solution:
                        rl.DrawRectangle(x, y, self.CELL_SIZE, self.CELL_SIZE, colors.GREEN)
                    else:
                        rl.DrawRectangle(x, y, self.CELL_SIZE, self.CELL_SIZE, colors.WHITE)
                else:
                    rl.DrawRectangle(x, y, self.CELL_SIZE, self.CELL_SIZE, colors.DARKGRAY)

                # Draw cell border
                rl.DrawRectangleLines(x, y, self.CELL_SIZE, self.CELL_SIZE, colors.LIGHTGRAY)

    def close(self):
        pass

if __name__ == '__main__':
    env = Nonogram()
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, env.size*env.size, (CACHE, 1))

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    print('Nonogram SPS:', int(steps / (time.time() - start)))

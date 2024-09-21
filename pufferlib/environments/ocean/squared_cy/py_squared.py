import numpy as np
import gymnasium
from .cy_squared_cy import CSquaredCy

class SquaredCyEnv(gymnasium.Env):
    MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]

    def __init__(self, distance_to_target=3, num_targets=1):
        super().__init__()

        self.distance_to_target = distance_to_target
        self.num_targets = num_targets

        self.c_env = CSquaredCy(distance_to_target, num_targets)

        grid_size = self.c_env.get_grid().shape

        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(8)

        self.render_mode = 'ansi'
        self.tick_calc = 0
        self.agent_total_reward = 0

    def reset(self, seed=None):
        if seed is not None:
            self.c_env.seed(seed)

        self.c_env.reset()
        grid = self.c_env.get_grid()
        return grid, {}

    def step(self, action):
        reward, done, info = self.c_env.step(action)
        grid = self.c_env.get_grid()
        self.agent_total_reward += reward

        return grid, reward, done, False, info

    def render(self):
        grid = self.c_env.get_grid()
        grid_rows, grid_cols = grid.shape

        chars = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                val = grid[row, col]
                if val == 1:
                    color = 94
                elif val == -1:
                    color = 91
                else:
                    color = 90
                chars.append(f'\033[{color}m██\033[0m')
            chars.append('\n')

        grid_render = ''.join(chars)

        self.tick_calc += 1
        average_agent_score = self.agent_total_reward * 2 / self.tick_calc

        tick_info = (f"Tick: {self.tick_calc}\n"
                    f"Agent Average Score: {average_agent_score:.4f}\n")

        return grid_render + tick_info


    def close(self):
        pass

def test_performance(distance_to_target=3, num_targets=1, atn_cache=1024, timeout=10):
    import time

    env = SquaredCyEnv(distance_to_target=distance_to_target, num_targets=num_targets)
    env.reset()
    tick = 0
    actions_cache = np.random.randint(0, 8, atn_cache)
    start = time.time()
    while time.time() - start < timeout:
        action = actions_cache[tick % atn_cache]
        env.step(action)
        tick += 1
    elapsed_time = time.time() - start
    sps = tick / elapsed_time
    print(f"SPS: {sps:.2f}")

if __name__ == '__main__':
    test_performance()

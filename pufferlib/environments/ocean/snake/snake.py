'''High-perf many-agent snake. Inspired by snake env from https://github.com/dnbt777'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.environments.ocean.snake.c_snake import CSnake, CMultiSnake

COLORS = np.array([
    [6, 24, 24, 255],     # Background
    [0, 0, 255, 255],     # Food
    [0, 128, 255, 255],   # Corpse
    [128, 128, 128, 255], # Wall
    [255, 0, 0, 255],     # Snake
    [255, 255, 255, 255], # Snake
    [255, 0, 0, 255],     # Snake
    [255, 255, 255, 255], # Snake
], dtype=np.uint8)

ANSI_COLORS = [30, 34, 36, 90, 31, 97, 91, 37]

class Snake(pufferlib.PufferEnv):
    def __init__(self, widths, heights, num_snakes, num_food, vision=15,
            leave_corpse_on_death=True, teleport_at_edge=True, render_mode='ansi'):
        super().__init__()
        self.grids = [np.zeros((h, w), dtype=np.uint8) for h, w in zip(heights, widths)]

        assert isinstance(vision, int)
        assert len(widths) == len(heights) == len(num_snakes) == len(num_food)
        if isinstance(leave_corpse_on_death, bool):
            self.leave_corpse_on_death = len(widths)*[leave_corpse_on_death]
        else:
            assert len(leave_corpse_on_death) == len(widths)

        for w, h in zip(widths, heights):
            assert w >= 2*vision+6 and h >= 2*vision+6, \
                'width and height must be at least 2*vision+6'

        # Note: it is possible to have a snake longer than 10k, which will mess up
        # the environment, but we must allocate this much memory for each snake
        self.max_snake_length = min(1024, max([w*h for h, w in zip(heights, widths)]))
        total_snakes = sum(num_snakes)
        self.snakes = np.zeros((total_snakes, self.max_snake_length, 2), dtype=np.int32) - 1
        self.snake_lengths = np.zeros(total_snakes, dtype=np.int32)
        self.snake_ptrs = np.zeros(total_snakes, dtype=np.int32)
        self.snake_lifetimes = np.zeros(total_snakes, dtype=np.int32)
        self.snake_colors = 4 + np.arange(total_snakes, dtype=np.int32) % 4
        self.num_snakes = num_snakes
        self.num_food = num_food
        self.vision = vision
        self.leave_corpse_on_death = len(widths)*[leave_corpse_on_death]
        self.teleport_at_edge = len(widths)*[teleport_at_edge]

        box = 2 * vision + 1
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(box, box), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(4)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = total_snakes
        self.render_mode = render_mode
        self.emulated = None
        self.done = False
        self.tick = 0

        self.buf = pufferlib.namespace(
            observations = np.zeros(
                (total_snakes, box, box), dtype=np.uint8),
            rewards = np.zeros(total_snakes, dtype=np.float32),
            terminals = np.zeros(total_snakes, dtype=bool),
            truncations = np.zeros(total_snakes, dtype=bool),
            masks = np.ones(total_snakes, dtype=bool),
        )
        self.actions = np.zeros(total_snakes, dtype=np.uint32)
        self.reward_sum = 0

        self.atn = None
        if render_mode == 'human':
            self.client = RaylibClient(80, 45, COLORS.tolist(), tile_size=16)
 
    def reset(self, seed=None):
        self.c_env = CMultiSnake(self.grids, self.snakes, self.buf.observations,
            self.snake_lengths, self.snake_ptrs, self.snake_lifetimes, self.snake_colors,
            self.actions, self.buf.rewards, self.num_snakes, self.num_food, self.vision,
            self.max_snake_length, self.leave_corpse_on_death, self.teleport_at_edge)

        self.c_env.reset()
        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        if self.atn is not None: # Human player
            self.actions[0] = self.atn

        self.c_env.step()

        info = {}
        self.reward_sum += self.buf.rewards.mean()
        if self.tick % 128 == 0:
            info = {
                'snake_length_min': np.min(self.snake_lengths),
                'snake_lifetime_min': np.min(self.snake_lifetimes),
                'snake_length_max': np.max(self.snake_lengths),
                'snake_lifetime_max': np.max(self.snake_lifetimes),
                'snake_length_mean': np.mean(self.snake_lengths),
                'snake_lifetime_mean': np.mean(self.snake_lifetimes),
                'reward': self.reward_sum / 128,
            }
            self.reward_sum = 0

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self, upscale=1):
        grid = self.grids[0]
        height, width = grid.shape
        if self.render_mode == 'human':
            snakes_in_first_env = self.num_snakes[0]
            snake_ptrs = self.snake_ptrs[:snakes_in_first_env]
            agent_positions = self.snakes[np.arange(snakes_in_first_env), snake_ptrs]
            actions = self.actions[:snakes_in_first_env]
            frame, self.atn = self.client.render(grid, agent_positions)
        elif self.render_mode == 'rgb_array':
            frame = COLORS[grid]
            if upscale > 1:
                rescaler = np.ones((upscale, upscale, 1), dtype=np.uint8)
                frame = np.kron(frame, rescaler)
        elif self.render_mode == 'ansi':
            lines = []
            if not self.teleport_at_edge[0]:
                grid = grid[
                    self.vision-1:-self.vision,
                    self.vision-1:-self.vision
                ]

            for line in grid:
                lines.append(''.join([
                    f'\033[{ANSI_COLORS[val]}m██\033[0m' for val in line]))

            frame = '\n'.join(lines)
        else:
            raise ValueError(f'Invalid render mode: {self.render_mode}')

        return frame

class RaylibClient:
    def __init__(self, width, height, asset_map, tile_size=16):
        self.width = width
        self.height = height
        self.asset_map = asset_map
        self.tile_size = tile_size

        from raylib import rl
        rl.InitWindow(width*tile_size, height*tile_size,
            "PufferLib Ray Snake".encode())
        rl.SetTargetFPS(15)
        self.rl = rl

        from cffi import FFI
        self.ffi = FFI()

    def _cdata_to_numpy(self):

        image = self.rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width*height*channels)
        return np.frombuffer(cdata, dtype=np.uint8
            ).reshape((height, width, channels))

    def render(self, grid, agent_positions):
        rl = self.rl
        action = None
        if rl.IsKeyDown(rl.KEY_UP) or rl.IsKeyDown(rl.KEY_W):
            action = 0
        elif rl.IsKeyDown(rl.KEY_DOWN) or rl.IsKeyDown(rl.KEY_S):
            action = 1
        elif rl.IsKeyDown(rl.KEY_LEFT) or rl.IsKeyDown(rl.KEY_A):
            action = 2
        elif rl.IsKeyDown(rl.KEY_RIGHT) or rl.IsKeyDown(rl.KEY_D):
            action = 3

        rl.BeginDrawing()
        rl.ClearBackground(self.asset_map[0])

        ts = self.tile_size
        main_r, main_c = agent_positions[0]
        for i, r in enumerate(range(main_r-self.height//2, main_r+self.height//2+1)):
            for j, c in enumerate(range(main_c-self.width//2, main_c+self.width//2+1)):
                if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
                    continue

                tile = grid[r, c]
                if tile == 0:
                    continue

                rl.DrawRectangle(j*ts, i*ts, ts, ts, self.asset_map[tile])

        #rl.DrawRectangle(64, 64, 256, 256, (0, 0, 255, 255))

        rl.EndDrawing()
        return self._cdata_to_numpy()[:, :, :3], action

def test_performance(timeout=10, atn_cache=1024):
    actions = np.random.randint(0, 4, (atn_cache, env.num_snakes[0]))
    env = Snake()
    env.reset()
    tick = 0

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', env.num_snakes[0] * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()

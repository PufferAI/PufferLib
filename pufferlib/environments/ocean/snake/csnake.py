'''High-perf many-agent snake. Inspired by snake env from https://github.com/dnbt777'''

import numpy as np
import gymnasium

from raylib import rl

import pufferlib
from pufferlib.environments.ocean.snake.cy_snake import Snake as CSnake
from pufferlib.environments.ocean import render

class Snake(pufferlib.PufferEnv):
    def __init__(self, widths=[2560], heights=[1440], num_snakes=[4096],
            num_food=[65536], vision=5, leave_corpse_on_death=True,
            reward_food=0.1, reward_corpse=0.1, reward_death=-1.0,
            report_interval=128,  max_snake_length=1024,
            render_mode='rgb_array'):

        assert isinstance(vision, int)
        if isinstance(leave_corpse_on_death, bool):
            leave_corpse_on_death = len(widths)*[leave_corpse_on_death]

        assert (len(widths) == len(heights) == len(num_snakes)
            == len(num_food) == len(leave_corpse_on_death))

        for w, h in zip(widths, heights):
            assert w >= 2*vision+2 and h >= 2*vision+2, \
                'width and height must be at least 2*vision+2'

        total_snakes = sum(num_snakes)
        max_area = max([w*h for h, w in zip(heights, widths)])
        self.max_snake_length = min(max_snake_length, max_area)
        snake_shape = (total_snakes, self.max_snake_length, 2)

        self.grids = [np.zeros((h, w), dtype=np.uint8) for h, w in zip(heights, widths)]
        self.snakes = -1 + np.zeros(snake_shape, dtype=np.int32)
        self.snake_lengths = np.zeros(total_snakes, dtype=np.int32)
        self.snake_ptrs = np.zeros(total_snakes, dtype=np.int32)
        self.snake_lifetimes = np.zeros(total_snakes, dtype=np.int32)
        self.snake_colors = 4 + np.arange(total_snakes, dtype=np.int32) % 4
        self.num_snakes = num_snakes
        self.num_food = num_food
        self.vision = vision
        self.leave_corpse_on_death = leave_corpse_on_death
        self.reward_food = reward_food
        self.reward_corpse = reward_corpse
        self.reward_death = reward_death
        self.report_interval = report_interval

        # This block required by advanced PufferLib env spec
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
        self.tick = 0
        self.atn = None

        if render_mode == 'ansi':
            self.client = render.AnsiRender()
        elif render_mode == 'rgb_array':
            self.client = render.RGBArrayRender()
        elif render_mode == 'raylib':
            self.client = render.GridRender(widths[0], heights[0])
        elif render_mode == 'human':
            colors = [(r, g, b, 255) for r, g, b in render.COLORS]
            self.client = RaylibClient(80, 45, colors)
        else:
            raise ValueError(f'Invalid render mode: {render_mode}')

    def reset(self, seed=None):
        ptr = end = 0
        self.c_envs = []
        for i in range(len(self.grids)):
            end += self.num_snakes[i]
            self.c_envs.append(CSnake(self.grids[i], self.snakes[ptr:end],
                self.buf.observations[ptr:end], self.snake_lengths[ptr:end],
                self.snake_ptrs[ptr:end], self.snake_lifetimes[ptr:end],
                self.snake_colors[ptr:end], self.actions[ptr:end],
                self.buf.rewards[ptr:end], self.num_food[i], self.vision,
                self.max_snake_length, self.leave_corpse_on_death[i],
                self.reward_food, self.reward_corpse, self.reward_death))
            self.c_envs[i].reset()
            ptr = end

        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        if self.atn is not None: # Human player
            self.actions[0] = self.atn

        for c in self.c_envs:
            c.step()

        info = {}
        self.reward_sum += self.buf.rewards.mean()
        if self.tick % self.report_interval == 0:
            info = {
                'snake_length_min': np.min(self.snake_lengths),
                'snake_lifetime_min': np.min(self.snake_lifetimes),
                'snake_length_max': np.max(self.snake_lengths),
                'snake_lifetime_max': np.max(self.snake_lifetimes),
                'snake_length_mean': np.mean(self.snake_lengths),
                'snake_lifetime_mean': np.mean(self.snake_lifetimes),
                'reward': self.reward_sum / self.report_interval,
            }
            self.reward_sum = 0

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self):
        grid = self.grids[0]
        height, width = grid.shape
        v = self.vision
        crop = grid[v:-v-1, v:-v-1]
        if self.render_mode == 'ansi':
            return self.client.render(crop)
        elif self.render_mode == 'rgb_array':
            return self.client.render(crop)
        elif self.render_mode == 'raylib':
            return self.client.render(grid)
            #return self.c_envs[0].render()
        elif self.render_mode == 'human':
            snakes_in_first_env = self.num_snakes[0]
            snake_ptrs = self.snake_ptrs[:snakes_in_first_env]
            agent_positions = self.snakes[np.arange(snakes_in_first_env), snake_ptrs]
            actions = self.actions[:snakes_in_first_env]

            self.atn = None
            if rl.IsKeyDown(rl.KEY_UP) or rl.IsKeyDown(rl.KEY_W):
                self.atn = 0
            elif rl.IsKeyDown(rl.KEY_DOWN) or rl.IsKeyDown(rl.KEY_S):
                self.atn = 1
            elif rl.IsKeyDown(rl.KEY_LEFT) or rl.IsKeyDown(rl.KEY_A):
                self.atn = 2
            elif rl.IsKeyDown(rl.KEY_RIGHT) or rl.IsKeyDown(rl.KEY_D):
                self.atn = 3

            return self.client.render(grid, agent_positions)
        else:
            raise ValueError(f'Invalid render mode: {self.render_mode}')

class RaylibClient:
    def __init__(self, width, height, colors, tile_size=16):
        self.width = width
        self.height = height
        self.colors = colors
        self.tile_size = tile_size

        rl.InitWindow(width*tile_size, height*tile_size,
            "PufferLib Ray Snake".encode())
        rl.SetTargetFPS(15)

    def render(self, grid, agent_positions):
        if rl.IsKeyDown(rl.KEY_ESCAPE):
            exit(0)

        rl.BeginDrawing()
        rl.ClearBackground(render.PUFF_BACKGROUND)

        ts = self.tile_size
        main_r, main_c = agent_positions[0]
        r_min = main_r - self.height//2
        r_max = main_r + self.height//2
        c_min = main_c - self.width//2
        c_max = main_c + self.width//2

        for i, r in enumerate(range(r_min, r_max+1)):
            for j, c in enumerate(range(c_min, c_max+1)):
                if (r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]):
                    continue

                tile = grid[r, c]
                if tile == 0:
                    continue

                rl.DrawRectangle(j*ts, i*ts, ts, ts, self.colors[tile])

        rl.EndDrawing()
        return render.cdata_to_numpy()

def test_performance(timeout=10, atn_cache=1024):
    env = Snake()
    env.reset()
    tick = 0

    total_snakes = sum(env.num_snakes)
    actions = np.random.randint(0, 4, (atn_cache, total_snakes))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', total_snakes * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()

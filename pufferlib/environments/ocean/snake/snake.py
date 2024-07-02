'''Env originally by https://github.com/dnbt777'''

import numpy as np
import random
import gymnasium

import pufferlib
from pufferlib.environments.ocean.snake.c_snake import CSnake, CMultiSnake

EMPTY = 0
SNAKE = 1
FOOD = 2
CORPSE = 3
WALL = 4

class Snake(pufferlib.PufferEnv):
    def __init__(self, widths, heights, num_snakes, num_food, vision=15,
            leave_corpse_on_death=True, render_mode='ansi'):
        super().__init__()
        self.grids = [np.zeros((h, w), dtype=np.uint8) for h, w in zip(heights, widths)]

        assert len(widths) == len(heights)
        for w, h in zip(widths, heights):
            assert w >= 48 and h >= 48

        total_snakes = sum(num_snakes)

        # TODO: 10k can fail but otherwise OOM
        max_snake_length = min(10000, max([w*h for h, w in zip(heights, widths)]))
        self.snakes = np.zeros((total_snakes, max_snake_length, 2), dtype=np.int32) - 1

        self.snake_lengths = np.zeros(total_snakes, dtype=np.int32)
        self.snake_ptrs = np.zeros(total_snakes, dtype=np.int32)

        self.num_snakes = num_snakes
        self.num_food = num_food
        self.vision = vision
        self.leave_corpse_on_death = len(widths)*[leave_corpse_on_death]

        self.render_mode = render_mode

        box = 2 * vision + 1
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(box, box), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(4)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = total_snakes
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

    def reset(self, seed=None):
        self.c_env = CMultiSnake(self.grids, self.snakes, self.buf.observations,
            self.snake_lengths, self.snake_ptrs, self.actions, self.buf.rewards,
            self.num_snakes, self.num_food, self.vision, self.leave_corpse_on_death)

        self.c_env.reset()
        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        self.c_env.step()

        info = {}
        if self.tick % 128 == 0:
            info = {
                'snake_length': np.mean(self.snake_lengths),
                'reward': self.buf.rewards.mean(),
            }

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self):
        grid = self.grids[0]
        height, width = grid.shape
        if self.render_mode == 'rgb_array':
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[grid==SNAKE] = np.array([255, 0, 0])
            frame[grid==FOOD] = np.array([0, 255, 0])
            frame[grid==CORPSE] = np.array([255, 0, 255])
            frame[grid==WALL] = np.array([0, 0, 255])
            return frame

        def _render(val):
            if val == 0:
                c = 90
            elif val == SNAKE:
                c = 92
            elif val == FOOD:
                c = 94
            elif val == CORPSE:
                c = 95
            elif val == WALL:
                c = 97

            return f'\033[{c}m██\033[0m'

        lines = []
        for line in grid:
            lines.append(''.join([_render(val) for val in line]))

        return '\n'.join(lines)

def perf_test():
    num_snakes = 1024
    #env = Snake([1024], [1024], [num_snakes], [1024])
    env = Snake(
        1024*[40],
        1024*[40],
        1024*[1],
        1024*[1],
    )
    env.reset()

    import numpy as np
    actions = np.random.randint(0, 4, (1000, num_snakes))

    import time
    start = time.time()
    done = True
    tick = 0
    while time.time() - start < 10:
        atns = actions[tick % 1000]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', num_snakes * tick / (time.time() - start))

if __name__ == '__main__':
    perf_test()

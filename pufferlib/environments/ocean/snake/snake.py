'''Env originally by https://github.com/dnbt777'''

import numpy as np
import random
import gymnasium

import pufferlib
from pufferlib.environments.ocean.snake.c_snake import CSnake

EMPTY = 0
SNAKE = 1
FOOD = 2
WALL = 3

class Snake(pufferlib.PufferEnv):
    def __init__(self, width, height, snakes=4, food=4, vision=5):
        super().__init__()
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.max_snake_length = width * height
        self.snake = np.zeros((snakes, self.max_snake_length, 2), dtype=np.int32) - 1
        self.snake_lengths = np.zeros(snakes, dtype=np.int32)
        self.snake_ptr = np.zeros(snakes, dtype=np.int32)

        self.width = width
        self.height = height
        self.render_mode = 'ansi'
        self.food = food

        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(height, width), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(4)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.emulated = None
        self.num_agents = snakes
        self.done = False

        self.vision = vision
        box = 2 * vision + 1
        self.buf = pufferlib.namespace(
            observations = np.zeros(
                (snakes, box, box), dtype=np.uint8),
            rewards = np.zeros(snakes, dtype=np.float32),
            terminals = np.zeros(snakes, dtype=bool),
            truncations = np.zeros(snakes, dtype=bool),
            masks = np.ones(snakes, dtype=bool),
        )
        self.actions = np.zeros(snakes, dtype=np.uint32)
        self.tick = 0

    def reset(self, seed=None):
        self.c_env = CSnake(self.grid, self.snake, self.buf.observations, self.snake_lengths,
            self.snake_ptr, self.actions, self.buf.rewards, self.food, self.vision)
        self.c_env.reset()
        return self.grid, {}

    def step(self, actions):
        self.actions[:] = actions
        self.c_env.step()
        info = {}
        if self.tick % 128 == 0:
            info = {'snake_length': np.mean(self.snake_lengths)}
        return self.grid, self.buf.rewards, self.buf.terminals, self.buf.truncations, info

    def render(self):
        def _render(val):
            if val == 0:
                c = 90
            elif val == SNAKE:
                c = 91
            elif val == FOOD:
                c = 94
            elif val == WALL:
                c = 93

            return f'\033[{c}m██\033[0m'

        lines = []
        for line in self.grid:
            lines.append(''.join([_render(val) for val in line]))

        return '\n'.join(lines)

def perf_test():
    num_snakes = 4
    env = Snake(40, 40)
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

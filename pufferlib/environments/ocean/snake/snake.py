'''Env originally by https://github.com/dnbt777'''

import numpy as np
import random
import gymnasium

from pufferlib.environments.ocean.snake.c_snake import CSnake


EMPTY = 0
SNAKE = 1
FOOD = 2

use_c = True

class Snake(gymnasium.Env):
    def __init__(self, width, height):
        super().__init__()
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.max_snake_length = width * height
        self.snake = np.zeros((self.max_snake_length, 2), dtype=np.int32)

        if use_c:
            self.c_env = CSnake(self.grid, self.snake)

        self.width = width
        self.height = height
        self.render_mode = 'ansi'

        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(height, width), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(4)

    def reset(self, seed=None):
        if use_c:
            self.c_env.reset()
            return self.grid, {}

        self.grid.fill(0)
        self.snake.fill(-1)

        #random.seed(seed)
        self.head_r = self.height // 2
        self.head_c = self.width // 2
        self.head_ptr = 0
        self.snake_length = 1
        self.grid[self.head_r, self.head_c] = 1

        self.place_food()
        self.done = False
        return self.grid, {}

    def place_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.grid[y, x] == 0:
                self.food_r = y
                self.food_c = x
                self.grid[y, x] = FOOD
                return

    def step(self, action):
        if use_c:
            reward = self.c_env.step(action)
            done = reward == -1
            info = {}
            #if done:
            #    info['snake_length'] = self.c_env.snake_length
            return self.grid, reward, done, False, info

        dr = 0
        dc = 0
        if action == 0:  # up
            dr = -1
        elif action == 1:  # down
            dr = 1
        elif action == 2:  # left
            dc = -1
        elif action == 3:  # right
            dc = 1

        next_r = self.head_r + dr
        next_c = self.head_c + dc

        hit_wall = (next_r < 0 or next_r >= self.height
            or next_c < 0 or next_c >= self.width)
        hit_self = self.grid[next_r, next_c] == SNAKE
        if hit_wall or hit_self:
            info = {'snake_length': len(self.snake)}
            return self.grid, -1, True, False, info

        self.head_ptr += 1
        if self.head_ptr >= self.max_snake_length:
            self.head_ptr = 0

        self.snake[self.head_ptr, 0] = next_r
        self.snake[self.head_ptr, 1] = next_c

        if self.grid[next_r, next_c] == FOOD:
            reward = 0.1
            self.snake_length += 1
            self.place_food()
        else:
            reward = 0.0
            tail_ptr = (self.head_ptr - self.snake_length + 1) % self.max_snake_length
            tail_r = self.snake[tail_ptr, 0]
            tail_c = self.snake[tail_ptr, 1]
            self.snake[tail_ptr, 0] = -1
            self.snake[tail_ptr, 1] = -1
            self.grid[tail_r, tail_c] = EMPTY

        self.grid[next_r, next_c] = SNAKE

        dist_to_food = abs(self.food_r - self.head_r) + abs(self.food_c - self.head_c)
        next_dist_food = abs(next_r - self.food_r) + abs(next_c - self.food_c)
        if next_dist_food < dist_to_food:
            reward += 0.01

        return self.grid, reward, self.done, False, {}

    def render(self):
        def _render(val):
            if val == 0:
                c = 90
            elif val == SNAKE:
                c = 91
            elif val == FOOD:
                c = 94

            return f'\033[{c}m██\033[0m'

        lines = []
        for line in self.grid:
            lines.append(''.join([_render(val) for val in line]))

        return '\n'.join(lines)

def perf_test():
    env = Snake(40, 40)

    import numpy as np
    actions = np.random.randint(0, 4, size=1000)

    import time
    start = time.time()
    done = True
    tick = 0
    while time.time() - start < 10:
        if done:
            env.reset()
            done = False
        else:
            _, _, done, _, _ = env.step(tick % 1000)

        tick += 1

    print(f'SPS: %f', tick / (time.time() - start))

if __name__ == '__main__':
    perf_test()

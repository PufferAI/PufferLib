'''Env originally by https://github.com/dnbt777'''

import numpy as np
import random
from math import copysign, log
import gymnasium

def abslogint(x):
    if x == 0:
        y = 0
    else:
        y = copysign(1, x)*abs(int(1+2*log(abs(x))))
    return y

class Snake(gymnasium.Env):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.render_mode = 'ansi'
        self.observation_space = gymnasium.spaces.Box(
            low=-128, high=128, shape=(7,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(4)

    def reset(self, seed=42):
        random.seed(seed)
        self.snake = [(self.width // 2, self.height// 2)]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.food = self.place_food()
        self.done = False
        self.previous_distance = self.get_distance_to_food()
        return self.get_state(), {}

    def place_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                return (x, y)

    def get_distance_to_food(self):
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def get_state(self):
        head = self.snake[0]
        food = self.food
        
        # Distance to walls
        distance_to_left = head[0]
        distance_to_right = self.width - head[0]
        distance_to_top = head[1]
        distance_to_bottom = self.height - head[1]
        
        # Relative position of the food to the head
        food_rel_x = food[0] - head[0]
        food_rel_y = food[1] - head[1]
        
        # Check for tail in adjacent cells
        adjacent_cells = [
            (head[0], head[1] - 1),  # Up
            (head[0], head[1] + 1),  # Down
            (head[0] - 1, head[1]),  # Left
            (head[0] + 1, head[1])   # Right
        ]
        
        tail_state = 0
        for i, cell in enumerate(adjacent_cells):
            if cell in self.snake[1:]:
                tail_state |= (1 << (3 - i))
        
        return np.array([
            abslogint(distance_to_left),
            abslogint(distance_to_right),
            abslogint(distance_to_top),
            abslogint(distance_to_bottom),
            abslogint(food_rel_x),
            abslogint(food_rel_y),
            tail_state
            ], dtype=np.float32)

    def step(self, action):
        new_direction = self.direction
        if action == 0:  # up
            new_direction = (0, -1)
        elif action == 1:  # down
            new_direction = (0, 1)
        elif action == 2:  # left
            new_direction = (-1, 0)
        elif action == 3:  # right
            new_direction = (1, 0)

        # Check if the new direction would cause the snake to move backwards
        if len(self.snake) > 1:
            head = self.snake[0]
            neck = self.snake[1]
            if (head[0] + new_direction[0], head[1] + new_direction[1]) == neck:
                # If moving backwards, keep the current direction
                new_direction = self.direction

        self.direction = new_direction
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check for collision with walls or self
        if (new_head in self.snake or 
            new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            self.done = True
            return self.get_state(), -1, self.done, False, {}  # Massive negative reward for death

        self.snake.insert(0, new_head)

        if new_head == self.food:
            reward = 0.1  # Large positive reward for eating food
            self.food = self.place_food()
            self.previous_distance = self.get_distance_to_food()
        else:
            self.snake.pop()
            new_distance = self.get_distance_to_food()
            if new_distance < self.previous_distance:
                reward = 0.01  # Small positive reward for moving towards food
            else:
                reward = 0.0
            #else:
            #    reward = -0.02  # Small negative reward for moving away from food
            self.previous_distance = new_distance

        return self.get_state(), reward, self.done, False, {}

    def render(self):
        def _render(c):
            return f'\033[{c}m██\033[0m'

        chars = self.height*[self.width*[90]]
        r, c = self.food
        chars[r][c] = 94
        for segment in self.snake:
            r, c = segment
            chars[r][c] = 91

        chars = [[_render(c) for c in row] for row in chars]
        chars = [''.join(row) for row in chars]
        return '\n'.join(chars)

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

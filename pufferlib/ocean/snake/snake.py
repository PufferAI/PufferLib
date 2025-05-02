'''High-perf many-agent snake. Inspired by snake env from https://github.com/dnbt777'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.exceptions import APIUsageError
from pufferlib.ocean.snake.cy_snake import CySnake

class Snake(pufferlib.PufferEnv):
    def __init__(self, num_envs=16, width=640, height=360,
            num_snakes=256, num_food=4096,
            vision=5, leave_corpse_on_death=True,
            reward_food=0.1, reward_corpse=0.1, reward_death=-1.0,
            report_interval=128, max_snake_length=1024,
            render_mode='human', buf=None, seed=0):
        
        if num_envs is not None:
            num_snakes = num_envs * [num_snakes]
            width = num_envs * [width]
            height = num_envs * [height]
            num_food = num_envs * [num_food]
            leave_corpse_on_death = num_envs * [leave_corpse_on_death]

        if not (len(num_snakes) == len(width) == len(height) == len(num_food)):
            raise APIUsageError('num_snakes, width, height, num_food must be lists of equal length')

        for w, h in zip(width, height):
            if w < 2*vision+2 or h < 2*vision+2:
                raise APIUsageError('width and height must be at least 2*vision+2')

        max_area = max([w*h for h, w in zip(height, width)])
        self.max_snake_length = min(max_snake_length, max_area)
        self.report_interval = report_interval

        # This block required by advanced PufferLib env spec
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(2*vision+1, 2*vision+1), dtype=np.int8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.num_agents = sum(num_snakes)
        self.render_mode = render_mode
        self.tick = 0

        self.cell_size = int(np.ceil(1280 / max(max(width), max(height))))

        super().__init__(buf)
        self.c_envs = CySnake(self.observations, self.actions,
            self.rewards, self.terminals, width, height,
            num_snakes, num_food, vision, max_snake_length,
            leave_corpse_on_death, reward_food, reward_corpse,
            reward_death)
 
    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render(self.cell_size)

    def close(self):
        self.c_envs.close()

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

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.froggy import binding

class Froggy(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=5, width=11, height=11, episode_length=1000, buf=None, seed=0):
        basic_low = np.array([0, 0, 0, 0, -np.inf], dtype=np.float32)  # frog_x, frog_y, lives, crossings, score
        basic_high = np.array([width-1, height-1, 3, np.inf, np.inf], dtype=np.float32)
        car_low = np.array([-1, -1, 0, 0, -1] * 15, dtype=np.float32)  # car: x, y, active, speed, direction
        car_high = np.array([1, 1, 1, 1, 1] * 15, dtype=np.float32)
        grid_low = np.array([-1] * 50, dtype=np.float32)  # 25 map cells + 25 car presence
        grid_high = np.array([1] * 50, dtype=np.float32)
        low = np.concatenate([basic_low, car_low, grid_low])
        high = np.concatenate([basic_high, car_high, grid_high])
        
        self.single_observation_space = gymnasium.spaces.Box(low=low, high=high,
            shape=(130,), dtype=np.float32)  # 5 basic + 75 car + 50 grid = 130 total
        # up, down, left, right
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, width=width, height=height, episode_length=episode_length)
        self.size = size
        self.width = width
        self.height = height
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        # np.nan_to_num(self.observations, copy=False)
        info = [binding.vec_log(self.c_envs)]
        # print('Froggy step info:', info[0])
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 4096
    env = Froggy(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    print('froggy SPS:', int(env.num_agents*steps / (time.time() - start)))

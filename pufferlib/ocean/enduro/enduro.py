import random
import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.enduro import binding

class Enduro(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 width=152, height=210, car_width=16, car_height=11,
                 max_enemies=10, frameskip=1, continuous=False,
                 log_interval=128, buf=None, seed=None):
        
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(8 + (5 * max_enemies) + 9 + 1,), dtype=np.float32
        )
        # Example: 9 discrete actions
        self.single_action_space = gymnasium.spaces.Discrete(9)
        
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.continuous = continuous
        self.log_interval = log_interval
        self.human_action = None
        self.tick = 0
        if seed is None:
            self.seed = random.randint(1, 1000000)
        else:
            self.seed = 0
        super().__init__(buf)

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, self.seed,
            width=width, height=height, car_width=car_width,
            car_height=car_height, max_enemies=max_enemies,
            frameskip=frameskip, continuous=continuous
        )

    def reset(self, seed=None):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        
        self.tick += 1
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = Enduro(num_envs=2)
    env.reset(env.seed)
    tick = 0

    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: {env.num_agents * tick / (time.time() - start)}')

if __name__ == '__main__':
    test_performance()
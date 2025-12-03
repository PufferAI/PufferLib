'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.boxoban import binding
from pufferlib.ocean.boxoban.parse_maps import write_bin 
import os

class Boxoban(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=10, buf=None, seed=0, difficulty="medium", max_steps = 500):
        self.shape = size*size*4 #agents walls boxes targets OHE

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.shape,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.max_steps = max_steps

        #Load maps
        """
        if difficulty == "medium":
            p = "boxoban-levels/medium/train"
            pv = "boxoban-levels/medium/valid"
            maps = [os.path.join(p, f) for f in os.listdir(p) if f.endswith('.txt')]
            maps_valid = [os.path.join(pv, f) for f in os.listdir(pv) if f.endswith('.txt')]
        elif difficulty == "hard":
            p = "boxoban-levels/hard"
            pv = "boxoban-levels/unfiltered/valid"
            maps = [os.path.join(p, f) for f in os.listdir(p) if f.endswith('.txt')]
            maps_valid = [os.path.join(pv, f) for f in os.listdir(pv) if f.endswith('.txt')]
        elif difficulty == "unfiltered":
            p = "boxoban-levels/unfiltered"
            pv = "boxoban-levels/unfiltered/valid"
            maps = [os.path.join(p, f) for f in os.listdir(p) if f.endswith('.txt')]
            maps_valid = [os.path.join(pv, f) for f in os.listdir(pv) if f.endswith('.txt')]
        else:
            raise ValueError("Invalid difficulty")
        write_bin(maps, 'boxoban_maps.bin')
        write_bin(maps_valid, 'boxoban_maps_valid.bin')"""


        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, size=size, max_steps = self.max_steps)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        self.actions[:] = actions
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

if __name__ == '__main__':
    N = 1

    env = Boxoban(num_envs=N)
    env.reset()
    env.render()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print('Boxoban SPS:', int(steps / (time.time() - start)))

'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.rubiks import binding

class Cube(pufferlib.PufferEnv):
    def __init__(self, 
                 num_envs=1,
                 render_mode=None, 
                 log_interval=128, 
                 N=3,
                 shuffles = 1,
                 obs_type='basic',
                 buf=None, 
                 seed=0):

        if obs_type == 'basic':
            self.single_observations_space = gymnasium.spaces.Box(low=0, 
                                                                 high=1, 
                                                                 shape=(6, N, N, 6), #faces, height, width, colours
                                                                 dtype=np.float32) 
        else:
            raise NotImplementedError(f'Cublets not yet implemented: {obs_type}')

        self.single_action_space = gymnasium.spaces.Discrete(12) # 6 faces, clockwise and anticlockwise


        self.render_mode = render_mode
        self.log_interval = log_interval
        self.size = np.prod(self.single_observations_space.shape)
        super().__init__(buf)
        self.c_envs = binding.env_init(self.observations,
                                       self.actions,
                                       self.rewards,
                                       self.terminals,
                                       self.truncations,
                                       shuffles = shuffles,
                                       N = N,
                                       obs_type = obs_type,
                                       size = self.size,
                                       seed= seed,
                                       num_envs = num_envs)
                        
                  
     
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
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 512

    env = Cube(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(env.single_action_space.nvec, size=(CACHE, 2))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Target SPS:', int(steps / (time.time() - start)))

'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.rubiks import binding

class Cube(pufferlib.PufferEnv):
    def __init__(self, 
                 num_envs=2,
                 num_agents=1,
                 render_mode=None, 
                 log_interval=128, 
                 N=3,
                 shuffles = 0,
                 obs_type='basic',
                 buf=None,
                 max_steps = 1000,
                 seed=0,
                 anim_time = 0.5):

        if obs_type == 'basic':
            self.single_observation_space = gymnasium.spaces.Box(low=0, 
                                                                 high=1, 
                                                                 shape=(6, N, N, 6), #faces, height, width, colours
                                                                 dtype=np.float32) 
        else:
            raise NotImplementedError(f'Cublets not yet implemented: {obs_type}')

        self.single_action_space = gymnasium.spaces.Discrete(12) # 6 faces, clockwise and anticlockwise
        self.num_envs = num_envs
        self.seed = seed
        self.num_envs = num_envs
        self.num_agents=num_envs
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.size = int(np.prod(self.single_observation_space.shape))
        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations,
                                       self.actions,
                                       self.rewards,
                                       self.terminals,
                                       self.truncations,
                                       num_envs,
                                       seed,
                                       shuffles = shuffles,
                                       N = N,
                                       size = self.size,
                                       max_episode_steps = max_steps,
                                       anim_time = anim_time
                                       )
                        
                  
     
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
    num_envs = 1
    env = Cube(num_envs = num_envs)
    env.reset()
    steps = 0
    env.render()
    CACHE = 1000
    actions = np.random.randint(0, 12, (CACHE, num_envs))
   
    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Rubiks SPS:', int(steps / (time.time() - start)))

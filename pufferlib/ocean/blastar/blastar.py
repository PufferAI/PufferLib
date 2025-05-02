import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.blastar import binding

class Blastar(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(6)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.num_obs = self.single_observation_space.shape[0]
        self.tick = 0
        self.log_interval = 1
        
        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            num_obs=self.num_obs
        )

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
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
    env = Blastar(num_envs=1000)
    env.reset(0)
    tick = 0

    rng = np.random.default_rng()
    actions = rng.integers(0, 6, (atn_cache, env.num_agents))
 
    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print('SPS:', env.num_agents * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()

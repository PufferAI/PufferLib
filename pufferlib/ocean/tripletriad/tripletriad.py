import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.tripletriad import binding

class TripleTriad(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
            width=990, height=690, card_width=192, card_height=224, buf=None, seed=0,
            max_num_threads=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(114,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(14)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf, binding, max_num_threads)
        self.c_envs = binding.vec_init(self.observations, self.actions,
            self.rewards, self.terminals, self.truncations, num_envs, seed, width=width, height=height,
            card_width=card_width, card_height=card_height)

    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = TripleTriad(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: {env.num_agents * tick / (time.time() - start):,}')

if __name__ == '__main__':
    test_performance()

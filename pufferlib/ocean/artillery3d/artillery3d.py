import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.artillery3d import binding

class Artillery3D(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 max_reward_dist=23.620013496047616,  target_size=15,
                 dist_fade=0.7493789405520497, turn_penalty_delay=72, turn_penalty_ramp=0.02, max_dist0=127.50246246114087,
                 turn_penalty=-0.003, miss_penalty=-0.1858434974084412, render=1, out_bounds_penalty=-0.01,
                 log_interval=128,
                 seed=0,
                 buf=None):
        obs_size = 7 + 12
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0

        self.single_action_space = gymnasium.spaces.Discrete(7)

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            max_reward_dist=max_reward_dist,  target_size=target_size,
            dist_fade=dist_fade, turn_penalty_delay=turn_penalty_delay, turn_penalty_ramp=turn_penalty_ramp, max_dist0=max_dist0,
            turn_penalty=turn_penalty, miss_penalty=miss_penalty, render=render,
            out_bounds_penalty=out_bounds_penalty,
        )

    def reset(self, seed=0):
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
    env = Artillery3D(num_envs=1)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 6, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:

        atn = actions[tick % atn_cache]

        env.step(atn)
        tick += 1

if __name__ == '__main__':
    test_performance()

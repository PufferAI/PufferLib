import numpy as np
import gymnasium
import time

import pufferlib
from pufferlib.ocean.artillery import binding

class Artillery(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 frameskip=1,
                 min_aim_angle=0.56, max_aim_angle=1.56, max_reward_dist=45,
                 adj=0.0144, dist_fade=0.11, turn_penalty_delay=98, max_dist0=105,
                 turn_penalty=-0.1, miss_penalty=-0.055, render=1, out_bounds_penalty=-0.1,
                 log_interval=128,
                 seed=7,
                 buf=None):
        obs_size = 11
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0

        self.single_action_space = gymnasium.spaces.MultiDiscrete([5, 5])

        super().__init__(buf)

        self.actions = self.actions

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards, self.terminals, self.truncations, num_envs,
            seed, num_envs=num_envs, seed=seed, frameskip=frameskip,
            min_aim_angle=min_aim_angle, max_aim_angle=max_aim_angle, max_reward_dist=max_reward_dist,
            adj=adj, dist_fade=dist_fade, turn_penalty_delay=turn_penalty_delay, max_dist0=max_dist0,
            turn_penalty=turn_penalty, miss_penalty=miss_penalty, render=render,
            out_bounds_penalty=out_bounds_penalty
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
    env = Artillery(num_envs=1)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 5, (atn_cache, env.num_agents, 2))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

if __name__ == '__main__':
    test_performance()

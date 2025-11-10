import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.artillery3d import binding

class Artillery3D(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 max_reward=1.0, max_reward_dist=23.620013496047616,  target_size=15,
                 dist_fade=0.7493789405520497, turn_penalty_delay=72.37761171826367, turn_penalty_ramp=0.2, max_dist0=127.50246246114087,
                 turn_penalty=-0.003, miss_penalty=-0.1858434974084412, render=1, out_bounds_penalty=-0.01,
                 log_interval=128,
                 seed=7,
                 buf=None, rng=7, i=1, debug=0, same_runs=0):
        obs_size = 7 + 12
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0

        self.single_action_space = gymnasium.spaces.Discrete(7)

        super().__init__(buf)

        c_envs = []
        for i in range(num_envs):
            env_id = binding.env_init(
                self.observations[i:i+1],
                self.actions[i:i+1],
                self.rewards[i:i+1],
                self.terminals[i:i+1],
                self.truncations[i:i+1],
                seed, num_envs=num_envs,
                max_reward=max_reward, max_reward_dist=max_reward_dist,  target_size=target_size,
                dist_fade=dist_fade, turn_penalty_delay=turn_penalty_delay, turn_penalty_ramp=turn_penalty_ramp, max_dist0=max_dist0,
                turn_penalty=turn_penalty, miss_penalty=miss_penalty, render=render,
                out_bounds_penalty=out_bounds_penalty,
                rng=rng+i, i=i, debug=debug, same_runs=same_runs
            )
            c_envs.append(env_id)
        self.c_envs = binding.vectorize(*c_envs)

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

    actions = np.random.randint(0, 4, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:

        atn = actions[tick % atn_cache]

        env.step(atn)
        tick += 1

if __name__ == '__main__':
    test_performance()

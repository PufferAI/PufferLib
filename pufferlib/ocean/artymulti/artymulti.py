import numpy as np
import gymnasium
import time

import pufferlib
from pufferlib.ocean.artymulti import binding

class ArtyMulti(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 frameskip=1, width=1280, height=720,
                 target_min_x=600, target_max_x=1230, target_min_y=300, target_max_y=670, target_size=15,
                 min_aim_angle=0.56, max_aim_angle=1.56, max_reward=1.0, max_reward_dist=8.5,
                 dist_fade=0.36, turn_penalty_delay=64, turn_penalty_ramp=0.023, max_dist0=100.0,
                 turn_penalty=-0.03, miss_penalty=-0.1, render=1, out_bounds_penalty=-0.1,
                 log_interval=128,
                 vm=150.0,
                 seed=7,
                 buf=None, rng=7, i=1, debug=0, same_runs=0):
        obs_size = 6
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0

        self.single_action_space = gymnasium.spaces.Discrete(5)
        #self.single_action_space = gymnasium.spaces.MultiDiscrete([5] * 2)

        super().__init__(buf)

        self.actions = self.actions.astype(np.float32)

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards, self.terminals, self.truncations, num_envs,
            seed, num_envs=num_envs, seed=seed, frameskip=frameskip, width=width, height=height,
            target_min_x=target_min_x, target_max_x=target_max_x, target_min_y=target_min_y, target_max_y=target_max_y, target_size=target_size,
            min_aim_angle=min_aim_angle, max_aim_angle=max_aim_angle, max_reward=max_reward, max_reward_dist=max_reward_dist,
            dist_fade=dist_fade, turn_penalty_delay=turn_penalty_delay, turn_penalty_ramp=turn_penalty_ramp, max_dist0=max_dist0,
            turn_penalty=turn_penalty, miss_penalty=miss_penalty, render=render,
            vm=vm, out_bounds_penalty=out_bounds_penalty,
            rng=rng+i, i=i, debug=debug, same_runs=same_runs
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

        #print('P Obs:', ' '.join(f'{x:.3f}' for x in self.observations.flatten()))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = ArtyMulti(num_envs=1)
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

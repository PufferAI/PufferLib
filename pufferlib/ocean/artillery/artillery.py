import numpy as np
import gymnasium
import time

import pufferlib
from pufferlib.ocean.artillery import binding

class Artillery(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 frameskip=1, width=1280, height=720, moving_target=1, timed_shell=0,
                 target_min_x=600, target_max_x=1230, target_min_y=300, target_max_y=670, target_size=15,
                 min_aim_angle=0.56, max_aim_angle=1.56, max_reward=1.0, max_reward_dist=30, max_score=1.0,
                 dist_fade=0.3, turn_penalty_delay=75, turn_penalty_ramp=0.015, max_dist0=250.0,
                 turn_penalty=-0.03, miss_penalty=-0.2, render=1, out_bounds_penalty=-0.1,
                 continuous=False, log_interval=128,
                 ftmp1=0.1, ftmp2=0.1, ftmp3=0.1, ftmp4=0.1, vm=150.0,
                 seed=7,
                 buf=None, rng=7, i=1, method=0, debug=0, same_runs=0):
        obs_size = 8 if moving_target == 1 else 6
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.continuous = continuous
        self.log_interval = log_interval
        self.tick = 0

        if continuous:
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.single_action_space = gymnasium.spaces.Discrete(5)

        super().__init__(buf)

        if continuous:
            self.actions = self.actions.flatten()
        else:
            self.actions = self.actions.astype(np.float32)

        c_envs = []
        for i in range(num_envs):
            env_id = binding.env_init(
                self.observations[i:i+1],
                self.actions[i:i+1],
                self.rewards[i:i+1],
                self.terminals[i:i+1],
                self.truncations[i:i+1],
                seed, num_envs=num_envs, seed=seed, frameskip=frameskip, width=width, height=height, moving_target=moving_target, timed_shell=timed_shell,
                target_min_x=target_min_x, target_max_x=target_max_x, target_min_y=target_min_y, target_max_y=target_max_y, target_size=target_size,
                min_aim_angle=min_aim_angle, max_aim_angle=max_aim_angle, max_reward=max_reward, max_reward_dist=max_reward_dist, max_score=max_score,
                dist_fade=dist_fade, turn_penalty_delay=turn_penalty_delay, turn_penalty_ramp=turn_penalty_ramp, max_dist0=max_dist0,
                turn_penalty=turn_penalty, miss_penalty=miss_penalty, render=render, continuous=continuous,
                ftmp1=ftmp1,ftmp2=ftmp2,ftmp3=ftmp3,ftmp4=ftmp4, vm=vm, out_bounds_penalty=out_bounds_penalty,
                rng=rng+i, i=i, method=method, debug=debug, same_runs=same_runs
            )
            c_envs.append(env_id)
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        if self.continuous:
            self.actions[:] = np.clip(actions.flatten(), -1.0, 1.0)
        else:
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

    actions = np.random.randint(0, 4, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:

        atn = actions[tick % atn_cache]

        env.step(atn)
        tick += 1

if __name__ == '__main__':
    test_performance()

import numpy as np
import gymnasium
import time

import pufferlib
from pufferlib.ocean.whisker_racer import binding

class WhiskerRacer(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 frameskip=4, width=1080, height=720,
                 llw_ang=-3.14/4, flw_ang=-3.14/6,
                 frw_ang=3.14/6, rrw_ang=3.14/4,
                 max_whisker_length=100,
                 turn_pi_frac=20,
                 maxv=5, render=0,
                 continuous=False, log_interval=128,
                 reward_yellow=0.25, reward_green=0.0, gamma=0.9, track_width=50,
                 num_radial_sectors=16, num_points=4, bezier_resolution=16, w_ang=0.523,
                 corner_thresh=0.5, ftmp1=0.1, ftmp2=0.1, ftmp3=0.1, ftmp4=0.1,
                 mode7=0, render_many=0, seed=42,
                 buf=None, rng=42, i=1, method=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
                                            shape=(3,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.continuous = continuous
        self.log_interval = log_interval
        self.tick = 0

        if continuous:
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.single_action_space = gymnasium.spaces.Discrete(3)

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
                seed, num_envs=num_envs, seed=seed, frameskip=frameskip, width=width, height=height,
                llw_ang=llw_ang, flw_ang=flw_ang, frw_ang=frw_ang, rrw_ang=rrw_ang, max_whisker_length=max_whisker_length,
                turn_pi_frac=turn_pi_frac, maxv=maxv, render=render, continuous=continuous,
                reward_yellow=reward_yellow, reward_green=reward_green, gamma=gamma, track_width=track_width,
                num_radial_sectors=num_radial_sectors, num_points=num_points, bezier_resolution=bezier_resolution, w_ang=w_ang,
                corner_thresh=corner_thresh, ftmp1=ftmp1,ftmp2=ftmp2,ftmp3=ftmp3,ftmp4=ftmp4,
                mode7=mode7, render_many=render_many, rng=rng+i, i=i, method=method
            )
            c_envs.append(env_id)
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []
    
    def step(self, actions):
        #start = time.time()
        if self.continuous:
            self.actions[:] = np.clip(actions.flatten(), -1.0, 1.0)
        else:
            self.actions[:] = actions

        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        #end = time.time()
        #print(f"python step took {end - start:.3e} seconds")
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    print("test_performance in whisker_racer.py")
    env = WhiskerRacer(num_envs=1)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        print("atn = actions[tick % atn_cache] in whisker_racer.py")
        atn = actions[tick % atn_cache]
        print("env.step in whisker_racer.py")
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_agents * tick / (time.time() - start))

if __name__ == '__main__':
    print("whisker_racer.py")
    test_performance()

import numpy as np
import gymnasium
import time

import pufferlib
from pufferlib.ocean.artillery import binding

class Artillery(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 frameskip=4, width=640, height=480,
                 target_min_x=50, target_max_x=1870, target_min_y=50, target_max_y=1030,
                 min_aim_angle=1.0, max_aim_angle=1.57, max_reward=1.0, max_reward_dist=100, max_score=100.0,
                 render=0,
                 continuous=False, log_interval=128,
                 ftmp1=0.1, ftmp2=0.1, ftmp3=0.1, ftmp4=0.1,
                 render_many=0, seed=42,
                 buf=None, rng=42, i=1, method=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
                                            shape=(5,), dtype=np.float32)
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
                seed, num_envs=num_envs, seed=seed, frameskip=frameskip, width=width, height=height,
                target_min_x=target_min_x, target_max_x=target_max_x, target_min_y=target_min_y, target_max_y=target_max_y,
                min_aim_angle=min_aim_angle, max_aim_angle=max_aim_angle, max_reward=max_reward, max_reward_dist=max_reward_dist, max_score=max_score,
                render=render, continuous=continuous,
                ftmp1=ftmp1,ftmp2=ftmp2,ftmp3=ftmp3,ftmp4=ftmp4,
                render_many=render_many, rng=rng+i, i=i, method=method
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
    print("test_performance in artillery.py")
    env = Artillery(num_envs=1)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 4, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        print("atn = actions[tick % atn_cache] in artillery.py")
        atn = actions[tick % atn_cache]
        print("env.step in artillery.py")
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_agents * tick / (time.time() - start))

if __name__ == '__main__':
    print("artillery.py")
    test_performance()

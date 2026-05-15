import gymnasium
import numpy as np
import pufferlib
from pufferlib.ocean.lock_key import binding

class LockKey(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=8, num_keys=3, buf=None, seed=0, obs_dist=2):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=3, shape=(size * size,), dtype=np.uint8
        )

        self.single_action_space = gymnasium.spaces.Discrete(5)

        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        super().__init__(buf)

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            size=size,
            num_keys=num_keys,
            obs_dist=obs_dist,
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
            info.append(binding.vec_log(self.c_envs))
        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

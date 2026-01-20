"""A minimal template for your own envs."""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.boss_fight import binding


class BossFight(pufferlib.PufferEnv):
    def __init__(
        self, num_envs=1, render_mode=None, log_interval=128, size=5, buf=None, seed=0
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-10, high=110, shape=(13,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(7)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            size=size,
        )
        self.size = size

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


if __name__ == "__main__":
    N = 4096
    env = BossFight(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    import time

    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    print("Squared SPS:", int(env.num_agents * steps / (time.time() - start)))

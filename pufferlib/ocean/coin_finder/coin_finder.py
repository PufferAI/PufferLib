"""A simple coin collection game for reinforcement learning."""

import gymnasium
import numpy as np
import pufferlib
from pufferlib.ocean.coin_finder import binding


class CoinFinder(pufferlib.PufferEnv):
    def __init__(
        self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0
    ):
        # Observation: agent (x,y) + 5 coins (x,y) each = 2 + 10 = 12 floats
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(12,), dtype=np.float32
        )
        # Action: 4 discrete actions (UP, DOWN, LEFT, RIGHT)
        self.single_action_space = gymnasium.spaces.Discrete(4)

        self.render_mode = render_mode
        self.num_agents = num_envs  # Single agent per env
        self.log_interval = log_interval

        super().__init__(buf)

        # Initialize C environments
        c_envs = []
        obs_size = 12  # Size of observation space

        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i * obs_size : (i + 1) * obs_size],
                self.actions[i : i + 1],  # Single action per env
                self.rewards[i : i + 1],  # Single reward per env
                self.terminals[i : i + 1],
                self.truncations[i : i + 1],
                seed,
            )
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

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
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.tcg import binding

class TCG(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1,
            shape=(624,), dtype=np.uint8
        )
        self.single_action_space = gymnasium.spaces.Discrete(12)

        self.render_mode = render_mode
        self.players_per_env = 2
        self.num_envs = num_envs
        self.num_agents = num_envs * self.players_per_env
        self.log_interval = log_interval

        super().__init__(buf)
        obs_dim = int(np.prod(self.single_observation_space.shape))
        self._obs_view = self.observations.reshape(num_envs, self.players_per_env, obs_dim)
        self._act_view = self.actions.reshape(num_envs, self.players_per_env)
        self._rew_view = self.rewards.reshape(num_envs, self.players_per_env)
        self._term_view = self.terminals.reshape(num_envs, self.players_per_env)
        self._trunc_view = self.truncations.reshape(num_envs, self.players_per_env)

        c_envs = []
        for i in range(num_envs):
            env_id = binding.env_init(
                self._obs_view[i],
                self._act_view[i],
                self._rew_view[i],
                self._term_view[i],
                self._trunc_view[i],
                seed + i,
            )
            c_envs.append(env_id)

        self.c_envs = binding.vectorize(*c_envs)

    def _update_masks(self):
        self.masks[:] = self.observations[:, 2].astype(bool)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        self._update_masks()
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self._update_masks()

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

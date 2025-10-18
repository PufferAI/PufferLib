import pufferlib

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.pacman import binding


class Pacman(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
            randomize_starting_position = 0,
            min_start_timeout = 0,
            max_start_timeout = 49,
            frightened_time = 35,
            max_mode_changes = 6,
            scatter_mode_length = 70,
            chase_mode_length = 140,
            log_interval=128,
            buf=None, seed=0, 
            max_num_threads=0):
        
        ghost_observations_count = 9
        player_observations_count = 11
        num_ghosts = 4

        num_dots = 244
        observations_count = (player_observations_count + ghost_observations_count * num_ghosts + num_dots)

        self.single_observation_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(observations_count,),
            dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(4)
        
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.human_action = None
        self.tick = 0

        super().__init__(buf, binding, max_num_threads)

        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            randomize_starting_position = randomize_starting_position,
            min_start_timeout = min_start_timeout,
            max_start_timeout = max_start_timeout,
            frightened_time = frightened_time,
            max_mode_changes = max_mode_changes,
            scatter_mode_length = scatter_mode_length,
            chase_mode_length = chase_mode_length,
        )

    def reset(self, seed=None):
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
        for _ in range(7):
            binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

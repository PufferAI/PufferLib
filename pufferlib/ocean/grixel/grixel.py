import numpy as np
import os

import gymnasium

import pdb

import pufferlib
from pufferlib.ocean.grixel import binding

class Grixel(pufferlib.PufferEnv):
    def __init__(self, render_mode='raylib', vision_range=5,
            num_envs=4096, num_maps=1000, map_size=-1, max_size=9,
            texture_mode=0,
            report_interval=128, buf=None, seed=0):
        assert map_size <= max_size
        
        
        self.texture_mode = texture_mode
        self.pixelize = 1
        self.block_size = 5
        self.nb_object_types = 7 # determines pocket size for observations

        # vision_range better be 5
        self.obs_diameter = 2*vision_range + 1
        if self.pixelize>0:
            self.obs_diameter *= self.block_size
        self.additional_obs_size = 3 + self.nb_object_types # reward, reset, extra + pocket size

        self.single_observation_space = gymnasium.spaces.Box(low=-100, high=100,
            shape=(self.obs_diameter*self.obs_diameter + self.additional_obs_size,), dtype=np.int8)
        #self.single_action_space = gymnasium.spaces.Discrete(5)
        # pass, forward, turn left, turn right, turn back, drop (see top of grixel.h)
        self.single_action_space = gymnasium.spaces.Discrete(6)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        
        # creates the buffers unless passed as args,
        # also probably reads the .ini file
        super().__init__(buf=buf) 
        

        # It's annoying that we have to pass env parameters to both binding.shared
        # and vec_init... but what's the alternative? Each of them independently
        # creates a (C) Grixel env (from grixel.h) and uses it for init_grid....
        self.float_actions = np.zeros_like(self.actions).astype(np.float32)
        self.c_state = binding.shared(num_maps=num_maps, max_size=max_size, size=map_size, 
                    pixelize=self.pixelize, block_size=self.block_size, 
                    additional_obs_size=self.additional_obs_size,
                    nb_object_types=self.nb_object_types)
        

        self.c_envs = binding.vec_init(self.observations, self.float_actions,
            self.rewards, self.terminals, self.truncations, num_envs, seed,
            state=self.c_state, max_size=max_size, num_maps=num_maps, pixelize=self.pixelize, block_size=self.block_size, additional_obs_size=self.additional_obs_size,
            texture_mode=self.texture_mode,
            nb_object_types=self.nb_object_types
            )
        pass

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.float_actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self, overlay=0):
        binding.vec_render(self.c_envs, overlay)

    def close(self):
        pass
        #binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    #env = CGrid(num_envs=1000)
    env = Grixel(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_envs * tick / (time.time() - start))

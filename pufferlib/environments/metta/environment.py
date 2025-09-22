import functools
import numpy as np
import gymnasium

import pufferlib
from pufferlib.pufferlib import set_buffers

from mettagrid.builder.envs import make_arena
from mettagrid.envs.mettagrid_env import MettaGridEnv

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(
    name,
    render_mode="auto",
    buf=None,
    seed=0,
    ore_reward=0.1,
    battery_reward=0.8,
    heart_reward=1.0,
    num_agents=24,
):
    """Metta creation function"""

    # Create a basic arena configuration using the make_arena function
    mettagrid_cfg = make_arena(num_agents=num_agents)

    # Apply reward shaping based on parameters - match easy_shaped_arena_basic
    mettagrid_cfg.game.agent.rewards.inventory = {
        "heart": heart_reward,
        "ore_red": ore_reward,
        "battery_red": battery_reward,
        "laser": 0.5,      # Match easy shaped config
        "armor": 0.5,      # Match easy shaped config
        "blueprint": 0.5,  # Match easy shaped config
    }

    # Set inventory max limits like easy shaped config
    mettagrid_cfg.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # CRITICAL: Easy converter - only 1 battery_red needed for 1 heart (instead of 3)
    mettagrid_cfg.game.objects["altar"].input_resources = {"battery_red": 1}

    env = MettaPuff(mettagrid_cfg, render_mode=render_mode, seed=seed)

    # CRITICAL: Set buffers correctly like Metta does it
    set_buffers(env, buf)

    env.async_reset(seed=42)

    return env
    

def oc_divide(a, b):
    """
    Divide a by b, returning an int if both inputs are ints and result is a whole number,
    otherwise return a float.
    """
    result = a / b
    # If both inputs are integers and the result is a whole number, return as int
    if isinstance(a, int) and isinstance(b, int) and result.is_integer():
        return int(result)
    return result

class MettaPuff(MettaGridEnv):
    def __init__(self, env_cfg, render_mode='human', seed=0):
        self.replay_writer = None
        #if render_mode == 'auto':
        #    self.replay_writer = ReplayWriter("metta/")
        super().__init__(
            env_cfg=env_cfg,
            render_mode=render_mode,
            replay_writer=self.replay_writer,
            is_training=True,  # Enable training mode for desync_episodes
        )
        self.infos = []

    def reset(self, seed=None):
        obs, info = super().reset(seed)

        # Update shared buffers if they exist (for vectorization)
        if hasattr(self, 'observations') and self.observations is not None:
            self.observations[:] = obs

        self.infos = [info] * self.num_agents
        return obs, self.infos

    def step(self, actions):
        obs, rewards, terminals, truncations, infos = super().step(actions)

        # Update shared buffers if they exist (for vectorization)
        if hasattr(self, 'observations') and self.observations is not None:
            self.observations[:] = obs
            self.rewards[:] = rewards
            self.terminals[:] = terminals
            self.truncations[:] = truncations

        self.infos = infos
        return obs, rewards, terminals, truncations, infos
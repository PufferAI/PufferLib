import functools
import numpy as np
import gymnasium

import pufferlib

from metta.mettagrid.builder.envs import make_arena
from metta.mettagrid.mettagrid_env import MettaGridEnv

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(
    name,
    config="pufferlib/environments/metta/metta.yaml",
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

    return MettaPuff(mettagrid_cfg, render_mode=render_mode, buf=buf, seed=seed)

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
    def __init__(self, env_cfg, render_mode='human', buf=None, seed=0):
        self.replay_writer = None
        #if render_mode == 'auto':
        #    self.replay_writer = ReplayWriter("metta/")

        super().__init__(
            env_cfg=env_cfg,
            render_mode=render_mode,
            replay_writer=self.replay_writer,
        )
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.actions = self.actions.astype(np.int32)


    def step(self, actions):
        obs, rew, term, trunc, info = super().step(actions)

        if all(term) or all(trunc):
            self.reset()
            if 'agent_raw' in info:
                del info['agent_raw']
            if 'episode_rewards' in info:
                info['score'] = info['episode_rewards']
        # Don't discard info during non-terminal steps - this preserves heart.gained statistics
        # The original code set info = [] which lost all intermediate statistics
        
        return obs, rew, term, trunc, [info]

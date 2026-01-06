import functools
import numpy as np
import gymnasium

import pufferlib

from omegaconf import OmegaConf
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.replay_writer import ReplayWriter

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config='pufferlib/environments/metta/metta.yaml', render_mode='auto', buf=None, seed=0,
         ore_reward=0.17088483842567775, battery_reward=0.9882859711234822, heart_reward=1.0):
    '''Metta creation function'''
    
    OmegaConf.register_new_resolver("div", oc_divide, replace=True)
    cfg = OmegaConf.load(config)
    
    # Update rewards under the new structure: agent.rewards.inventory
    inventory_rewards = cfg['game']['agent']['rewards']['inventory']
    inventory_rewards['ore_red'] = float(ore_reward)
    inventory_rewards['heart'] = float(heart_reward)
    inventory_rewards['battery_red'] = float(battery_reward)
    
    curriculum = SingleTaskCurriculum('puffer', cfg)
    return MettaPuff(curriculum, render_mode=render_mode, buf=buf, seed=seed)

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
    def __init__(self, curriculum, render_mode='human', buf=None, seed=0):
        self.replay_writer = None
        #if render_mode == 'auto':
        #    self.replay_writer = ReplayWriter("metta/")

        super().__init__(
            curriculum=curriculum,
            render_mode=render_mode,
            buf=buf,
            replay_writer=self.replay_writer
        )
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.actions = self.actions.astype(np.int32)

    @property
    def single_action_space(self):
        return gymnasium.spaces.MultiDiscrete(super().single_action_space.nvec, dtype=np.int32)

    def step(self, actions):
        obs, rew, term, trunc, info = super().step(actions)

        if all(term) or all(trunc):
            self.reset()
            if 'agent_raw' in info:
                del info['agent_raw']
            if 'episode_rewards' in info:
                info['score'] = info['episode_rewards']

        else:
            info = []

        return obs, rew, term, trunc, [info]

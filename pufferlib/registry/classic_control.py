from gym.envs import classic_control

import torch

import pufferlib
import pufferlib.emulation
import pufferlib.utils
import pufferlib.models


Policy = pufferlib.models.Default

def make_cartpole_env():
    '''CartPole creation function
    
    This env is a useful test because it works without
    any additional dependencies 
    '''
    return pufferlib.emulation.GymPufferEnv(
        env=classic_control.CartPoleEnv()
    )
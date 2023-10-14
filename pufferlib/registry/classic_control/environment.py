from gymnasium.envs import classic_control

import pufferlib
import pufferlib.emulation


def env_creator(name):
    if name == 'cartpole':
        return classic_control.CartPoleEnv
    raise ValueError(f'Unknown environment: {name}')

def make_env(name='cartpole'):
    '''Create an environment by name'''
    env = env_creator(name)()
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

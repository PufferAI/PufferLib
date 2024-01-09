import pufferlib.emulation

from .squared import Squared
from .bandit import Bandit
from .memory import Memory
from .password import Password
from .stochastic import Stochastic


def env_creator(name='squared'):
    assert name in 'squared bandit memory password stochastic'.split(), 'Invalid environment name'
    if name == 'squared':
        return make_squared
    elif name == 'bandit':
        return make_bandit
    elif name == 'memory':
        return make_memory
    elif name == 'password':
        return make_password
    elif name == 'stochastic':
        return make_stochastic
    else:
        raise ValueError('Invalid environment name')

def make_squared(distance_to_target=3, num_targets=1):
    '''Puffer Squared environment'''
    env = Squared(distance_to_target=distance_to_target, num_targets=num_targets)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_bandit(num_actions=10, reward_scale=1, reward_noise=1):
    env = Bandit(num_actions=num_actions, reward_scale=reward_scale,
        reward_noise=reward_noise)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_memory(mem_length=2, mem_delay=2):
    env = Memory(mem_length=mem_length, mem_delay=mem_delay)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_password(password_length=5):
    env = Password(password_length=password_length)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_stochastic(p=0.7, horizon=100):
    env = Stochastic(p=p, horizon=100)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

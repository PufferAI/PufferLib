import functools
from nmmo3 import PuffEnv

def env_creator(name='nmmo3'):
    return functools.partial(make, name)

def make(name, num_envs=1):
    return PuffEnv(
        width=2*[2048],
        height=2*[2048],
        num_envs=2,
    )

import functools

import pufferlib
import pufferlib.emulation

def env_creator(name='Craftax-Symbolic-v1'):
    return functools.partial(make, name)

def make(name, num_envs=2048, buf=None):
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name(name, auto_reset=True)
    env_params = env.default_params
    return pufferlib.emulation.GymnaxPufferEnv(env, env_params, num_envs=num_envs, buf=buf)

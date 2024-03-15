# from pdb import set_trace as T
# import gymnasium
# import functools
# from pokegym import Environment
# import pufferlib.emulation

# def env_creator(name='pokemon_red'):
#     return functools.partial(make, name)

# def make(name, headless: bool = True, state_path=None):
#     '''Pokemon Red'''
#     env = Environment()
#     return pufferlib.emulation.GymnasiumPufferEnv(env=env,
#         postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
import sys
import os

# Ensure the directory of the correct environment.py is in sys.path
correct_path = '/bet_adsorption_xinpw8/back2bulba/pokegym/pokegym'
if correct_path not in sys.path:
    sys.path.insert(0, correct_path)
    
import functools
import pufferlib.emulation
from pokegym import Environment
from stream_wrapper import StreamWrapper

def env_creator(name="pokemon_red"):
    return functools.partial(make, name)

def make(name, **kwargs,):
    """Pokemon Red"""
    env = Environment(kwargs)
    env = StreamWrapper(env, stream_metadata={"user": "PUFFERBOX3 |BET|\npokegym\n"})
    # Looks like the following will optionally create the object for you
    # Or use the one you pass it. I'll just construct it here.
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor
    )
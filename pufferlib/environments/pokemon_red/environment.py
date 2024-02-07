from pdb import set_trace as T

import gymnasium
import functools

from pokegym import Environment

import pufferlib.emulation
from stream_agent_wrapper import StreamWrapper
import random
from pokegym import ram_map
import pyboy as PyBoy
from pokegym import Environment
from pokegym import ram_map, game_map
import multiprocessing
from pokegym.bin.ram_reader.red_memory_battle import *
from pokegym.bin.ram_reader.red_memory_env import *
from pokegym.bin.ram_reader.red_memory_items import *
from pokegym.bin.ram_reader.red_memory_map import *
from pokegym.bin.ram_reader.red_memory_menus import *
from pokegym.bin.ram_reader.red_memory_player import *
from enum import IntEnum

bet = '''
┌────────────┐
│░█▀▄░█▀▀░▀█▀│
│░█▀▄░█▀▀░░█░│
│░▀▀░░▀▀▀░░▀░│
└────────────┘
'''
fb = '''
01000110 01000010
01010010 01001001
01000101 01001100
01000101 01001100 
'''

def env_creator(name='pokemon_red'):
    return functools.partial(make, name)

def make(name, headless: bool = True, state_path=None):
    '''Pokemon Red'''
    env = Environment(headless=headless, state_path=state_path)
    env = StreamWrapper(
            env, 
            stream_metadata = {
                "user": '', # your username
                "env_id": '', # environment identifier
                "color": '#d622b5', # color for your text :)
                "extra": '', # any extra text you put here will be displayed THAT'S A LIE
            }
        )
    # wrapped_env = StreamWrapper(env)
    # return pufferlib.emulation.GymnasiumPufferEnv(env=wrapped_env,
    # postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)

        # pokemon_name = ram_map.pokemon(self.game)
        # self.name_poke = pokemon_name['0xD164']



# from pdb import set_trace as T

# import gymnasium
# import functools

# from pokegym import Environment

# import pufferlib.emulation


# def env_creator(name='pokemon_red'):
#     return functools.partial(make, name)

# def make(name, headless: bool = True, state_path=None):
#     '''Pokemon Red'''
#     env = Environment(headless=headless, state_path=state_path)
#     return pufferlib.emulation.GymnasiumPufferEnv(env=env,
#         postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
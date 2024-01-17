from pdb import set_trace as T
import numpy as np
import functools

import pufferlib
from pufferlib import namespace
import pufferlib.emulation
import pufferlib.environments


def env_creator(name='connect_four'):
    '''OpenSpiel creation function'''
    return functools.partial(make, name)

def make(
        name,
        multiplayer=False,
        n_rollouts=5,
        max_simulations=10,
        min_simulations=None
    ):
    '''OpenSpiel creation function'''
    pyspiel = pufferlib.environments.try_import('pyspiel', 'open_spiel')
    env = pyspiel.load_game(name)

    if min_simulations is None:
        min_simulations = max_simulations

    from pufferlib.environments.open_spiel.gymnasium_environment import (
        OpenSpielGymnasiumEnvironment
    )
    from pufferlib.environments.open_spiel.pettingzoo_environment import (
        OpenSpielPettingZooEnvironment
    )

    kwargs = dict(
        env=env,
        n_rollouts=int(n_rollouts),
        min_simulations=int(min_simulations),
        max_simulations=int(max_simulations),
    )
 
    if multiplayer:
        env = OpenSpielPettingZooEnvironment(**kwargs)
        wrapper_cls = pufferlib.emulation.PettingZooPufferEnv
    else:
        env = OpenSpielGymnasiumEnvironment(**kwargs)
        wrapper_cls = pufferlib.emulation.GymnasiumPufferEnv

    return wrapper_cls(
        env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor,
    )


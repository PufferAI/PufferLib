from pdb import set_trace as T
import numpy as np

import pufferlib
from pufferlib import namespace
import pufferlib.emulation
import pufferlib.environments

from pufferlib.environments.openspiel.gymnasium_environment import (
    OpenSpielGymnasiumEnvironment
)

from pufferlib.environments.openspiel.pettingzoo_environment import (
    OpenSpielPettingZooEnvironment
)


def env_creator():
    '''OpenSpiel creation function'''
    pyspiel = pufferlib.environment.try_import('pyspiel', 'openspiel')
    return pyspiel.load_game

def make_env(
        name='connect_four',
        multiplayer=False,
        n_rollouts=5,
        max_simulations=10,
        min_simulations=None
    ):
    '''OpenSpiel creation function'''
    if min_simulations is None:
        min_simulations = max_simulations

    env = env_creator()(name)

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


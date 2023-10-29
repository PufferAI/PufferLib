import gym

from pufferlib.environments.squared import init, reset, step, render

class Squared(gym.Env):
    __init__ = init
    reset = reset
    step = step
    render = render


def PokemonRed(*args, **kwargs):
    from .pokemon_red import PokemonRed
    return PokemonRed(*args, **kwargs)

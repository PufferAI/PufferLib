import gym

from pufferlib.environments.squared import init, reset, step, render

class Squared(gym.Env):
    __init__ = init
    reset = reset
    step = step
    render = render

def PokemonRed(*args, **kwargs):
    from .pokemon_red_updated import PokemonRed
    return PokemonRed(*args, **kwargs)

def PokemonRedMinimal(*args, **kwargs):
    from .pokemon_red_minimal import PokemonRed as PokemonRedMinimal
    return PokemonRedMinimal(*args, **kwargs)

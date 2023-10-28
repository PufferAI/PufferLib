from pdb import set_trace as T
import numpy as np
from pufferlib.environments import PokemonRed

env = PokemonRed()
r_ob, r_info = env.reset()
isinstance(r_ob, np.ndarray)
isinstance(r_info, dict)

for i in range(100):
    action = env.action_space.sample()
    ob, reward, terminal, truncated, info = env.step(action)
    print(f'Step: {i}, Info: {info}')

    # check datatype output
    isinstance(ob, np.ndarray)
    isinstance(reward, float)
    isinstance(terminal, bool)
    isinstance(truncated, bool)
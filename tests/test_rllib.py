from pdb import set_trace as T

import pufferlib
from environments import bindings


for binding in bindings:
    tuner = pufferlib.rllib.make_rllib_tuner(binding)
    result = tuner.fit()[0]
    print('Saved ', result.checkpoint)

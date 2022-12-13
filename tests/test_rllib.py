from pdb import set_trace as T

import pufferlib
import env_defs

for binding in env_defs.bindings:
    tuner = pufferlib.rllib.make_rllib_tuner(binding)
    result = tuner.fit()[0]
    print('Saved ', result.checkpoint)

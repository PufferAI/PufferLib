from pufferlib.environments import Squared
import pufferlib.models


Policy = pufferlib.models.Default

def make_squared(distance_to_target=3, num_targets=1):
    '''Puffer Diamond environment'''
    return pufferlib.emulation.GymPufferEnv(
        env_creator=Squared,
        env_args=[distance_to_target, num_targets],
    )

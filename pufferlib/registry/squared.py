from pufferlib import namespace
from pufferlib.environments import Squared
import pufferlib.models
from pufferlib.registry import RecurrentArgs
from pufferlib.registry import DefaultPolicyArgs as PolicyArgs


RECURRENCE_RECOMMENDED = False

env_creator = Squared

@pufferlib.dataclass
class EnvArgs:
    distance_to_target: int = 3
    num_targets: int = 1

Policy = pufferlib.models.Default

def make_env(**kwargs):
    '''Puffer Squared environment'''
    env = Squared(**kwargs)
    return pufferlib.emulation.GymPufferEnv(env=env)

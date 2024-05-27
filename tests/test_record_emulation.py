import pufferlib.emulation

from pufferlib.environments.ocean import env_creator

env = env_creator('spaces')()
env.reset()
env.step([1,0])
breakpoint()

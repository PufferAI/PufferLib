from pdb import set_trace as T
import pufferlib.emulation

from pufferlib.environments import PokemonRed as env_creator


def make_env():
    '''Pokemon Red'''
    env = env_creator()
    ob = env.reset()[0]
    # Save ob as rgb image
    #import cv2; cv2.imwrite('ob.png', ob.astype('uint8'))
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)

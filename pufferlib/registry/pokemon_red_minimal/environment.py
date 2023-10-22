from pdb import set_trace as T

import gymnasium

import pufferlib.emulation
from pufferlib.environments import PokemonRedMinimal as env_creator


def make_env(framestack=4):
    '''Pokemon Red'''
    env = env_creator()
    ob = env.reset()[0]
    # Save ob as rgb image
    #import cv2; cv2.imwrite('ob.png', ob.astype('uint8'))
    env = gymnasium.wrappers.FrameStack(env, framestack)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

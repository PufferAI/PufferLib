from pdb import set_trace as T
import gym

import pufferlib.emulation
import pufferlib.wrappers

import bsuite
from bsuite.utils import gym_wrapper


def env_creator():
    bsuite = pufferlib.environments.try_import('bsuite')
    return bsuite.load_and_record_to_csv

def make_env(name='bandit/0', results_dir='experiments/bsuite', overwrite=True):
    '''Puffer Squared environment'''
    env = env_creator()(name, results_dir=results_dir, overwrite=True)
    from bsuite.utils import gym_wrapper
    env = gym_wrapper.GymFromDMEnv(env)
    env = BSuiteStopper(env)
    env = pufferlib.wrappers.GymToGymnasium(env)
    env = pufferlib.emulation.GymnasiumPufferEnv(env)
    return env

class BSuiteStopper:
    def __init__(self, env):
        self.env = env
        self.num_episodes = 0

        self.step = self.env.step
        self.render = self.env.render
        self.close = self.env.close
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        '''Forces the environment to stop after the
        number of episodes required by bsuite'''
        self.num_episodes += 1

        if self.num_episodes >= self.env.bsuite_num_episodes:
            exit(0)

        return self.env.reset()

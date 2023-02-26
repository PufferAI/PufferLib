# TODO: Simple test for the speed of each envionment.
# Possibly include in the emulation performance test as a baseline

from pdb import set_trace as T

import numpy as np
import time

import gym

env = gym.make('BreakoutNoFrameskip-v4')

steps = 10000

ob = env.reset()

start = time.time()
for i in range(steps):
    ob, reward, done, info = env.step(env.action_space.sample())
import gym

from pufferlib.environments.squared import init, reset, step, render

class Squared(gym.Env):
    __init__ = init
    reset = reset
    step = step
    render = render

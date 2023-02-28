from pdb import set_trace as T

import numpy as np
import gym

import pufferlib
import pufferlib.registry


def test_env(env_creator, seed=42, steps=32):
    env_1 = env_creator()
    env_2 = env_creator()

    env_1.seed(seed)
    env_2.seed(seed)

    env_1.reset()
    env_2.reset()

    for i in range(steps):
        atn_1 = env_1.action_space.sample()
        atn_2 = env_2.action_space.sample()

        ob_1, reward_1, done_1, info_1 = env_1.step(atn_1)
        ob_2, reward_2, done_2, info_2 = env_2.step(atn_2)

        assert np.array_equal(ob_1, ob_2)
        assert reward_1 == reward_2
        assert done_1 == done_2
        assert info_1 == info_2


if __name__ == '__main__':
    test_env(lambda: gym.make('BreakoutNoFrameskip-v4'))
    #test_env(pufferlib.registry.Atari('BreakoutNoFrameskip-v4').raw_env_creator)
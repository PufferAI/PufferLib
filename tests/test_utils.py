import sys
import gym

import pufferlib
import pufferlib.utils

def test_suppress():
    with pufferlib.utils.Suppress():
        gym.make('Breakout-v4')
        print('stdout (you should not see this)', file=sys.stdout)
        print('stderr (you should not see this)', file=sys.stderr)

if __name__ == '__main__':
    test_suppress()
from pdb import set_trace as T

import time

import pettingzoo
import gym


def check_env(env_cls):
    assert issubclass(env_cls, gym.Env), "Not a gymnasium env (are you on old gym?)"

def is_dict_space(space):
    # Compatible with gym/gymnasium
    return type(atn_space).__name__ == 'Dict'

def is_multiagent(env_cls):
    if not issubclass(env_cls, pettingzoo.AECEnv) and not issubclass(env_cls, pettingzoo.ParallelEnv):
        assert issubclass(env_cls, gym.Env), 'Environment must subclass pettingzoo.AECEnv/ParallelEnv or gym.Env'
        return False
    return True

def current_datetime():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

def myprint(d):
    stack = d.items()
    while stack:
        k, v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.iteritems())
        else:
            print("%s: %s" % (k, v))

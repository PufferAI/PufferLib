from pdb import set_trace as T

import time

import pettingzoo
import gym


def check_env(env):
    #assert issubclass(env_cls, gym.Env), "Not a gymnasium env (are you on old gym?)"
    assert hasattr(env, 'possible_agents')
    assert len(env.possible_agents)
    obs_space = env.observation_space(env.possible_agents[0])
    atn_space = env.action_space(env.possible_agents[0])
    for e in env.possible_agents:
        assert env.observation_space(e) == obs_space, 'All agents must have same obs space'
        assert env.action_space(e) == atn_space, 'All agents must have same atn space'


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

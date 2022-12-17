from pdb import set_trace as T

import time
import os
import sys

import inspect

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

def is_multiagent(env):
    if inspect.isclass(env):
        env_cls = env
    else:
        env_cls = type(env)

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

class dotdict(dict):
    """dot.notation access to dictionary attributes
    
    TODO: Err on access bad key
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Suppress():
    def __init__(self):
        sys.stdout.flush()
        sys.stderr.flush()

        self.null_1 = os.open(os.devnull, os.O_WRONLY|os.O_TRUNC|os.O_CREAT)
        self.null_2 = os.open(os.devnull, os.O_WRONLY|os.O_TRUNC|os.O_CREAT)
   
    def __enter__(self):
        # Suppress C library outputs
        self.orig_stdout = os.dup(1)
        self.orig_stderr = os.dup(2)
        self.new_stdout = os.dup(1)
        self.new_stderr = os.dup(2)
        os.dup2(self.null_1, 1)
        os.dup2(self.null_2, 2)
        sys.stdout = os.fdopen(self.new_stdout, 'w')
        sys.stderr = os.fdopen(self.new_stderr, 'w')

        # Suppress Python outputs
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

       
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Enable C library outputs
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self.orig_stdout, 1)
        os.dup2(self.orig_stderr, 2)
        os.close(self.orig_stdout)
        os.close(self.orig_stderr)
        
        os.close(self.null_1)
        os.close(self.null_2)

        # Enable Python outputs
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stdout
from pdb import set_trace as T

import numpy as np

import time
import os
import sys

import inspect

import pettingzoo
import gym

# Define a SetupError exception class below
class SetupError(Exception):
    def __init__(self, env):
        self.message = f'{env}: Binding setup failed. Please ensure that the environment is installed correctly.'
        super().__init__(self.message)

def check_env(env):
    #assert issubclass(env_cls, gym.Env), "Not a gymnasium env (are you on old gym?)"
    assert hasattr(env, 'possible_agents')
    assert len(env.possible_agents)
    obs_space = env.observation_space(env.possible_agents[0])
    atn_space = env.action_space(env.possible_agents[0])
    for e in env.possible_agents:
        assert env.observation_space(e) == obs_space, 'All agents must have same obs space'
        assert env.action_space(e) == atn_space, 'All agents must have same atn space'

def _get_dtype_bounds(dtype):
    assert dtype in {np.float32, np.uint8}
    if dtype == np.uint8:
        return np.iinfo(dtype).min, np.iinfo(dtype).max
    elif dtype == np.float32:
        return np.finfo(dtype).min, np.finfo(dtype).max

def is_dict_space(space):
    # Compatible with gym/gymnasium
    return type(space).__name__ == 'Dict'

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

class RandomState:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def random(self):
        return self.rng.random()

    def probabilistic_round(self, n):
            frac, integer = np.modf(n)
            if self.random() < frac:
                return int(integer) + 1
            else:
                return int(integer)

    def sample(self, ary, n):
        n_rounded = self.probabilistic_round(n)
        return self.rng.choice(ary, n_rounded, replace=False).tolist()

    def choice(self, ary):
        return self.sample(ary, 1)[0]


class Profiler:    
    def __init__(self):
        self.elapsed = 0
        self.calls = 0

    def tik(self):
        self.start = time.time()

    def tok(self):
        self.elapsed += time.time() - self.start
        self.calls += 1

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed += self.end - self.start
        self.calls += 1

def profile(func):
    timer = Profiler()
    name = func.__name__ + '_timer'

    def wrapper(*args, **kwargs):
        with timer:
            result = func(*args, **kwargs)

        self = args[0]
        assert hasattr(self, 'timers'), 'Must have timers dict'
        self.timers[name] = timer
        return result

    return wrapper

class dotdict(dict):
    """dot.notation access to dictionary attributes
    
    TODO: Err on access bad key
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        # Return the current state of the object (the underlying dictionary)
        return self.copy()

    def __setstate__(self, state):
        # Restore the state of the object (the underlying dictionary)
        self.update(state)

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
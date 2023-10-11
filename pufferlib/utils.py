from pdb import set_trace as T

from collections import OrderedDict

import numpy as np

import time
import os
import sys
import pickle
import subprocess
from filelock import FileLock
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import inspect

import pettingzoo
import gym


def install_requirements(env):
    '''Pip install dependencies for specified environment'''
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "-e" f".[{env}]"]
    proc = subprocess.run(pip_install_cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"Error installing requirements: {proc.stderr}")

def check_env(env):
    #assert issubclass(env_cls, gym.Env), "Not a gymnasium env (are you on old gym?)"
    assert hasattr(env, 'possible_agents')
    assert len(env.possible_agents)
    obs_space = env.observation_space(env.possible_agents[0])
    atn_space = env.action_space(env.possible_agents[0])
    for e in env.possible_agents:
        assert env.observation_space(e) == obs_space, 'All agents must have same obs space'
        assert env.action_space(e) == atn_space, 'All agents must have same atn space'

def make_zeros_like(data):
    if isinstance(data, dict):
        return {k: make_zeros_like(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_zeros_like(v) for v in data]
    elif isinstance(data, np.ndarray):
        return np.zeros_like(data)
    elif isinstance(data, (int, float)):
        return 0
    else:
        raise ValueError(f'Unsupported type: {type(data)}')

def compare_arrays(array_1, array_2):
    assert isinstance(array_1, np.ndarray)
    assert isinstance(array_2, np.ndarray)
    assert array_1.shape == array_2.shape
    return np.allclose(array_1, array_2)

def compare_dicts(dict_1, dict_2, idx):
    assert isinstance(dict_1, (dict, OrderedDict))
    assert isinstance(dict_2, (dict, OrderedDict))

    if not all(k in dict_2 for k in dict_1):
        raise ValueError("Keys do not match between dictionaries.")

    for k, v in dict_1.items():
        if not compare_space_samples(v, dict_2[k], idx):
            return False

    return True

def compare_lists(list_1, list_2, idx):
    assert isinstance(list_1, (list, tuple))
    assert isinstance(list_2, (list, tuple))

    if len(list_1) != len(list_2):
        raise ValueError("Lengths do not match between lists/tuples.")

    for v1, v2 in zip(list_1, list_2):
        if not compare_space_samples(v1, v2, idx):
            return False
        
    return True
    
def compare_space_samples(sample_1, sample_2, sample_2_batch_idx=None):
    '''Compare two samples from the same space
    
    Optionally, sample_2 may be a batch of samples from the same space
    concatenated along the first dimension of the leaves. In this case,
    sample_2_batch_idx specifies which sample to compare.
    '''
    if isinstance(sample_1, (dict, OrderedDict)):
        return compare_dicts(sample_1, sample_2, sample_2_batch_idx)
    elif isinstance(sample_1, (list, tuple)):
        return compare_lists(sample_1, sample_2, sample_2_batch_idx)
    elif isinstance(sample_1, np.ndarray):
        assert isinstance(sample_2, np.ndarray)
        if sample_2_batch_idx is not None:
            sample_2 = sample_2[sample_2_batch_idx]
        return compare_arrays(sample_1, sample_2)
    elif isinstance(sample_1, (int, float)):
        if sample_2_batch_idx is not None:
            sample_2 = sample_2[sample_2_batch_idx]
        if isinstance(sample_2, np.ndarray):
            assert sample_2.size == 1, "Cannot compare scalar to non-scalar."
            sample_2 = sample_2[0]
        return sample_1 == sample_2
    else:
        raise ValueError(f"Unsupported type: {type(sample_1)}")

def _get_dtype_bounds(dtype):
    if dtype == bool:
        return 0, 1
    elif np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min, np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.unsignedinteger):
        return np.iinfo(dtype).min, np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        # Gym fails on float64
        return np.finfo(np.float32).min, np.finfo(np.float32).max
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

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
        self.prev = 0

    @property
    def serial(self):
        return {
            'elapsed': self.elapsed,
            'calls': self.calls,
            'delta': self.delta,
        }

    @property
    def delta(self):
        ret = self.elapsed - self.prev
        self.prev = self.elapsed
        return ret

    def tik(self):
        self.start = time.perf_counter()

    def tok(self):
        self.elapsed += time.perf_counter() - self.start
        self.calls += 1

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed += self.end - self.start
        self.calls += 1

def profile(func):
    name = func.__name__

    def wrapper(*args, **kwargs):
        self = args[0]

        if not hasattr(self, '_timers'):
            self._timers = {}

        if name not in self._timers:
            self._timers[name] = Profiler()

        timer = self._timers[name]

        with timer:
            result = func(*args, **kwargs)

        return result

    return wrapper

def aggregate_profilers(profiler_dicts):
    merged = {}

    for key in list(profiler_dicts[0].keys()):
        merged[key] = Profiler()
        for prof_dict in profiler_dicts:
            merged[key].elapsed += prof_dict[key].elapsed
            merged[key].calls += prof_dict[key].calls

    return merged

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
        self.f = StringIO()
        self.null_1 = os.open(os.devnull, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
        self.null_2 = os.open(os.devnull, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)

    def __enter__(self):
        # Suppress C library outputs
        self.orig_stdout = os.dup(1)
        self.orig_stderr = os.dup(2)
        os.dup2(self.null_1, 1)
        os.dup2(self.null_2, 2)

        # Suppress Python outputs
        self._stdout_redirector = redirect_stdout(self.f)
        self._stderr_redirector = redirect_stderr(self.f)
        self._stdout_redirector.__enter__()
        self._stderr_redirector.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Enable C library outputs
        os.dup2(self.orig_stdout, 1)
        os.dup2(self.orig_stderr, 2)
        os.close(self.orig_stdout)
        os.close(self.orig_stderr)
        os.close(self.null_1)
        os.close(self.null_2)

        # Enable Python outputs
        self._stdout_redirector.__exit__(exc_type, exc_val, exc_tb)
        self._stderr_redirector.__exit__(exc_type, exc_val, exc_tb)
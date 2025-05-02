from pdb import set_trace as T

from collections import OrderedDict
from contextlib import nullcontext

import numpy as np

import time
import os
import sys
import pickle
import subprocess
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from io import StringIO
import psutil

import warnings
from functools import wraps

import functools
import inspect
import importlib

def validate_args(fn, kwargs):
    fn_kwargs = get_init_args(fn)
    for param, val in kwargs.items():
        if param not in fn_kwargs:
            raise ValueError(
                f'Invalid argument\n{param}\nto\n{fn}\n'
                f'which takes \n{fn_kwargs}\n'
                f'Double check your config'
            )

def get_init_args(fn):
    if fn is None:
        return {}

    if isinstance(fn, functools.partial):
        return fn.keywords

    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in ['env', 'policy']:
            # Hack to avoid duplicate kwargs
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            kwargs[name] = param.default if param.default is not inspect.Parameter.empty else None
    return kwargs


def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v

def install_requirements(env):
    '''Pip install dependencies for specified environment'''
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "-e" f".[{env}]"]
    proc = subprocess.run(pip_install_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Error installing requirements: {proc.stderr}")

def install_and_import(package):
    '''Install and import a package'''
    try:
        module = importlib.import_module(package)
    except ImportError:
        install_requirements(package)
        module = importlib.import_module(package)

    return module

def silence_warnings(original_func, category=DeprecationWarning):
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=category)
            return original_func(*args, **kwargs)
    return wrapper

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
    import pettingzoo
    import gym
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

def format_bytes(size):
    if size >= 1024 ** 4:
        return f'{size / (1024 ** 4):.2f} TB'
    elif size >= 1024 ** 3:
        return f'{size / (1024 ** 3):.2f} GB'
    elif size >= 1024 ** 2:
        return f'{size / (1024 ** 2):.2f} MB'
    elif size >= 1024:
        return f'{size / 1024:.2f} KB'
    else:
        return f'{size} B'

# TODO: 5% perf gain by doing cuda sync less frequently
class Profiler:
    def __init__(self, elapsed=True, calls=True, memory=False,
            pytorch_memory=False, sync_cuda=True, frequency=10, amp_context=nullcontext()):
        self.elapsed = 0 if elapsed else None
        self.calls = 0 if calls else None
        self.memory = None
        self.pytorch_memory = None
        self.prev = 0
        self.delta = 0
        
        self.track_elapsed = elapsed
        self.track_calls = calls
        self.track_memory = memory
        self.track_pytorch_memory = pytorch_memory
        self.sync_cuda = sync_cuda
        self.frequency = frequency
        self.epoch = 0
        
        if memory:
            self.process = psutil.Process()

        if pytorch_memory or sync_cuda:
            import torch
            self.torch = torch

        self.amp_context = amp_context

    '''
    @property
    def serial(self):
        return {
            'elapsed': self.elapsed,
            'calls': self.calls,
            'memory': self.memory,
            'pytorch_memory': self.pytorch_memory,
            'delta': self.delta
        }

    @property
    def delta(self):
        ret = self.elapsed - self.prev if self.elapsed is not None else None
        self.prev = self.elapsed
        return ret
    '''

    def __call__(self, epoch):
        self.epoch = epoch
        return self

    def __enter__(self):
        if self.epoch % self.frequency != 0:
            return self

        if self.sync_cuda:
            self.torch.cuda.synchronize()
        self.amp_context.__enter__()
        if self.track_elapsed:
            self.start_time = time.perf_counter()
        if self.track_memory:
            self.start_mem = self.process.memory_info().rss
        if self.track_pytorch_memory:
            self.start_torch_mem = self.torch.cuda.memory_allocated()
        return self

    def __exit__(self, *args):
        if self.epoch % self.frequency != 0:
            return self

        self.amp_context.__exit__(None, None, None)
        if self.sync_cuda:
            self.torch.cuda.synchronize()
        if self.track_elapsed:
            self.end_time = time.perf_counter()
            self.delta += self.end_time - self.start_time
            self.elapsed += self.delta
        if self.track_calls:
            self.calls += 1
        if self.track_memory:
            self.end_mem = self.process.memory_info().rss
            self.memory = self.end_mem - self.start_mem
        if self.track_pytorch_memory:
            self.end_torch_mem = self.torch.cuda.memory_allocated()
            self.pytorch_memory = self.end_torch_mem - self.start_torch_mem

    def __repr__(self):
        parts = []
        if self.track_elapsed:
            parts.append(f'Elapsed: {self.elapsed:.4f} s')
        if self.track_calls:
            parts.append(f'Calls: {self.calls}')
        if self.track_memory:
            parts.append(f'Memory: {format_bytes(self.memory)}')
        if self.track_pytorch_memory:
            parts.append(f'PyTorch Memory: {format_bytes(self.pytorch_memory)}')
        return ", ".join(parts)

    # Aliases for use without context manager
    start = __enter__
    stop = __exit__

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

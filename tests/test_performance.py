from pdb import set_trace as T
import time
from tqdm import tqdm
import importlib
import random
import sys

import pufferlib
import pufferlib.utils
import pufferlib.exceptions
import pufferlib.emulation
import pufferlib.environments

import numpy as np

import pufferlib
from pufferlib.environments import ocean
from pufferlib.vector import Multiprocessing, Serial, Ray, make, autotune

import time
import psutil
import gymnasium

DEFAULT_TIMEOUT = 10

import time
from functools import wraps

class TimedEnv:
    def __init__(self, env):
        self._env = env
        self.reset_times = []
        self.step_times = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, *args, **kwargs):
        start = time.time()
        result = self._env.step(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        self.step_times.append(elapsed)
        return result

    def reset(self, *args, **kwargs):
        start = time.time()
        result = self._env.reset(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        self.reset_times.append(elapsed)
        return result

def profile_emulation(env_creator, timeout=DEFAULT_TIMEOUT, seed=42):
    reset_times = []
    step_times = []
    agent_step_count = 0
    terminal = False
    truncated = False
    reset = True

    random.seed(seed)
    np.random.seed(seed)

    env = env_creator()
    env.env = TimedEnv(env.env)
    multiagent = callable(env.action_space)

    start = time.time()
    while time.time() - start < timeout:
        if reset:
            s = time.time()
            ob, info = env.reset(seed=seed)
            reset_times.append(time.time() - s)

        if multiagent:
            action = {agent: env.action_space(agent).sample() for agent in ob}
            agent_step_count += len(env.agents)    
        else:
            action = env.action_space.sample()
            agent_step_count += 1

        s = time.time()
        ob, reward, terminal, truncated, info = env.step(action)
        step_times.append(time.time() - s)

        reset = (multiagent and len(env.agents) == 0) or terminal or truncated

    env.close()

    puf_total_reset = sum(reset_times)
    puf_total_step = sum(step_times)
    puf_reset_mean = np.mean(reset_times)
    puf_step_mean = np.mean(step_times)
    puf_step_std = np.std(step_times)

    raw_total_reset = sum(env.env.reset_times)
    raw_total_step = sum(env.env.step_times)
    raw_reset_mean = np.mean(env.env.reset_times)
    raw_step_mean = np.mean(env.env.step_times)
    raw_step_std = np.std(env.env.step_times)

    env_sps = agent_step_count / (puf_total_step + puf_total_reset)
    env_percent_reset = 100 * puf_total_reset / (puf_total_reset + puf_total_step)
    env_percent_step_std = 100 * puf_step_std / puf_step_mean
    env_overhead = 100 * (puf_total_step - raw_total_step + puf_total_reset - raw_total_reset) / (puf_total_step + puf_total_reset)


    print(f'    SPS        : {env_sps:.1f}')
    print(f'    Overhead   : {env_overhead:.2g}%')
    print(f'    Reset      : {env_percent_reset:.3g}%')
    print(f'    Step STD   : {env_percent_step_std:.3g}%')

def profile_puffer(env_creator, timeout=DEFAULT_TIMEOUT, **kwargs):
    vecenv = make(env_creator, **kwargs)
    actions = [vecenv.action_space.sample() for _ in range(1000)]

    agent_steps = 0
    vecenv.reset()
    start = time.time()
    while time.time() - start < timeout:
        vecenv.send(actions[agent_steps%1000])
        o, r, d, t, i, env_id, mask = vecenv.recv()
        agent_steps += sum(mask)

    sps = agent_steps / (time.time() - start)
    vecenv.close()

    backend = kwargs.get('backend', Serial)
    if backend == Multiprocessing and 'batch_size' in kwargs:
        print(f'    Puffer     : {(sps):.1f} - Pool')
    else:
        print(f'    Puffer     : {(sps):.1f} - {backend.__name__}')
    return sps

def profile_gymnasium_vec(env_creator, num_envs, timeout=DEFAULT_TIMEOUT):
    vecenv = gymnasium.vector.AsyncVectorEnv([env_creator] * num_envs)
    actions = [vecenv.action_space.sample() for _ in range(1000)]

    steps = 0
    vecenv.reset()
    start = time.time()
    while time.time() - start < timeout:
        vecenv.step(actions[steps%1000])
        steps += 1

    sps = steps * vecenv.num_envs / (time.time() - start)
    vecenv.close()

    print(f'    Gymnasium  : {(sps):.1f}')
    return sps

def profile_sb3_vec(env_creator, num_envs, timeout=DEFAULT_TIMEOUT):
    with pufferlib.utils.Suppress():
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vecenv = SubprocVecEnv([env_creator] * num_envs)
        actions = [[vecenv.action_space.sample() for _ in range(num_envs)]
            for _ in range(1000)]

        steps = 0
        vecenv.reset()
        start = time.time()
        while time.time() - start < timeout:
            vecenv.step(actions[steps%1000])
            steps += 1

        sps = steps * vecenv.num_envs / (time.time() - start)
        vecenv.close()

    print(f'    SB3        : {(sps):.1f}')
    return sps

def profile_all(name, env_creator, num_envs, num_workers=24,
        env_batch_size=None, zero_copy=True, timeout=DEFAULT_TIMEOUT):
    if env_batch_size is None:
        env_batch_size = num_envs

    print(name)
    profile_emulation(env_creator, timeout=timeout)
    profile_puffer(env_creator, num_envs=env_batch_size,
        backend=Multiprocessing, timeout=timeout,
        num_workers=min(num_workers, env_batch_size),
    )
    if env_batch_size is not None and env_batch_size != num_envs:
        profile_puffer(env_creator, num_envs=num_envs,
            backend=Multiprocessing, timeout=timeout, num_workers=num_workers,
            batch_size=env_batch_size, zero_copy=zero_copy
        )
    profile_gymnasium_vec(env_creator, num_envs=env_batch_size, timeout=timeout)
    profile_sb3_vec(env_creator, num_envs=env_batch_size, timeout=timeout)
    print()

if __name__ == '__main__':
    from pufferlib.environments import nocturne
    env_creator = nocturne.env_creator()
    profile_emulation(env_creator)
    #profile_puffer(env_creator, num_envs=8, backend=Multiprocessing)
    exit(0)

    from pufferlib.environments import vizdoom
    env_creator = vizdoom.env_creator()
    #profile_emulation(env_creator)
    profile_puffer(env_creator, num_envs=24,
        batch_size=8, backend=Multiprocessing, zero_copy=False)

    from pufferlib.environments import ocean
    env_creator = ocean.env_creator('grid')
    #profile_emulation(env_creator)

    import cProfile
    cProfile.run('profile_emulation(env_creator)', 'stats.profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('stats.profile')
    p.sort_stats(SortKey.TIME).print_stats(10)

    exit(0)

    from pufferlib.environments import nmmo
    print('Neural MMO')
    env_creator = nmmo.env_creator()
    profile_emulation(env_creator)
    #profile_puffer(env_creator, num_envs=8, backend=Multiprocessing)
    profile_puffer(env_creator, num_envs=96,
        batch_size=48, backend=Multiprocessing, zero_copy=False)
    print()

    from pufferlib.environments import nethack
    profile_all('NetHack', nethack.env_creator(), num_envs=48)

    from pufferlib.environments import minihack
    profile_all('MiniHack', minihack.env_creator(), num_envs=48)

    from pufferlib.environments import pokemon_red
    profile_all('Pokemon Red', pokemon_red.env_creator(),
        num_envs=144, env_batch_size=48, zero_copy=False)

    from pufferlib.environments import procgen
    profile_all('ProcGen', procgen.env_creator('bigfish'),
        num_envs=144, env_batch_size=48, num_workers=24, zero_copy=False)

    from pufferlib.environments import classic_control
    profile_all('Classic Control', classic_control.env_creator(),
        num_envs=1152, env_batch_size=48)

    from pufferlib.environments import ocean
    profile_all('Ocean Squared', ocean.env_creator('squared'),
        num_envs=1152, env_batch_size=48)

    from pufferlib.environments import atari
    profile_all('Atari Breakout', atari.env_creator('BreakoutNoFrameskip-v4'),
        num_envs=144, env_batch_size=48, zero_copy=False)

    from pufferlib.environments import crafter
    profile_all('Crafter', crafter.env_creator(),
        num_envs=24, env_batch_size=8, zero_copy=False)

    from pufferlib.environments import minigrid
    profile_all('MiniGrid', minigrid.env_creator(),
        num_envs=192, env_batch_size=48, zero_copy=False)

    exit(0)

    '''
    # Small scale version for laptop
    from pufferlib.environments import nmmo
    print('Neural MMO')
    env_creator = nmmo.env_creator()
    profile_emulation(env_creator)
    profile_puffer(env_creator, num_envs=4, num_workers=4, backend=Multiprocessing)
    profile_puffer(env_creator, num_envs=12, num_workers=6,
        batch_size=4, backend=Multiprocessing)
    print()

    from pufferlib.environments import nethack
    profile_all('NetHack', nethack.env_creator(), num_envs=12, num_workers=6)

    from pufferlib.environments import minihack
    profile_all('MiniHack', minihack.env_creator(), num_envs=12, num_workers=6)

    from pufferlib.environments import pokemon_red
    profile_all('Pokemon Red', pokemon_red.env_creator(),
        num_envs=36, num_workers=6, env_batch_size=12, zero_copy=False)

    from pufferlib.environments import classic_control
    profile_all('Classic Control', classic_control.env_creator(),
        num_envs=36, num_workers=6, env_batch_size=12, zero_copy=False)

    from pufferlib.environments import ocean
    profile_all('Ocean Squared', ocean.env_creator('squared'),
        num_envs=36, num_workers=6, env_batch_size=12, zero_copy=False)

    from pufferlib.environments import atari
    profile_all('Atari Breakout', atari.env_creator('BreakoutNoFrameskip-v4'),
        num_envs=36, num_workers=6, env_batch_size=12, zero_copy=False)

    from pufferlib.environments import crafter
    profile_all('Crafter', crafter.env_creator(),
        num_envs=12, num_workers=6, env_batch_size=4, zero_copy=False)

    from pufferlib.environments import minigrid
    profile_all('MiniGrid', minigrid.env_creator(),
        num_envs=36, num_workers=6, env_batch_size=12, zero_copy=False)

    exit(0)
    '''

    #from functools import partial
    #counts = [1e5, 1e6, 1e7, 1e8]
    #delays = [0, 0.1, 0.25, 0.5, 1]
    #bandwidth = [1, 1e4, 1e5, 1e6]

    
    #synthetic_creators = {}
    #for count in counts:
    #    name = f'test_delay_{count}'

    #env_creators.test = partial(
    #    ocean.env_creator('performance_empiric'),
    #    count_n=270_000, bandwidth=150_000
    #)

    #timeout = 60
    #cores = psutil.cpu_count(logical=False)
    #for key, creator in env_creators.items():
    #    prof = profile_emulation(creator, timeout)
    #    profile_vec(creator, cores, timeout, prof.puf.sps)
    #    print()

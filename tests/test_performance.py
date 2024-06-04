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


    print(f'{env_sps:.1f} SPS | {env_overhead:.2g}% Overhead')
    print(f'    Reset: {env_percent_reset:.3g}%')
    print(f'    Step STD: {env_percent_step_std:.3g}%')

def profile_vec(vecenv, timeout=DEFAULT_TIMEOUT):
    actions = [vecenv.action_space.sample() for _ in range(1000)]
    vecenv.reset()

    steps = 0
    start = time.time()
    while time.time() - start < timeout:
        vecenv.step(actions[steps%1000])
        steps += 1

    sps = steps * vecenv.num_envs / (time.time() - start)
    vecenv.close()
    return sps

def profile_puffer(env_creator, timeout=DEFAULT_TIMEOUT, **kwargs):
    env = make(env_creator, **kwargs)
    sps = profile_vec(env, timeout)
    backend = kwargs.get('backend', Serial)
    print(f'    Puffer     : {(sps):.3f} - {backend.__name__}')
    return sps

def profile_gymnasium_vec(env_creator, num_envs, timeout=DEFAULT_TIMEOUT):
    env = gymnasium.vector.AsyncVectorEnv([env_creator] * num_envs)
    sps = profile_vec(env, timeout)
    print(f'    Gymnasium  : {(sps):.3f}')
    return sps

def profile_sb3_vec(env_creator, num_envs, timeout=DEFAULT_TIMEOUT):
    with pufferlib.utils.Suppress():
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([env_creator] * num_envs)
        sps = profile_vec(env, timeout)

    print(f'    SB3        : {(sps):.3f}')
    return sps

def sanity_check(env_creator, timeout=5):
    profile_emulation(env_creator, timeout)
    for backend in [Serial, Multiprocessing]:#, Ray]:
        profile_puffer(env_creator, backend=backend, timeout=timeout)
    #profile_gymnasium_vec(env_creator, num_envs=1, timeout=timeout)
    #profile_sb3_vec(env_creator, num_envs=1, timeout=timeout)

if __name__ == '__main__':
    from pufferlib.environments import pokemon_red
    env_creator = pokemon_red.env_creator('pokemon_red')
    print('Sanity: Pokemon Red')
    profile_emulation(env_creator, timeout=30)
    exit(0)
    '''
    sanity_check(env_creator)
    autotune(env_creator, batch_size=32, max_envs=96)
    exit(0)

    from pufferlib.environments import classic_control
    env_creator = classic_control.env_creator()
    print('Sanity: Classic Control')
    sanity_check(env_creator)

    from pufferlib.environments import ocean
    env_creator = ocean.env_creator('spaces')
    print('Sanity: Ocean Spaces')
    sanity_check(env_creator)

    from pufferlib.environments import atari
    env_creator = atari.env_creator('BreakoutNoFrameskip-v4')
    print('Sanity: Atari Breakout')
    sanity_check(env_creator)
    autotune(env_creator, batch_size=48, max_envs=192)
    exit(0)

    from pufferlib.environments import crafter
    env_creator = crafter.env_creator()
    print('Sanity: Crafter')
    sanity_check(env_creator)

    from pufferlib.environments import minigrid
    env_creator = minigrid.env_creator()
    print('Sanity: MiniGrid')
    sanity_check(env_creator)
    '''

    from pufferlib.environments import nethack
    env_creator = nethack.env_creator()
    print('Sanity: NetHack')
    autotune(env_creator, batch_size=48)
    sanity_check(env_creator)
    exit(0)

    from pufferlib.environments import nmmo3
    env_creator = nmmo3.env_creator()
    print('Sanity: NMMO3')
    sanity_check(env_creator)
 
    exit(0)

    #from pufferlib.environments import nmmo3
    #env_creators.nmmo3 = nmmo3.env_creator()

    #result = profile_puffer(env_creators.nmmo3, num_envs=1, timeout=10)
    #print(f'    Puffer Serial: {(result):.3f}')
    #profile_vec(env_creators.nmmo3, 
        #num_envs=18, num_workers=6, batch_size=6, timeout=10)
    #exit(0)
 
    #profile_vec(env_creators.nmmo3, 
    #    num_envs=1, num_workers=1, batch_size=1, timeout=10)
    #exit(0)
    #profile_vec(env_creators.nmmo3, 2, 10, 1000)

    #from pufferlib.environments import pokemon_red
    #env_creators.pokemon_red = pokemon_red.env_creator('pokemon_red')
    #profile_vec(env_creators.pokemon_red,
    #    num_envs=1, num_workers=1, batch_size=1, timeout=10)

    #exit(0)

    # 20k on Nethack on laptop via 1 worker per batch
    # not triggering a giant copy
    #from pufferlib.environments import nethack
    #env_creators.nethack = nethack.env_creator()
    #profile_vec(env_creators.nethack, num_envs=6,
    #    num_workers=6, batch_size=1, timeout=10)
    #    num_workers=6, batch_size=1, timeout=10)
    #exit(0)
    #profile_vec(env_creators.nethack, 1, 10, 20000)


    from pufferlib.environments import ocean
    env_creators.ocean_spaces = ocean.env_creator('spaces')
    result = profile_puffer(env_creators.ocean_spaces, num_envs=1, timeout=10)
    result = profile_gymnasium_vec(env_creators.ocean_spaces, num_envs=1, timeout=10)
    exit(0)

    from functools import partial
    env_creators.test = partial(
        ocean.env_creator('performance_empiric'),
        count_n=20_000, bandwidth=208_000,
    )


    profile_vec(env_creators.test,
        num_envs=3, num_workers=3, batch_size=1, timeout=5)


    '''
    import cProfile
    cProfile.run('profile_vec(env_creators.nmmo3, 6, 3, 1)', 'profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('profile')
    p.sort_stats(SortKey.TIME).print_stats(10)
    T()
    '''

    #profile_environment(env_creators.nmmo3, 5)
    #profile_vec(env_creators.nmmo3, 1, 10, 1)
    #profile_vec(env_creators.nethack, 5, 10, 1)


    exit(0)

    from functools import partial
    counts = [1e5, 1e6, 1e7, 1e8]
    delays = [0, 0.1, 0.25, 0.5, 1]
    bandwidth = [1, 1e4, 1e5, 1e6]

    
    synthetic_creators = {}
    for count in counts:
        name = f'test_delay_{count}'

    env_creators.test = partial(
        ocean.env_creator('performance_empiric'),
        count_n=270_000, bandwidth=150_000
    )

    timeout = 60
    cores = psutil.cpu_count(logical=False)
    for key, creator in env_creators.items():
        prof = profile_emulation(creator, timeout)
        #profile_vec(creator, cores, timeout, prof.puf.sps)
        print()

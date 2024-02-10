from pdb import set_trace as T
import time
from tqdm import tqdm
import importlib
import sys

import pufferlib
import pufferlib.utils
import pufferlib.exceptions
import pufferlib.emulation
import pufferlib.environments

import numpy as np

import pufferlib
from pufferlib.environments import ocean
from pufferlib.vectorization import Multiprocessing

import time
import psutil
import gymnasium



def profile_environment(env_creator, timeout):
    reset_times = []
    step_times = []
    agent_step_count = 0
    terminal = False
    truncated = False
    reset = True

    env = env_creator()
    multiagent = callable(env.action_space)

    start = time.time()
    while time.time() - start < timeout:
        if reset:
            s = time.time()
            ob, info = env.reset()
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

    total_reset = sum(reset_times)
    total_step = sum(step_times)
    reset_mean = np.mean(reset_times)
    step_mean = np.mean(step_times)
    step_std = np.std(step_times)

    return pufferlib.namespace(
        sps=len(step_times) / (total_step + total_reset),
        percent_reset=100 * total_reset / (total_reset + total_step),
        reset_mean=reset_mean,
        percent_step_std=100 * step_std / step_mean,
        step_mean=step_mean,
        step_std=step_std,
    )

def profile_emulation(raw_creator, puf_creator, timeout):
    raw = profile_environment(raw_creator, timeout)
    puf = profile_environment(puf_creator, timeout)

    overhead = 100 * (puf.sps - raw.sps) / raw.sps
    print(f'{key} - {puf.sps:.1f}/{raw.sps:.1f} SPS (puf/raw) | {overhead:.2g}% Overhead')
    print('  Emulation')
    print(f'    Raw Reset  : {raw.reset_mean:.3g} ({raw.percent_reset:.2g})%')
    print(f'    Puf Reset  : {puf.reset_mean:.3g} ({puf.percent_reset:.2g})%')
    print(f'    Raw Step   : {raw.step_mean:.3g} +/- {raw.step_std:.3g} ({raw.percent_step_std:.2f})%')
    print(f'    Puf Step   : {puf.step_mean:.3g} +/- {puf.step_std:.3g} ({puf.percent_step_std:.2f})%')

    return pufferlib.namespace(raw=raw, puf=puf)


def profile_puffer_vec(env_creator, num_envs, timeout, sps):
    envs_per_worker = max(int(sps/500), 1)
    num_envs = int(num_envs * envs_per_worker)
    env = Multiprocessing(env_creator,
        num_envs=num_envs, envs_per_worker=envs_per_worker)
    env.reset()
    steps = 0

    actions = [env.single_action_space.sample() for _ in range(num_envs)]
    start = time.time()
    while time.time() - start < timeout:
        obs = env.step(actions)[0]
        steps += num_envs

    sps = steps / (time.time() - start)
    env.close()
    return sps

def profile_puffer_pool_vec(env_creator, num_envs, timeout, sps):
    envs_per_worker = max(int(sps/1000), 1)
    num_envs = int(num_envs * envs_per_worker)
    env = Multiprocessing(env_creator, num_envs=3*num_envs,
        envs_per_worker=envs_per_worker, envs_per_batch=num_envs, env_pool=True)
    env.async_reset()
    steps = 0

    actions = [env.single_action_space.sample() for _ in range(num_envs)]
    start = time.time()
    while time.time() - start < timeout:
        env.recv()
        env.send(actions)
        steps += num_envs

    sps = steps / (time.time() - start)
    env.close()
    return sps

def profile_gymnasium_vec(env_creator, num_envs, timeout):
    env = gymnasium.vector.AsyncVectorEnv([env_creator] * num_envs)
    env.reset()
    steps = 0
    actions = env.action_space.sample()
    start = time.time()
    while time.time() - start < timeout:
        env.step(actions)
        steps += num_envs

    sps = steps / (time.time() - start)
    env.close()
    return sps

def profile_sb3_vec(env_creator, num_envs, timeout):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = SubprocVecEnv([env_creator] * num_envs)
    env.reset()
    steps = 0

    actions = [env.action_space.sample() for _ in range(num_envs)]
    start = time.time()
    while time.time() - start < timeout:
        env.step(actions)
        steps += num_envs

    sps = steps / (time.time() - start)
    env.close()
    return sps

def profile_vec(env_creator, cores, timeout, sps):
    print(f'  Vectorization - {cores} core speed factor')

    result = profile_puffer_vec(env_creator, cores, timeout, sps)
    print(f'    Pufferlib  : {(result/sps):.3f}')

    puf_async = profile_puffer_pool_vec(env_creator, cores, timeout, sps)
    print(f'    Puffer Pool: {(puf_async/sps):.3f}')

    with pufferlib.utils.Suppress():
        sb3 = profile_sb3_vec(env_creator, cores, timeout)
    print(f'    SB3        : {(sb3/sps):.3f}')

    gym = profile_gymnasium_vec(env_creator, cores, timeout)
    print(f'    Gymnasium  : {(gym/sps):.3f}')


if __name__ == '__main__':
    env_creators = pufferlib.namespace()

    '''
    from pufferlib.environments import pokemon_red
    env_creators.pokemon_red = pokemon_red.env_creator('pokemon_red')

    from pufferlib.environments import classic_control
    env_creators.cartpole = classic_control.env_creator()

    from pufferlib.environments import ocean
    env_creators.ocean_squared = ocean.env_creator('squared')

    from pufferlib.environments import atari
    env_creators.atari_breakout = atari.env_creator('BreakoutNoFrameskip-v4')

    from pufferlib.environments import nethack
    env_creators.nethack = nethack.env_creator()

    from pufferlib.environments import crafter
    env_creators.crafter = crafter.env_creator()

    from pufferlib.environments import minigrid
    env_creators.minigrid = minigrid.env_creator()
    '''

    from pufferlib.environments import pokemon_red
    env_creators.pokemon_red = pokemon_red.env_creator('pokemon_red')

    timeout = 5
    cores = psutil.cpu_count(logical=False)
    for key, creator in env_creators.items():
        prof = profile_emulation(creator, creator, timeout)
        profile_vec(creator, cores, timeout, prof.puf.sps)
        print()

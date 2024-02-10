from pdb import set_trace as T
import numpy as np

import pufferlib
from pufferlib.environments import ocean
from pufferlib.vectorization import Multiprocessing

import time
import psutil
import gymnasium

def test_env_performance(env_creator, timeout=10):
    env = env_creator()
    reset_times = []
    step_times = []

    timeout_start = time.time()
    start = time.time()
    env.reset()
    reset_times.append(time.time() - start)

    while time.time() - timeout_start < timeout:
        action = env.action_space.sample()
        start = time.time()
        _, _, terminal, truncated, _ = env.step(action)
        step_times.append(time.time() - start)

        if terminal or truncated:
            start = time.time()
            env.reset()
            reset_times.append(time.time() - start)

    env.close()

    total_reset = sum(reset_times)
    total_step = sum(step_times)
    reset_mean = np.mean(reset_times)
    step_mean = np.mean(step_times)
    step_std = np.std(step_times)

    return pufferlib.namespace(
        sps=len(step_times) / (total_step + total_reset),
        percent_reset=total_reset / (total_reset + total_step),
        reset_mean=reset_mean,
        percent_step_std=step_std / step_mean,
        step_mean=step_mean,
        step_std=step_std,
    )


'''
def test_serial(env_creator, num_envs, steps):
    start = time.time()
    for _ in range(num_envs):
        env = env_creator()
        env.reset()
        actions = env.action_space.sample()
        for _ in range(steps):
            _, _, terminal, truncated, _ = env.step(actions)
            if terminal or truncated:
                env.reset()
        env.close()
    return steps * num_envs / (time.time() - start)
'''

def test_pufferlib(env_creator, num_envs, timeout, sps):
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

def test_pufferlib_async(env_creator, num_envs, timeout, sps):
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

def test_gymnasium(env_creator, num_envs, timeout):
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

def test_stable_baselines3(env_creator, num_envs, timeout):
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

def calibrate(n, trials=10):
    timings = []
    for _ in range(trials):
        start = time.time()
        idx = 0
        while idx < n:
            idx += 1
        end = time.time()
        timings.append(end - start)
    return sum(timings) / trials

def run_tests(env_creator, cores, sps, timeout=10):
    #serial_time = test_serial(env_creator, cores, sps)
    puf_time = test_pufferlib(env_creator, cores, timeout, sps)
    puf_async_time = test_pufferlib_async(env_creator, cores, timeout, sps)
    with pufferlib.utils.Suppress():
        sb3_time = test_stable_baselines3(env_creator, cores, timeout)
    gym_time = test_gymnasium(env_creator, cores, timeout)

    return pufferlib.namespace(puf=puf_time, puf_async=puf_async_time,
        sb3=sb3_time, gym=gym_time)

    #return pufferlib.namespace(serial=serial_time, puf=puf_time,
    #    puf_async=puf_async_time, sb3=sb3_time, gym=gym_time)

if __name__ == '__main__':
    # PufferLib has a good way to pass env args directly but other libraries don't
    #print(calibrate(3_300_000))
    #exit()

    # Synthetic environments
    #delay_means = [1e-1]*3 + [1e-2]*3 + [1e-3]*3 + [1e-4]*3
    #delay_stds = 4*[0.0, 0.25, 1.0]

    '''
    delay_means = [1e-1, 1e-2, 1e-3, 1e-4]
    delay_stds = [0.0, 0.0, 0.0, 0.0]
    #delay_stds = [0.25, 0.25, 0.25, 0.25]

    delay_means = [1e-1]
    delay_stds = [0.0]

    num_envs = 4

    synthetic_results = []
    for mean, std in zip(delay_means, delay_stds):
        env_creator = lambda: ocean.env_creator('performance_empiric')(
            count_n=int(75_000_000*mean), count_std=std)
        result = run_tests(env_creator, num_envs, int(10/mean))
        synthetic_results.append(result)
    '''

    results = pufferlib.namespace()
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

    from pufferlib.environments import nethack
    env_creators.nethack = nethack.env_creator()


    cores = psutil.cpu_count(logical=False)
    for key, creator in env_creators.items():
        r = test_env_performance(creator)
        print(f'Environment: {key} ({r.sps:.1f} SPS)')

        result = run_tests(creator, cores, r.sps, timeout=5)
        print(f'Reset: {r.reset_mean:.3g} ({100*r.percent_reset:.2g})%')
        print(f'Step: {r.step_mean:.3g} +/- {r.step_std:.3g} ({100*r.percent_step_std:.2f})%')
        print(f'Pufferlib: {(result.puf/r.sps):.3f}')
        print(f'Pufferlib Async: {(result.puf_async/r.sps):.3f}')
        print(f'Stable Baselines 3: {(result.sb3/r.sps):.3f}')
        print(f'Gymnasium: {(result.gym/r.sps):.3f}')
        print()


    exit(0)

    for key, r in results.items():
        print(f'Environment: {key} ({r.sps:.1f} SPS)')
        print(f'Reset: {r.reset_mean:.3g} ({100*r.percent_reset:.2g})%')
        print(f'Step: {r.step_mean:.3g} +/- {r.step_std:.3g} ({100*r.percent_step_std:.2f})%')
        print()


    for mean, std, result in zip(delay_means, delay_stds, synthetic_results):
        print(f'Mean: {mean:.3f}, Std: {std:.3f}')
        print(f'Serial: {result.serial:.3f}')
        print(f'Pufferlib: {result.puf:.3f}')
        print(f'Pufferlib Async: {result.puf_async:.3f}')
        print(f'Stable Baselines 3: {result.sb3:.3f}')
        print(f'Gymnasium: {result.gym:.3f}')
        print()

    #env_creator = lambda: ocean.env_creator('performance_empiric')(
    #    count_n=270_000, count_std=0)

    # TODO: What is wrong with pokegym?
    #from pufferlib.environments import pokemon_red
    #env_creator = pokemon_red.env_creator('pokemon_red')

    #from pufferlib.environments import atari
    #env_creator = atari.env_creator('BreakoutNoFrameskip-v4')

    # Need to tune envs per worker
    #from pufferlib.environments import nethack
    #env_creator = nethack.env_creator()


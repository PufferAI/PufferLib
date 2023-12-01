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

import config

def test_environment(name, timeout=5):
    try:
        module = importlib.import_module(f'pufferlib.environments.{name}')
    except pufferlib.exceptions.SetupError as e:
        raise

    total_steps = 0
    raw_env = module.make_env()
    # TODO: Figure out seeding
    raw_obs, raw_infos = raw_env.reset()
    raw_profiler = pufferlib.utils.Profiler()

    puf_env = module.make_env()
    puf_obs, puf_infos = puf_env.reset(seed=0)
    puf_profiler = pufferlib.utils.Profiler()

    if isinstance(puf_env, pufferlib.emulation.GymnasiumPufferEnv):
        multiagent = raw_done = puf_done = False
    elif isinstance(puf_env, pufferlib.emulation.PettingZooPufferEnv):
        multiagent = True
    else:
        raise TypeError(f'Unknown environment type: {type(puf_env)}')

    start = time.time()
    step = 0
    while time.time() - start < timeout:
        step += 1
        if multiagent:
            total_steps += len(raw_obs)
            raw_actions = {}
            for agent, ob in raw_obs.items():
                raw_actions[agent] = raw_env.env.action_space(agent).sample()
            with raw_profiler:
                if len(raw_env.env.agents) == 0:
                    raw_obs, raw_infos = raw_env.env.reset(seed=step)
                else:
                    raw_obs, raw_rewards, raw_dones, raw_truncateds, raw_infos = raw_env.env.step(raw_actions)

            puf_actions = {}
            for agent, ob in puf_obs.items():
                puf_actions[agent] = puf_env.action_space(agent).sample()
            with puf_profiler:
                if len(puf_env.agents) == 0:
                    puf_obs, puf_infos = puf_env.env.reset(seed=step)
                else:
                    puf_obs, puf_rewards, puf_dones, puf_truncateds, puf_infos = puf_env.step(puf_actions)
        else:
            total_steps += 1
            raw_action = raw_env.env.action_space.sample()
            with raw_profiler:
                if raw_done:
                    raw_ob, raw_info = raw_env.env.reset()
                    raw_done = False
                else:
                    raw_ob, raw_reward, raw_done, raw_truncated, raw_info = raw_env.env.step(raw_action)

            puf_action = puf_env.action_space.sample()
            with puf_profiler:
                if puf_done:
                    puf_ob, puf_info = puf_env.reset()
                    puf_done = False
                else:
                    puf_ob, puf_reward, puf_done, puf_truncated, puf_info = puf_env.step(puf_action)

    print(
        f'{name} - Performance Factor: {raw_profiler.elapsed / puf_profiler.elapsed:<4.3}\n' 
        f'Raw SPS: {total_steps // raw_profiler.elapsed}, Puffer SPS: {total_steps // puf_profiler.elapsed}\n'
        f'Total steps: {total_steps}, Time Elapsed: {(time.time()-start):.2f}s\n'
    )

if __name__ == '__main__':
    env = sys.argv[1]
    test_environment(env)

from pdb import set_trace as T
import time
from tqdm import tqdm

import pufferlib
import pufferlib.utils
import pufferlib.exceptions
import pufferlib.emulation

import config


def test_environments(steps=1_000_000, timeout=30):
    config_fns = config.all()
    for name, config_fn in config_fns.items():
        try:
            print('Setting up', name)
            config_fn()
        except pufferlib.exceptions.SetupError as e:
            print(e.message)
            continue

        total_steps = 0
        for creator, kwargs in config_fn().env_creators.items():
            raw_env = creator(**kwargs)
            raw_obs = raw_env.reset(0)
            raw_profiler = pufferlib.utils.Profiler()

            puf_env = creator(**kwargs)
            puf_obs = puf_env.reset(seed=0)
            puf_profiler = pufferlib.utils.Profiler()

            if isinstance(puf_env, pufferlib.emulation.GymPufferEnv):
                multiagent = raw_done = puf_done = False
            elif isinstance(puf_env, pufferlib.emulation.PettingZooPufferEnv):
                multiagent = True
            else:
                raise TypeError(f'Unknown environment type: {type(puf_env)}')

            start = time.time()
            for step in tqdm(range(steps)):
                if time.time() - start > timeout:
                    break

                if multiagent:
                    total_steps += len(raw_obs)
                    raw_actions = {}
                    for agent, ob in raw_obs.items():
                        raw_actions[agent] = raw_env.env.action_space(agent).sample()
                    with raw_profiler:
                        if len(raw_env.env.agents) == 0:
                            raw_obs = raw_env.env.reset(seed=step)
                        else:
                            raw_obs, raw_rewards, raw_dones, raw_infos = raw_env.env.step(raw_actions)

                    puf_actions = {}
                    for agent, ob in puf_obs.items():
                        puf_actions[agent] = puf_env.action_space(agent).sample()
                    with puf_profiler:
                        if len(puf_env.agents) == 0:
                            puf_obs = puf_env.env.reset(seed=step)
                        else:
                            puf_obs, puf_rewards, puf_dones, puf_infos = puf_env.step(puf_actions)
                else:
                    total_steps += 1
                    raw_action = raw_env.env.action_space.sample()
                    with raw_profiler:
                        if raw_done:
                            raw_ob = raw_env.env.reset()
                            raw_done = False
                        else:
                            raw_ob, raw_reward, raw_done, raw_info = raw_env.env.step(raw_action)

                    puf_action = puf_env.action_space.sample()
                    with puf_profiler:
                        if puf_done:
                            puf_ob = puf_env.reset()
                            puf_done = False
                        else:
                            puf_ob, puf_reward, puf_done, puf_info = puf_env.step(puf_action)

            env_name = f'{name} - {creator.__name__}'
            print(
                f'{env_name} - Performance Factor: {raw_profiler.elapsed / puf_profiler.elapsed:<4.3}\n' 
                f'Raw SPS: {total_steps // raw_profiler.elapsed}, Puffer SPS: {total_steps // puf_profiler.elapsed}\n'
                f'Total steps: {total_steps}, Time Elapsed: {(time.time()-start):.2f}s\n'
            )

if __name__ == '__main__':
    test_environments()
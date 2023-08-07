from pdb import set_trace as T

import pufferlib
import pufferlib.utils
import pufferlib.emulation

import config


def test_environments():
    config_fns = config.all()
    for name, config_fn in config_fns.items():
        try:
            print('Setting up', name)
            config_fn()
        except pufferlib.utils.SetupError as e:
            print(e.message)
            continue

        conf = config_fn()
        for creator, kwargs in conf.env_creators.items():
            env = creator(**kwargs)
            obs = env.reset()

            if isinstance(env, pufferlib.emulation.GymPufferEnv):
                assert env.observation_space.contains(obs)
                actions = env.action_space.sample()
            elif isinstance(env, pufferlib.emulation.PettingZooPufferEnv):
                actions = {}
                for agent, ob in obs.items():
                    assert env.observation_space(agent).contains(ob)
                    actions[agent] = env.action_space(agent).sample()
            else:
                raise TypeError(f'Unknown environment type: {type(env)}')

            obs, rewards, dones, infos = env.step(actions)

def test_bindings_performance(steps=1000):
    for binding in make_all_bindings().values():
        raw_profiler = pufferlib.utils.Profiler()
        puf_profiler = pufferlib.utils.Profiler()

        raw_env = binding.raw_env_creator()
        puf_env = binding.env_creator()

        with raw_profiler:
            raw_obs = raw_env.reset()

        with puf_profiler:
            puf_obs = puf_env.reset()

        raw_atns = {}
        puf_atns = {}

        raw_done = False
        puf_done = False

        for step in range(steps):
            if not pufferlib.utils.is_multiagent(raw_env):
                raw_atns = binding.raw_single_action_space.sample()
            else:
                for agent in raw_obs.keys():
                    raw_atns[agent] = binding.raw_single_action_space.sample()

            for agent in puf_obs.keys():
                puf_atns[agent] = binding.single_action_space.sample()

            with raw_profiler:
                if raw_done:
                    raw_obs = raw_env.reset()
                    raw_done = False
                else:
                    raw_obs, raw_rewards, raw_done, raw_infos = raw_env.step(raw_atns)
                
                    if pufferlib.utils.is_multiagent(raw_env):
                        raw_done = all(raw_done.values())

            with puf_profiler:
                if puf_env.done:
                    puf_obs = puf_env.reset()
                    puf_done = False
                else:
                    puf_obs, puf_rewards, puf_done, puf_infos = puf_env.step(puf_atns)
                    puf_done = puf_env.done

        print(
            f'{binding.env_name} - Performance Factor: {raw_profiler.elapsed / puf_profiler.elapsed:<4.3}\n' 
            f'Raw SPS: {steps // raw_profiler.elapsed}, Puffer SPS: {steps // puf_profiler.elapsed}')

        for name, timer in puf_env._timers.items():
            print(f'\t{name}: {timer.elapsed:.2f} in {timer.calls} calls')


if __name__ == '__main__':
    test_environments()
    #test_bindings()
    #test_bindings_performance()
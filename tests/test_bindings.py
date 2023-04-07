from pdb import set_trace as T

import pufferlib
import pufferlib.utils
import pufferlib.registry
import pufferlib.registry.atari
import pufferlib.registry.butterfly
import pufferlib.registry.classic_control
import pufferlib.registry.crafter
import pufferlib.registry.dm_lab
import pufferlib.registry.griddly
import pufferlib.registry.magent
import pufferlib.registry.microrts
import pufferlib.registry.nethack
import pufferlib.registry.nmmo
import pufferlib.registry.smac


def make_all_bindings():
    binding_creation_fns = [
        lambda: pufferlib.registry.atari.make_binding('BeamRider-v4', framestack=1),
        lambda: pufferlib.registry.atari.make_binding('Breakout-v4', framestack=1),
        lambda: pufferlib.registry.atari.make_binding('Enduro-v4', framestack=1),
        lambda: pufferlib.registry.atari.make_binding('Pong-v4', framestack=1),
        lambda: pufferlib.registry.atari.make_binding('Qbert-v4', framestack=1),
        lambda: pufferlib.registry.atari.make_binding('Seaquest-v4', framestack=1),
        lambda: pufferlib.registry.atari.make_binding('SpaceInvaders-v4', framestack=1),
        pufferlib.registry.butterfly.make_cooperative_pong_v5_binding,
        pufferlib.registry.butterfly.make_knights_archers_zombies_v10_binding,
        pufferlib.registry.classic_control.make_cartpole_binding,
        pufferlib.registry.crafter.make_binding,
        pufferlib.registry.dm_lab.make_binding,
        #pufferlib.registry.griddly.make_spider_v0_binding(),
        pufferlib.registry.magent.make_battle_v4_binding,
        pufferlib.registry.microrts.make_binding,
        pufferlib.registry.nethack.make_binding,
        pufferlib.registry.nmmo.make_binding,
        pufferlib.registry.smac.make_binding,
    ]

    names, bindings = set(), {}
    for fn in binding_creation_fns:
        try: 
            binding = fn()
        except pufferlib.utils.SetupError as e:
            print(e.message)
        else:
            name = binding.env_name
            assert name is not None
            assert name not in names, 'Duplicate env name'
            names.add(name)
            bindings[name] = binding

    return pufferlib.utils.dotdict(bindings)

def test_bindings():
    for binding in make_all_bindings().values():
        env = binding.env_creator()
        obs = env.reset()

        actions = {}
        for agent, ob in obs.items():
            assert env.observation_space(agent).contains(ob)
            actions[agent] = env.action_space(agent).sample()

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
            if binding._emulate_multiagent:
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
                
                    if not binding._emulate_multiagent:
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
            f'\tRaw SPS: {steps // raw_profiler.elapsed}, Puffer SPS: {steps // puf_profiler.elapsed}')

if __name__ == '__main__':
    test_bindings()
    test_bindings_performance()
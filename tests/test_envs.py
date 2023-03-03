from pdb import set_trace as T

import random
import numpy as np
import gym

import pufferlib
import pufferlib.registry
from pufferlib.vecenvs import VecEnvs


def test_singleagent_env(env_creator, seed=42, steps=32):
    with pufferlib.utils.Suppress():
        env_1 = env_creator()
        env_2 = env_creator()

    env_1.seed(seed)
    env_2.seed(seed)

    env_1.action_space.seed(seed)
    env_2.action_space.seed(seed)

    ob_1 = env_1.reset()
    ob_2 = env_2.reset()

    assert np.array_equal(ob_1, ob_2)
 
    done_1 = False
    done_2 = False

    for i in range(steps):
        atn_1 = env_1.action_space.sample()
        atn_2 = env_2.action_space.sample()

        if done_1:
            assert done_2

            ob_1 = env_1.reset()
            ob_2 = env_2.reset()

            done_1 = False
            done_2 = False

            assert np.array_equal(ob_1, ob_2)
        else:
            ob_1, reward_1, done_1, info_1 = env_1.step(atn_1)
            ob_2, reward_2, done_2, info_2 = env_2.step(atn_2)

            assert np.array_equal(ob_1, ob_2)
            assert reward_1 == reward_2
            assert done_1 == done_2
            assert info_1 == info_2


def test_multiagent_env(env_creator, seed=42, steps=10000):
    with pufferlib.utils.Suppress():
        env_1 = env_creator()
        env_2 = env_creator()

    env_1.seed(seed)
    env_2.seed(seed)

    for a in env_1.possible_agents:
        env_1.action_space(a).seed(seed)
        env_2.action_space(a).seed(seed)

    ob_1 = env_1.reset()
    ob_2 = env_2.reset()

    for agent in env_1.agents:
        assert np.array_equal(ob_1[agent], ob_2[agent])
 
    done_1 = False
    done_2 = False

    for i in range(steps):
        atn_1 = {a: env_1.action_space(a).sample() for a in env_1.possible_agents}
        atn_2 = {a: env_2.action_space(a).sample() for a in env_2.possible_agents}

        # TODO: check on dones for pufferlib style
        if done_1:
            assert env_1.done
            assert env_2.done
            assert done_2

            ob_1 = env_1.reset()
            ob_2 = env_2.reset()

            done_1 = False
            done_2 = False

            for agent in env_1.agents:
                assert np.array_equal(ob_1[agent], ob_2[agent])
        else:
            assert not env_1.done
            assert not env_2.done

            ob_1, reward_1, done_1, info_1 = env_1.step(atn_1)
            ob_2, reward_2, done_2, info_2 = env_2.step(atn_2)

            for agent in env_1.agents:
                assert np.array_equal(ob_1[agent], ob_2[agent])
                assert reward_1[agent] == reward_2[agent]
                assert done_1[agent] == done_2[agent]
                assert info_1[agent] == info_2[agent]

            done_1 = all(done_1.values())
            done_2 = all(done_2.values())


def test_vec_env(binding, seed=42, steps=10000):
    import ray
    ray.shutdown()
    ray.init(include_dashboard=False, ignore_reinit_error=True)

    env_1 = VecEnvs(binding, num_workers=4, envs_per_worker=1)
    env_2 = VecEnvs(binding, num_workers=2, envs_per_worker=2)

    env_1.seed(seed)
    env_2.seed(seed)

    ob_1 = env_1.reset()
    ob_2 = env_2.reset()

    assert np.array_equal(ob_1, ob_2)

    for i in range(steps):
        atn_2 = atn_1 = [binding.single_action_space.sample() for _ in range(4*binding.max_agents)]

        ob_1, reward_1, done_1, info_1 = env_1.step(atn_1)
        ob_2, reward_2, done_2, info_2 = env_2.step(atn_2)

        assert np.array_equal(ob_1, ob_2)
        assert reward_1 == reward_2
        assert done_1 == done_2
        #assert info_1 == info_2

# TODO: Conditional test infos

if __name__ == '__main__':
    import pufferlib.registry.atari
    #test_singleagent_env(lambda: gym.make('BreakoutNoFrameskip-v4'))
    #test_singleagent_env(lambda: pufferlib.registry.atari.env_creator('BreakoutNoFrameskip-v4', framestack=1))
    binding = pufferlib.registry.atari.create_binding('BreakoutNoFrameskip-v4', framestack=1)
    #test_singleagent_env(binding.raw_env_creator)
    #test_multiagent_env(binding.env_creator)
    test_vec_env(binding)

    #import pufferlib.registry.nmmo
    #binding = pufferlib.registry.nmmo.create_binding()
    #test_multiagent_env(binding.raw_env_creator)
    #test_multiagent_env(binding.env_creator)

    #import pufferlib.registry.nethack
    #binding = pufferlib.registry.nethack.create_binding()
    #test_singleagent_env(binding.raw_env_creator)
    #test_multiagent_env(binding.env_creator)

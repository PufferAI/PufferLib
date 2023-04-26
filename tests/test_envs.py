from pdb import set_trace as T

import numpy as np
import gym

import pufferlib
import pufferlib.emulation
import pufferlib.registry
import pufferlib.vectorization.serial


def flat_equal(ob_1, ob_2, flat_space):
    return np.array_equal(
        pufferlib.emulation._flatten_ob(ob_1, flat_space),
        pufferlib.emulation._flatten_ob(ob_2, flat_space)
    )
    

def test_singleagent_env(env_creator, action_space, seed=42, steps=1000):
    with pufferlib.utils.Suppress():
        env_1 = env_creator()
        env_2 = env_creator()

    ob_1 = env_1.reset(seed=seed)
    ob_2 = env_2.reset(seed=seed)

    assert flat_equal(ob_1, ob_2)
 
    done_1 = False
    done_2 = False

    for i in range(steps):
        atn_1 = action_space.sample()
        if type(atn_1) != int:
            atn_2 = atn_1.copy()
        else:
            atn_2 = atn_1

        if done_1:
            assert done_2

            ob_1 = env_1.reset()
            ob_2 = env_2.reset()

            done_1 = False
            done_2 = False

            assert flat_equal(ob_1, ob_2)
        else:
            ob_1, reward_1, done_1, info_1 = env_1.step(atn_1)
            ob_2, reward_2, done_2, info_2 = env_2.step(atn_2)

            assert flat_equal(ob_1, ob_2)
            assert reward_1 == reward_2
            assert done_1 == done_2


def test_multiagent_env(binding, action_space, seed=42, steps=1000):
    with pufferlib.utils.Suppress():
        env_1 = binding.env_creator()
        env_2 = binding.env_creator()

    ob_1 = env_1.reset(seed=seed)
    ob_2 = env_2.reset(seed=seed)

    for agent in env_1.agents:
        assert flat_equal(ob_1[agent], ob_2[agent], binding._featurized_single_observation_space)
 
    done_1 = False
    done_2 = False

    for i in range(steps):
        atn_1 = {a: action_space.sample() for a in env_1.possible_agents}
        atn_2 = atn_1.copy()

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
                assert flat_equal(ob_1[agent], ob_2[agent], binding._featurized_single_observation_space)
        else:
            assert not env_1.done
            assert not env_2.done

            ob_1, reward_1, done_1, _ = env_1.step(atn_1)
            ob_2, reward_2, done_2, _ = env_2.step(atn_2)

            for agent in env_1.agents:
                assert flat_equal(ob_1[agent], ob_2[agent], binding._featurized_single_observation_space)
                assert reward_1[agent] == reward_2[agent]
                assert done_1[agent] == done_2[agent]

            done_1 = all(done_1.values())
            done_2 = all(done_2.values())


def test_vec_env(binding, seed=42, steps=1000):
    import ray
    ray.shutdown()
    ray.init(include_dashboard=False, ignore_reinit_error=True)

    env_1 = pufferlib.vectorization.serial.VecEnv(binding, num_workers=4, envs_per_worker=1)
    env_2 = pufferlib.vectorization.serial.VecEnv(binding, num_workers=2, envs_per_worker=2)

    env_1.seed(seed)
    env_2.seed(seed)

    ob_1 = env_1.reset()
    ob_2 = env_2.reset()

    assert np.array_equal(ob_1, ob_2)

    for i in range(steps):
        atn_2 = atn_1 = [binding.single_action_space.sample() for _ in range(4*binding.max_agents)]

        ob_1, reward_1, done_1, _ = env_1.step(atn_1)
        ob_2, reward_2, done_2, _ = env_2.step(atn_2)

        assert np.array_equal(ob_1, ob_2)
        assert reward_1 == reward_2
        assert done_1 == done_2


if __name__ == '__main__':
    import mock_environments

    for env_cls in mock_environments.MOCK_ENVIRONMENTS:
        binding = pufferlib.emulation.Binding(env_cls=env_cls)
        test_multiagent_env(binding, binding.single_action_space)

    import pufferlib.registry.atari
    binding = pufferlib.registry.atari.make_binding('BreakoutNoFrameskip-v4', framestack=1)
    test_singleagent_env(lambda: gym.make('BreakoutNoFrameskip-v4'), binding.raw_single_action_space)
    test_singleagent_env(binding.raw_env_creator, binding.raw_single_action_space)
    test_multiagent_env(binding.env_creator, binding.single_action_space)
    test_vec_env(binding)

    # TODO: Examine Vectorization test
    import pufferlib.registry.nethack
    binding = pufferlib.registry.nethack.make_binding()
    test_singleagent_env(binding.raw_env_creator, binding.raw_single_action_space)
    test_multiagent_env(binding.env_creator, binding.single_action_space)
    #test_vec_env(binding)

    # TODO: Fix NMMO randomization and then redo this test
    #import pufferlib.registry.nmmo
    #binding = pufferlib.registry.nmmo.make_binding()
    #test_multiagent_env(binding.raw_env_creator, binding.raw_single_action_space)
    #test_multiagent_env(binding.env_creator, binding.single_action_space)
    #test_vec_env(binding)
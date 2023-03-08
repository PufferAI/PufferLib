from pdb import set_trace as T

import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.registry


def test_flatten():
    inputs = [
        {
            'foo': {
                'bar': 0
            },
            'baz': 1
        },
        0
    ]
    outputs = [
        {
            ('foo', 'bar'): 0,
            ('baz',): 1
        },
        0
    ]
    for inp, out in zip(inputs, outputs):
        test_out = pufferlib.emulation._flatten(inp)
        assert out == test_out, f'\n\tOutput: {test_out}\n\tExpected: {out}'


def test_unflatten():
    input = [1, 2, 3]

    structures = [
        {
            'foo': None,
            'bar': None,
            'baz': None,
        },
        {
            'foo': {
                'bar': None,
                'baz': None,
            },
            'qux': None,
        },
    ]

    outputs = [
        {
            'foo': 1,
            'bar': 2,
            'baz': 3,
        },
        {
            'foo': {
                'bar': 1,
                'baz': 2,
            },
            'qux': 3,
        }
    ]


    for struct, out in zip(structures, outputs):
        test_out = pufferlib.emulation._unflatten(input, struct)
        assert out == test_out, f'\n\tOutput: {test_out}\n\tExpected: {out}'


def test_pack_and_batch_obs(binding):
    env = binding.env_creator()
    env.seed(42)
    obs = env.reset()
    packed = pufferlib.emulation._pack_and_batch_obs(obs)
    assert type(packed) == np.ndarray
    assert len(packed) == len(obs)


def test_singleagent_emulation(binding, seed=42, steps=1000):
    with pufferlib.utils.Suppress():
        env_1 = binding.raw_env_creator()
        env_2 = binding.env_creator()

    binding.raw_single_action_space.seed(seed)

    env_1.seed(seed)
    env_2.seed(seed)

    ob_1 = env_1.reset()
    ob_2 = env_2.reset()

    flat = pufferlib.emulation._flatten_ob(ob_1)
    assert np.array_equal(flat, ob_2[1])
 
    done_1 = False
    done_2 = False

    for i in range(steps):
        atn_1 = binding.raw_single_action_space.sample()
        atn_2 = pufferlib.emulation._flatten(atn_1)

        if type(atn_2) == int:
            atn_2 = [atn_2]

        if done_1:
            assert done_2
            assert env_2.done

            ob_1 = env_1.reset()
            ob_2 = env_2.reset()[1]

            done_1 = False
            done_2 = False

            flat = pufferlib.emulation._flatten_ob(ob_1)
            assert np.array_equal(flat, ob_2)
        else:
            assert not env_2.done

            ob_1, reward_1, done_1, _ = env_1.step(atn_1)
            ob_2, reward_2, done_2, _ = env_2.step({1: atn_2})
            ob_2, reward_2, done_2 = ob_2[1], reward_2[1], done_2[1]

            flat = pufferlib.emulation._flatten_ob(ob_1)
            assert np.array_equal(flat, ob_2)
            assert reward_1 == reward_2
            assert done_1 == done_2


def test_multiagent_emulation(binding, seed=42, steps=1000):
    with pufferlib.utils.Suppress():
        env_1 = binding.raw_env_creator()
        env_2 = binding.env_creator()

    assert env_1.possible_agents == env_2.possible_agents
    binding.raw_single_action_space.seed(seed)

    env_1.seed(seed)
    env_2.seed(seed)

    ob_1 = env_1.reset()
    ob_2 = env_2.reset()

    for agent in env_1.agents:
        flat = pufferlib.emulation._flatten_ob(ob_1[agent])
        assert np.array_equal(flat, ob_2[agent])
 
    done_1 = False
    done_2 = False

    for i in range(steps):
        assert env_1.agents == env_2.agents

        atn_1 = {a: binding.raw_single_action_space.sample() for a in env_1.possible_agents}

        atn_2 = {}
        for k, v in atn_1.items():
            a = list(pufferlib.emulation._flatten(v).values())
            atn_2[k] = [a] if type(a) == int else a

        # TODO: check on dones for pufferlib style
        if done_1:
            assert env_2.done
            assert done_2

            ob_1 = env_1.reset()
            ob_2 = env_2.reset()

            done_1 = False
            done_2 = False

            for agent in env_1.agents:
                flat = pufferlib.emulation._flatten_ob(ob_1[agent])
                assert np.array_equal(flat, ob_2[agent])
        else:
            assert not env_2.done

            ob_1, reward_1, done_1, _ = env_1.step(atn_1)
            ob_2, reward_2, done_2, _ = env_2.step(atn_2)

            for agent in env_1.agents:
                flat = pufferlib.emulation._flatten_ob(ob_1[agent])
                assert np.array_equal(flat, ob_2[agent])
                assert reward_1[agent] == reward_2[agent]
                assert done_1[agent] == done_2[agent]

            done_1 = all(done_1.values())
            done_2 = all(done_2.values())


if __name__ == '__main__':
    import mock_environments
    binding = pufferlib.emulation.Binding(env_cls=mock_environments.TestEnv)
    test_multiagent_emulation(binding)

    # TODO: Fix this test
    import pufferlib.registry.nethack
    binding = pufferlib.registry.nethack.make_binding()
    test_singleagent_emulation(binding)

    # TODO: Fix this test
    import pufferlib.registry.atari
    binding = pufferlib.registry.atari.make_binding('BreakoutNoFrameskip-v4', framestack=1)
    test_singleagent_emulation(binding)

    test_flatten()
    test_unflatten()
    test_pack_and_batch_obs(binding)
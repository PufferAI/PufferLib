from pdb import set_trace as T

import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.registry


def test_emulation(binding, seed=42, steps=1000):
    with pufferlib.utils.Suppress():
        env_1 = binding.raw_env_creator()
        env_2 = binding.env_creator()

    assert env_1.possible_agents == env_2.possible_agents
    binding.raw_single_action_space.seed(seed)

    ob_1 = env_1.reset(seed=seed)
    ob_2 = env_2.reset(seed=seed)

    flat_atn_space = pufferlib.emulation._flatten_space(binding.single_action_space)

    for k, o in binding.pack_obs(ob_1).items():
        assert np.array_equal(o, ob_2[k])
 
    done_1 = False
    done_2 = False

    for i in range(steps):
        assert set(env_1.agents) == set(env_2.agents)

        atn_1 = {a: binding.single_action_space.sample() for a in env_1.possible_agents}

        # TODO: Rename flatten_ob
        # atn_2 = pufferlib.emulation._pack_obs(atn_1, flat_atn_space)
        atn_2 = {k: pufferlib.emulation._flatten_to_array(v, flat_atn_space) for k, v in atn_1.items()}

        # TODO: check on dones for pufferlib style
        if done_1:
            assert env_2.done
            assert done_2

            ob_1 = env_1.reset()
            ob_2 = env_2.reset()

            done_1 = False
            done_2 = False

            flat = binding.pack_obs(ob_1)
            for agent in env_1.agents:
                assert np.array_equal(flat[agent], ob_2[agent])
        else:
            assert not env_2.done

            ob_1, reward_1, done_1, _ = env_1.step(atn_1)
            ob_2, reward_2, done_2, _ = env_2.step(atn_2)

            flat = binding.pack_obs(ob_1)
            for agent in env_1.agents:
                assert np.array_equal(flat[agent], ob_2[agent])
                assert reward_1[agent] == reward_2[agent]
                assert done_1[agent] == done_2[agent]

            done_1 = all(done_1.values())
            done_2 = all(done_2.values())


if __name__ == '__main__':
    import mock_environments
    for env in mock_environments.MOCK_ENVIRONMENTS:
        binding = pufferlib.emulation.Binding(env_cls=env)
        test_emulation(binding)
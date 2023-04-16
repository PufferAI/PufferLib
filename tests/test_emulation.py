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

    for agent in env_1.agents:
        flat = pufferlib.emulation._flatten_ob(ob_1[agent])
        assert np.array_equal(flat, ob_2[agent])
 
    done_1 = False
    done_2 = False

    for i in range(steps):
        assert set(env_1.agents) == set(env_2.agents)

        atn_1 = {a: binding.single_action_space.sample() for a in env_1.possible_agents}
        atn_2 = {k[0]: v for k, v in pufferlib.emulation._flatten(atn_1).items()}

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
    for env in mock_environments.MOCK_ENVIRONMENTS:
        binding = pufferlib.emulation.Binding(env_cls=env)
        test_emulation(binding)
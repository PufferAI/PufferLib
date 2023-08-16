from pdb import set_trace as T

import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.registry
import pufferlib.vectorization


def test_gym_emulation(env_cls, steps=100):
    raw_env = env_cls()
    puf_env = pufferlib.emulation.GymPufferEnv(env_creator=env_cls)

    raw_done = True
    puf_done = True

    flat_obs_space = puf_env.flat_observation_space

    for i in range(steps):
        assert puf_done == raw_done

        if raw_done:
            puf_ob = puf_env.reset()
            raw_ob = raw_env.reset()

        # Reconstruct original obs format from puffer env and compare to raw
        puf_ob = pufferlib.emulation.unflatten(
            pufferlib.emulation.split(
                puf_ob, flat_obs_space, batched=False
            )
        )
 
        assert pufferlib.utils._compare_space_samples(raw_ob, puf_ob)

        action = raw_env.action_space.sample()
        raw_ob, raw_reward, raw_done, _ = raw_env.step(action)

        # Convert raw actions to puffer format
        action = pufferlib.emulation.concatenate(pufferlib.emulation.flatten(action))
        action = [action] if type(action) == int else action
        action = np.array(action)

        puf_ob, puf_reward, puf_done, _ = puf_env.step(action)
        assert puf_reward == raw_reward

def test_pettingzoo_emulation(env_cls, steps=100):
    raw_env = env_cls()
    puf_env = pufferlib.emulation.PettingZooPufferEnv(env_creator=env_cls)

    flat_obs_space = puf_env.flat_observation_space

    for i in range(steps):
        raw_done = len(raw_env.agents) == 0
        puf_done = len(puf_env.agents) == 0

        assert puf_done == raw_done

        if raw_done:
            puf_obs = puf_env.reset()
            raw_obs = raw_env.reset()

        # Reconstruct original obs format from puffer env and compare to raw
        for agent in puf_env.possible_agents:
            if agent not in raw_obs:
                assert sum(puf_obs[agent] != 0) == 0
                continue
            
            raw_ob = raw_obs[agent]
            puf_ob = pufferlib.emulation.unflatten(
                pufferlib.emulation.split(
                    puf_obs[agent], flat_obs_space, batched=False
                )
            )

            assert pufferlib.utils._compare_space_samples(raw_ob, puf_ob)

        raw_actions = {a: raw_env.action_space(a).sample()
            for a in raw_env.agents}
        raw_obs, raw_rewards, raw_dones, _ = raw_env.step(raw_actions)

        # Convert raw actions to puffer format
        puf_actions = {}
        dummy_action = raw_env.action_space(0).sample()
        for agent in puf_env.possible_agents:
            if agent in raw_env.agents:
                action = raw_actions[agent]
            else:
                action = dummy_action

            action = pufferlib.emulation.concatenate(pufferlib.emulation.flatten(action))
            action = [action] if type(action) == int else action
            action = np.array(action)
            puf_actions[agent] = action

        puf_obs, puf_rewards, puf_dones, _ = puf_env.step(puf_actions)

        for agent in raw_rewards:
            assert puf_rewards[agent] == raw_rewards[agent]

        for agent in raw_dones:
            assert puf_dones[agent] == raw_dones[agent]


if __name__ == '__main__':
    import mock_environments
    for env_cls in mock_environments.MOCK_SINGLE_AGENT_ENVIRONMENTS:
        test_gym_emulation(env_cls)

    for env_cls in mock_environments.MOCK_MULTI_AGENT_ENVIRONMENTS:
        test_pettingzoo_emulation(env_cls)
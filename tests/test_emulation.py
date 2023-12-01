from pdb import set_trace as T

import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils
import pufferlib.vectorization


def test_gym_emulation(env_cls, observation_space, action_space, steps=100):
    raw_profiler = pufferlib.utils.Profiler()
    puf_profiler = pufferlib.utils.Profiler()

    # Do not profile env creation
    raw_env = env_cls(observation_space, action_space)
    puf_env = pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_cls,
        env_args=(observation_space, action_space))

    raw_done = puf_done = True
    raw_truncated = puf_truncated = False

    flat_obs_space = puf_env.flat_observation_space

    for i in range(steps):
        assert puf_done == raw_done
        assert puf_truncated == raw_truncated

        if raw_done:
            with puf_profiler:
                puf_ob, _ = puf_env.reset()
            with raw_profiler:
                raw_ob, _ = raw_env.reset()
                structure = pufferlib.emulation.flatten_structure(raw_ob)

        # Reconstruct original obs format from puffer env and compare to raw
        puf_ob = pufferlib.emulation.unflatten(
            pufferlib.emulation.split(
                puf_ob, flat_obs_space, batched=False
            ), structure
        )
 
        pufferlib.utils.compare_space_samples(raw_ob, puf_ob)

        action = raw_env.action_space.sample()

        with raw_profiler:
            raw_ob, raw_reward, raw_done, raw_truncated, _ = raw_env.step(action)

        # Convert raw actions to puffer format

        if not isinstance(action, int):
            action = pufferlib.emulation.concatenate(pufferlib.emulation.flatten(action))
            action = [action] if type(action) == int else action
            action = np.array(action)

        with puf_profiler:
            puf_ob, puf_reward, puf_done, puf_truncated, _ = puf_env.step(action)

        assert puf_reward == raw_reward

    return raw_profiler.elapsed/steps, puf_profiler.elapsed/steps

def test_pettingzoo_emulation(env_cls, observation_space, action_space, steps=100):
    raw_profiler = pufferlib.utils.Profiler()
    puf_profiler = pufferlib.utils.Profiler()

    # Do not profile env creation
    raw_env = env_cls(observation_space, action_space)
    puf_env = pufferlib.emulation.PettingZooPufferEnv(env_creator=env_cls,
        env_args=(observation_space, action_space))

    flat_obs_space = puf_env.flat_observation_space

    for i in range(steps):
        raw_done = len(raw_env.agents) == 0
        puf_done = len(puf_env.agents) == 0

        assert puf_done == raw_done

        if raw_done:
            with puf_profiler:
                puf_obs, _ = puf_env.reset()
            with raw_profiler:
                raw_obs, _ = raw_env.reset()

        # Reconstruct original obs format from puffer env and compare to raw
        for agent in puf_env.possible_agents:
            if agent not in raw_obs:
                assert np.sum(puf_obs[agent] != 0) == 0
                continue
            
            raw_ob = raw_obs[agent]
            puf_ob = pufferlib.emulation.unflatten(
                pufferlib.emulation.split(
                    puf_obs[agent], flat_obs_space, batched=False
                ), puf_env.flat_observation_structure
            )

            assert pufferlib.utils.compare_space_samples(raw_ob, puf_ob)

        raw_actions = {a: raw_env.action_space(a).sample()
            for a in raw_env.agents}

        with raw_profiler:
            raw_obs, raw_rewards, raw_dones, raw_truncateds, _ = raw_env.step(raw_actions)

        # Convert raw actions to puffer format
        puf_actions = {}
        dummy_action = raw_env.action_space(0).sample()
        for agent in puf_env.possible_agents:
            if agent in raw_env.agents:
                action = raw_actions[agent]
            else:
                action = dummy_action

            if not isinstance(action, int):
                action = pufferlib.emulation.concatenate(pufferlib.emulation.flatten(action))
                action = [action] if type(action) == int else action
                action = np.array(action)

            puf_actions[agent] = action

        with puf_profiler:
            puf_obs, puf_rewards, puf_dones, puf_truncateds, _ = puf_env.step(puf_actions)

        for agent in raw_rewards:
            assert puf_rewards[agent] == raw_rewards[agent]

        for agent in raw_dones:
            assert puf_dones[agent] == raw_dones[agent]

    return raw_profiler.elapsed/steps, puf_profiler.elapsed/steps


if __name__ == '__main__':
    from pufferlib.environments import test

    raw_gym, puf_gym= [], []
    for obs_space in test.MOCK_OBSERVATION_SPACES:
        for atn_space in test.MOCK_ACTION_SPACES:
            raw_t, puf_t = test_gym_emulation(
                test.GymnasiumTestEnv, obs_space, atn_space)
            raw_gym.append(raw_t)
            puf_gym.append(puf_t)

    gym_overhead = (np.array(puf_gym) - np.array(raw_gym))*1000
    gym_performance = (
        'Gym Emulation Overhead (ms)\n'
            f'\t Min: {min(gym_overhead):.2f}\n'
            f'\t Max: {max(gym_overhead):.2f}\n'
            f'\t Mean: {np.mean(gym_overhead):.2f}\n'
    )
    print(gym_performance)

    raw_pz, puf_pz = [], []
    for obs_space in test.MOCK_OBSERVATION_SPACES:
        for atn_space in test.MOCK_ACTION_SPACES:
            raw_t, puf_t = test_pettingzoo_emulation(
                test.PettingZooTestEnv, obs_space, atn_space)
            raw_pz.append(raw_t)
            puf_pz.append(puf_t)

    pz_overhead = (np.array(puf_pz) - np.array(raw_pz))*1000
    pz_performance = (
        'PettingZoo Emulation Overhead (ms)\n'
            f'\t Min: {min(pz_overhead):.2f}\n'
            f'\t Max: {max(pz_overhead):.2f}\n'
            f'\t Mean: {np.mean(pz_overhead):.2f}\n'
    )
    print(pz_performance)

    performance = '\n'.join([gym_performance, pz_performance])
    with open ('performance.txt', 'w') as f:
        f.write(performance)

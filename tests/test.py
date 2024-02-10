from pdb import set_trace as T

import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.utils
import pufferlib.vectorization
from pufferlib.environments import test

# Deprecation warnings from gymnasium
import gymnasium
import warnings
warnings.filterwarnings("ignore")


def gymnasium_emulation(env_cls, steps=100):
    raw_env = env_cls()
    puf_env = pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_cls)

    raw_done = puf_done = True
    raw_truncated = puf_truncated = False

    flat_obs_space = puf_env.flat_observation_space

    sz = [int(np.prod(subspace.shape))
        for subspace in puf_env.flat_observation_space.values()]

    for i in range(steps):
        assert puf_done == raw_done
        assert puf_truncated == raw_truncated

        if raw_done:
            puf_ob, _ = puf_env.reset()
            raw_ob, _ = raw_env.reset()
            structure = pufferlib.emulation.flatten_structure(raw_ob)

        # Reconstruct original obs format from puffer env and compare to raw
        puf_ob = pufferlib.emulation.unflatten(
            pufferlib.emulation.split(
                puf_ob, flat_obs_space, sz, batched=False
            ),
            structure
        )
        pufferlib.utils.compare_space_samples(raw_ob, puf_ob)

        action = raw_env.action_space.sample()
        raw_ob, raw_reward, raw_done, raw_truncated, _ = raw_env.step(action)

        # Convert raw actions to puffer format
        if not isinstance(action, int):
            action = pufferlib.emulation.concatenate(pufferlib.emulation.flatten(action))
            if len(action) == 1:
                action = action[0]

        puf_ob, puf_reward, puf_done, puf_truncated, _ = puf_env.step(action)
        assert puf_reward == raw_reward

def pettingzoo_emulation(env_cls, steps=100):
    raw_env = env_cls()
    puf_env = pufferlib.emulation.PettingZooPufferEnv(env_creator=env_cls)

    flat_obs_space = puf_env.flat_observation_space

    sz = [int(np.prod(subspace.shape))
        for subspace in puf_env.flat_observation_space.values()]

    for i in range(steps):
        raw_done = len(raw_env.agents) == 0
        puf_done = len(puf_env.agents) == 0

        assert puf_done == raw_done

        if raw_done:
            puf_obs, _ = puf_env.reset()
            raw_obs, _ = raw_env.reset()

        # Reconstruct original obs format from puffer env and compare to raw
        for agent in puf_env.possible_agents:
            if agent not in raw_obs:
                assert np.sum(puf_obs[agent] != 0) == 0
                continue
            
            raw_ob = raw_obs[agent]
            puf_ob = pufferlib.emulation.unflatten(
                pufferlib.emulation.split(
                    puf_obs[agent], flat_obs_space, sz, batched=False
                ), puf_env.flat_observation_structure
            )

            assert pufferlib.utils.compare_space_samples(raw_ob, puf_ob)

        raw_actions = {a: raw_env.action_space(a).sample()
            for a in raw_env.agents}

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
                if len(action) == 1:
                    action = action[0]

            puf_actions[agent] = action

        puf_obs, puf_rewards, puf_dones, puf_truncateds, _ = puf_env.step(puf_actions)

        for agent in raw_rewards:
            assert puf_rewards[agent] == raw_rewards[agent]

        for agent in raw_dones:
            assert puf_dones[agent] == raw_dones[agent]

def gymnasium_vectorization(env_cls, vectorization,
        num_envs=1, envs_per_worker=1, steps=100):
    raw_envs = [env_cls() for _ in range(num_envs)]
    vec_envs = vectorization(
        env_creator=pufferlib.emulation.GymnasiumPufferEnv,
        env_kwargs={'env_creator': env_cls},
        num_envs=num_envs,
        envs_per_worker=envs_per_worker,
    )

    # agents per env
    raw_dones = [False for _ in range(num_envs)]

    raw_obs = [raw_env.reset()[0] for raw_env in raw_envs]
    vec_obs, _, _, _ = vec_envs.reset()

    for _ in range(steps):
        raw_obs = np.stack([
            pufferlib.emulation.concatenate(
                pufferlib.emulation.flatten(r_ob)
            ) for r_ob in raw_obs
        ], axis=0)

        assert raw_obs.shape == vec_obs.shape
        assert np.all(raw_obs == vec_obs)

        raw_actions = [r_env.action_space.sample() for r_env in raw_envs]

        # Copy reset behavior of VecEnv
        raw_obs, raw_rewards, nxt_dones = [], [], []
        for idx, r_env in enumerate(raw_envs):
            if raw_dones[idx]:
                raw_obs.append(r_env.reset()[0])
                raw_rewards.append(0)
                nxt_dones.append(False)
            else:
                r_ob, r_rew, r_done, _, _ = r_env.step(raw_actions[idx])
                raw_obs.append(r_ob)
                raw_rewards.append(r_rew)
                nxt_dones.append(r_done)
        raw_dones = nxt_dones
                
        # Convert raw actions to puffer format
        vec_actions = []
        for idx, r_a in enumerate(raw_actions):
            if not isinstance(r_a, int):
                r_a = pufferlib.emulation.concatenate(pufferlib.emulation.flatten(r_a))
                r_a = [r_a] if type(r_a) == int else r_a
                r_a = np.array(r_a)
            vec_actions.append(r_a)

        vec_obs, vec_rewards, vec_dones, _, _, _, _ = vec_envs.step(vec_actions)

        for idx in range(num_envs):
            assert raw_rewards[idx] == vec_rewards[idx]
            assert raw_dones[idx] == vec_dones[idx]

    vec_envs.close()
    for raw_env in raw_envs:
        raw_env.close()

def pettingzoo_vectorization(env_cls, vectorization,
        steps=100, num_envs=1, envs_per_worker=1):
    raw_envs = [env_cls() for _ in range(num_envs)]
    vec_envs = vectorization(
        env_creator=pufferlib.emulation.PettingZooPufferEnv,
        env_kwargs={'env_creator': env_cls},
        num_envs=num_envs,
        envs_per_worker=envs_per_worker,
    )

    possible_agents = raw_envs[0].possible_agents
    raw_terminated = [False for _ in range(num_envs)]

    raw_obs = [raw_env.reset()[0] for raw_env in raw_envs]
    vec_obs, _, _, _ = vec_envs.reset()

    for _ in range(steps):
        idx = 0
        for r_obs in raw_obs:
            for agent in possible_agents:
                if agent in raw_obs:
                    assert np.all(raw_obs[agent] == vec_obs[idx])
                idx += 1

        raw_actions = [
            {agent: r_env.action_space(agent).sample() for agent in possible_agents}
            for r_env in raw_envs
        ]

        # Copy reset behavior of VecEnv
        raw_obs, raw_rewards, nxt_dones = [], [], []
        for idx, r_env in enumerate(raw_envs):
            if raw_terminated[idx]:
                raw_obs.append(r_env.reset()[0])
                raw_rewards.append({agent: 0 for agent in possible_agents})
                nxt_dones.append({agent: False for agent in possible_agents})
            else:
                r_ob, r_rew, r_done, _, _ = r_env.step(raw_actions[idx])
                raw_obs.append(r_ob)
                raw_rewards.append(r_rew)
                nxt_dones.append(r_done)
            raw_terminated[idx] = len(r_env.agents) == 0
        raw_dones = nxt_dones
                
        # Convert raw actions to puffer format
        vec_actions = []
        dummy_action = raw_envs[0].action_space('agent_1').sample()
        for r_atns in raw_actions:
            for agent in possible_agents:
                if agent in r_atns:
                    action = r_atns[agent]
                else:
                    action = dummy_action

                if not isinstance(action, int):
                    action = pufferlib.emulation.concatenate(pufferlib.emulation.flatten(action))
                    action = [action] if type(action) == int else action
                    action = np.array(action)
                vec_actions.append(action)
        vec_actions = np.array(vec_actions)

        vec_obs, vec_rewards, vec_dones, _, _, _, _ = vec_envs.step(vec_actions)

        idx = 0
        for r_rewards, r_dones in zip(raw_rewards, raw_dones):
            for agent in possible_agents:
                if agent in r_rewards:
                    assert abs(vec_rewards[idx] - r_rewards[agent]) < 1e-5
                    if vec_dones[idx] != r_dones[agent]:
                        print(vec_dones[idx], r_dones[agent])
                    assert vec_dones[idx] == r_dones[agent]
                else:
                    assert vec_rewards[idx] == 0
                    # No assert for dones, depends on emulation
                
                idx += 1

    vec_envs.close()
    for raw_env in raw_envs:
        raw_env.close()

def test_emulation():
    for env_cls in test.MOCK_SINGLE_AGENT_ENVIRONMENTS:
        gymnasium_emulation(env_cls)

    print('Gymnasium emulation tests passed')

    for env_cls in test.MOCK_MULTI_AGENT_ENVIRONMENTS:
        pettingzoo_emulation(env_cls)

    print('PettingZoo emulation tests passed')

def test_vectorization():
    for vectorization in [
            pufferlib.vectorization.Serial,
            pufferlib.vectorization.Multiprocessing,
            pufferlib.vectorization.Ray]:
        for env_cls in test.MOCK_SINGLE_AGENT_ENVIRONMENTS:
            gymnasium_vectorization(
                env_cls,
                vectorization=vectorization,
                num_envs=6,
                envs_per_worker=2
            )

        print(f'Gymnasium {vectorization.__name__} vectorization tests passed')

        for env_cls in test.MOCK_MULTI_AGENT_ENVIRONMENTS:
            pettingzoo_vectorization(
                env_cls,
                vectorization=vectorization,
                num_envs=6,
                envs_per_worker=2
            )

        print(f'PettingZoo {vectorization.__name__} vectorization tests passed')

if __name__ == '__main__':
    test_emulation()
    test_vectorization()
    exit(0) # For Ray

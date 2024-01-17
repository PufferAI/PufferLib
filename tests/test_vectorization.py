# TODO: Figure out what I want to do with this for 0.5
# This is a stupidly complicated test to maintain. Do we
# lose anything by testing emulation, vectorization, and performance
# fully separately?
from pdb import set_trace as T

import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.vectorization


def test_gym_vectorization(env_cls, vectorization, steps=100, num_workers=1, envs_per_worker=1):
    raw_profiler = pufferlib.utils.Profiler()
    puf_profiler = pufferlib.utils.Profiler()

    # Do not profile env creation or first reset
    raw_envs = [env_cls() for _ in range(num_workers * envs_per_worker)]
    puf_envs = vectorization(
        env_creator=pufferlib.emulation.GymnasiumPufferEnv,
        env_kwargs={'env_creator': env_cls},
        num_workers=num_workers,
        envs_per_worker=envs_per_worker,
    )

    raw_dones = [False for _ in range(num_workers * envs_per_worker)]

    raw_obs = [raw_env.reset() for raw_env in raw_envs]
    puf_obs = puf_envs.reset()

    for _ in range(steps):
        puf_obs = puf_envs.unpack_batched_obs(puf_obs)
 
        for idx, r_ob in enumerate(raw_obs):
            assert pufferlib.utils.compare_space_samples(r_ob, puf_obs, idx)

        raw_actions = [r_env.action_space.sample() for r_env in raw_envs]

        # Copy reset behavior of VecEnv
        raw_obs, raw_rewards, nxt_dones = [], [], []
        for idx, r_env in enumerate(raw_envs):
            if raw_dones[idx]:
                with raw_profiler:
                    raw_obs.append(r_env.reset())
                raw_rewards.append(0)
                nxt_dones.append(False)
            else:
                with raw_profiler:
                    r_ob, r_rew, r_done, _, _ = r_env.step(raw_actions[idx])
                raw_obs.append(r_ob)
                raw_rewards.append(r_rew)
                nxt_dones.append(r_done)
        raw_dones = nxt_dones
                
        # Convert raw actions to puffer format
        puf_actions = []
        for idx, r_a in enumerate(raw_actions):
            if not isinstance(r_a, int):
                r_a = pufferlib.emulation.concatenate(pufferlib.emulation.flatten(r_a))
                r_a = [r_a] if type(r_a) == int else r_a
                r_a = np.array(r_a)
            puf_actions.append(r_a)

        with puf_profiler:
            puf_obs, puf_rewards, puf_dones, _, _ = puf_envs.step(puf_actions)

        for idx in range(num_workers * envs_per_worker):
            assert raw_rewards[idx] == puf_rewards[idx]
            assert raw_dones[idx] == puf_dones[idx]

    puf_envs.close()
    return raw_profiler.elapsed/steps/num_workers, puf_profiler.elapsed/steps


def test_pettingzoo_vectorization(env_cls, vectorization, steps=100, num_workers=1, envs_per_worker=1):
    raw_profiler = pufferlib.utils.Profiler()
    puf_profiler = pufferlib.utils.Profiler()

    # Do not profile env creation or first reset
    raw_envs = [env_cls() for _ in range(num_workers * envs_per_worker)]
    puf_envs = vectorization(
        env_creator=pufferlib.emulation.PettingZooPufferEnv,
        env_kwargs={'env_creator': env_cls},
        num_workers=num_workers,
        envs_per_worker=envs_per_worker,
    )

    possible_agents = raw_envs[0].possible_agents
    raw_terminated = [False for _ in range(num_workers * envs_per_worker)]

    raw_obs = [raw_env.reset() for raw_env in raw_envs]
    flat_puf_obs = puf_envs.reset()

    for _ in range(steps):
        puf_obs = puf_envs.unpack_batched_obs(flat_puf_obs)
 
        idx = 0
        for r_obs in raw_obs:
            for agent in possible_agents:
                if agent in raw_obs:
                    pufferlib.utils.compare_space_samples(r_obs[agent], puf_obs, idx)
                # Currently, vectorization does not reset padding to 0
                # This is for efficiency... need to do some timing
                #else:
                #    assert np.sum(flat_puf_obs[idx] != 0) == 0
                idx += 1

        raw_actions = [
            {agent: r_env.action_space(agent).sample() for agent in possible_agents}
            for r_env in raw_envs
        ]

        # Copy reset behavior of VecEnv
        raw_obs, raw_rewards, nxt_dones = [], [], []
        for idx, r_env in enumerate(raw_envs):
            if raw_terminated[idx]:
                with raw_profiler:
                    raw_obs.append(r_env.reset())
                raw_rewards.append({agent: 0 for agent in possible_agents})
                nxt_dones.append({agent: False for agent in possible_agents})
            else:
                with raw_profiler:
                    r_ob, r_rew, r_done, _, _ = r_env.step(raw_actions[idx])
                raw_obs.append(r_ob)
                raw_rewards.append(r_rew)
                nxt_dones.append(r_done)
            raw_terminated[idx] = len(r_env.agents) == 0
        raw_dones = nxt_dones
                
        # Convert raw actions to puffer format
        puf_actions = []
        dummy_action = raw_envs[0].action_space(0).sample()
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
                puf_actions.append(action)
        puf_actions = np.array(puf_actions)

        with puf_profiler:
            flat_puf_obs, puf_rewards, puf_dones, _, _ = puf_envs.step(puf_actions)

        idx = 0
        for r_rewards, r_dones in zip(raw_rewards, raw_dones):
            for agent in possible_agents:
                if agent in r_rewards:
                    assert puf_rewards[idx] == r_rewards[agent]
                    assert puf_dones[idx] == r_dones[agent]
                else:
                    assert puf_rewards[idx] == 0
                    # No assert for dones, depends on emulation
                
                idx += 1

    puf_envs.close()
    return raw_profiler.elapsed/steps/num_workers, puf_profiler.elapsed/steps


if __name__ == '__main__':
    from pufferlib.environments import test
    import numpy as np

    performance = []
    headers = "\t\t| Cores | Envs/Core |   Min   |   Max   |  Mean  "
    vectorizations = [pufferlib.vectorization.Serial]#, pufferlib.vectorization.Multiprocessing, pufferlib.vectorization.Ray]
    num_workers_list = [1, 2, 4]
    envs_per_worker_list = [1, 2, 4]

    # Gym Results
    title = 'Gym Vectorization Overhead (ms)'
    performance.append(title)
    print(title)
    for vectorization in vectorizations:
        vec_name = f'\t{vectorization.__name__}'
        performance.append(vec_name)
        performance.append(headers)
        print(vec_name)
        print(headers)
        for num_workers in num_workers_list:
            for envs_per_worker in envs_per_worker_list:
                raw_gym = []
                for env_cls in test.MOCK_SINGLE_AGENT_ENVIRONMENTS:
                    raw_t, puf_t = test_gym_vectorization(
                        env_cls, vectorization,
                        num_workers=num_workers,
                        envs_per_worker=envs_per_worker
                    )
                    raw_gym.append((np.array(puf_t) - np.array(raw_t)) * 1000)

                result = f"\t\t| {num_workers:^5} | {envs_per_worker:^9} | {min(raw_gym):^7.2f} | {max(raw_gym):^7.2f} | {np.mean(raw_gym):^7.2f}"
                performance.append(result)
                print(result)

    performance.append('\n')
    print()

    # PettingZoo Results
    title = 'PettingZoo Vectorization Overhead (ms)'
    performance.append(title)
    print(title)
    for vectorization in vectorizations:
        vec_name = f'\t{vectorization.__name__}'
        performance.append(vec_name)
        performance.append(headers)
        print(vec_name)
        print(headers)
        for num_workers in num_workers_list:
            for envs_per_worker in envs_per_worker_list:
                raw_pz = []
                for env_cls in test.MOCK_MULTI_AGENT_ENVIRONMENTS:
                    raw_t, puf_t = test_pettingzoo_vectorization(
                        env_cls, vectorization,
                        num_workers=num_workers,
                        envs_per_worker=envs_per_worker
                    )
                    raw_pz.append((np.array(puf_t) - np.array(raw_t)) * 1000)

                result = f"\t\t| {num_workers:^5} | {envs_per_worker:^9} | {min(raw_pz):^7.2f} | {max(raw_pz):^7.2f} | {np.mean(raw_pz):^7.2f}"
                performance.append(result)
                print(result)

    with open ('performance.txt', 'a') as f:
        f.write('\n'.join(performance))

    # Otherwise ray will hang
    exit(0)

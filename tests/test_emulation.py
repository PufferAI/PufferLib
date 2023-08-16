from pdb import set_trace as T

import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.registry
import pufferlib.vectorization


def make_env(wrapper, env_cls):
    return wrapper(env_creator=env_cls)

class MockGymVecEnv:
    def __init__(self, env_creator, num_workers, envs_per_worker=1):
        self.num_envs = num_workers * envs_per_worker
        self.envs = [env_creator()
            for _ in range(self.num_envs)]
        self.dones = [False for _ in range(self.num_envs)]

    def reset(self, seed=42):
        return [e.reset() for e in self.envs]

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for idx, (env, atn) in enumerate(zip(self.envs, actions)):
            if self.dones[idx]:
                o = env.reset()
                r = 0
                d = False
                i = {}
            else:
                o, r, d, i = env.step(atn)

            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(i)

        self.dones = dones
        return obs, rewards, dones, infos


def test_gym_emulation(env_cls, steps=100, num_workers=1, envs_per_worker=2):
    # Create raw environments and vectorized puffer environments
    raw_env = MockGymVecEnv(env_cls, num_workers, envs_per_worker)
    puf_env = pufferlib.vectorization.Serial(
        env_creator=make_env,
        env_args=[pufferlib.emulation.GymPufferEnv, env_cls],
        num_workers=num_workers,
        envs_per_worker=envs_per_worker,
    )

    puf_ob = puf_env.reset()
    raw_ob = raw_env.reset()

    obs_space = raw_env.envs[0].observation_space
    flat_obs_space = puf_env.envs[0].flat_observation_space
    flat_atn_space = puf_env.envs[0].envs[0].flat_action_space

    total_agents = num_workers * envs_per_worker

    for i in range(steps):
        # Reconstruct original obs format from puffer env and compare to raw
        orig_puf_ob = puf_ob
        puf_ob = pufferlib.emulation.unpack_batched_obs(puf_ob, obs_space, flat_obs_space)
        idx = 0
        for r_ob in raw_ob:
            assert pufferlib.utils._compare_space_samples(
                r_ob, puf_ob, idx)

        atn = [raw_env.envs[0].action_space.sample() for _ in range(total_agents)]
        raw_ob, raw_reward, raw_done, _ = raw_env.step(atn)

        # Convert raw actions to puffer format
        actions = [
            pufferlib.emulation.concatenate(
                pufferlib.emulation.flatten(a)
            ) for a in atn
        ]

        puf_ob, puf_reward, puf_done, _ = puf_env.step(actions)

        idx = 0 
        for r_reward, r_done in zip(raw_reward, raw_done):
            assert puf_reward[idx] == r_reward
            assert puf_done[idx] == r_done
            idx += 1


class MockPettingZooVecEnv:
    def __init__(self, env_creator, num_workers, envs_per_worker=1):
        self.num_envs = num_workers * envs_per_worker
        self.envs = [env_creator()
            for _ in range(self.num_envs)]
        self.possible_agents = self.envs[0].possible_agents

    def reset(self, seed=42):
        return [e.reset() for e in self.envs]

    @property
    def agents(self):
        return [e.agents for e in self.envs]

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for env, atns in zip(self.envs, actions):
            if not env.agents:
                o = env.reset()
                r = {k: 0 for k in env.possible_agents}
                d = {k: False for k in env.possible_agents}
                i = {}
            else:
                atns = {k: v for k, v in zip(env.agents, atns)}
                o, r, d, i = env.step(atns)

            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(i)

        return obs, rewards, dones, infos


def test_pettingzoo_emulation(env_cls, steps=100, num_workers=1, envs_per_worker=2):
    # Create raw environments and vectorized puffer environments
    raw_env = MockPettingZooVecEnv(env_cls, num_workers, envs_per_worker)
    puf_env = pufferlib.vectorization.Serial(
        env_creator=make_env,
        env_args=[pufferlib.emulation.PettingZooPufferEnv, env_cls],
        num_workers=num_workers,
        envs_per_worker=envs_per_worker,
    )

    puf_ob = puf_env.reset()
    raw_ob = raw_env.reset()

    obs_space = raw_env.envs[0].observation_space(0)
    flat_obs_space = puf_env.envs[0].flat_observation_space
    flat_atn_space = puf_env.envs[0].envs[0].flat_action_space

    for i in range(steps):
        # Reconstruct original obs format from puffer env and compare to raw
        orig_puf_ob = puf_ob
        puf_ob = pufferlib.emulation.unpack_batched_obs(puf_ob, obs_space, flat_obs_space)
        idx = 0
        for r_ob in raw_ob:
            for agent in raw_env.possible_agents:
                if agent not in r_ob:
                    idx += 1
                    continue
                else:
                    assert pufferlib.utils._compare_space_samples(
                        r_ob[agent], puf_ob, idx)
                idx += 1

        atn = [{a: raw_env.envs[0].action_space(a).sample() for a in agents}
               for agents in raw_env.agents]
        raw_ob, raw_reward, raw_done, _ = raw_env.step(atn)

        # Convert actions to puffer format
        actions = []
        dummy = raw_env.envs[0].action_space(0).sample()
        dummy = pufferlib.emulation.concatenate(pufferlib.emulation.flatten(dummy))
        for a in atn:
            for agent in raw_env.possible_agents:
                if agent in a:
                    actions.append(
                        pufferlib.emulation.concatenate(
                            pufferlib.emulation.flatten(a[agent])
                        )
                   )
                else:
                    actions.append(dummy)

        # TODO: Add shape asserts to vec envs
        puf_ob, puf_reward, puf_done, _ = puf_env.step(actions)

        idx = 0 
        for r_rewards, r_dones in zip(raw_reward, raw_done):
            for a in raw_env.possible_agents:
                if a not in r_rewards:
                    assert puf_reward[idx] == 0
                else:
                    assert puf_reward[idx] == r_rewards[a]
                if a not in r_dones:
                    assert puf_done[idx] == False
                else:
                    assert puf_done[idx] == r_dones[a]
                idx += 1

if __name__ == '__main__':
    import mock_environments
    for env_cls in mock_environments.MOCK_SINGLE_AGENT_ENVIRONMENTS:
        test_gym_emulation(env_cls)

    for env_cls in mock_environments.MOCK_MULTI_AGENT_ENVIRONMENTS:
        test_pettingzoo_emulation(env_cls)
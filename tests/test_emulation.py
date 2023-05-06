from pdb import set_trace as T

import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.registry
import pufferlib.vectorization.serial


class MockVecEnv:
    def __init__(self, binding, num_workers, envs_per_worker=1):
        self.binding = binding

        self.num_envs = num_workers * envs_per_worker
        self.envs = [binding.pz_env_creator()
            for _ in range(self.num_envs)]
        self.possible_agents = self.envs[0].possible_agents

    @property
    def agents(self):
        return [e.agents for e in self.envs]

    def reset(self, seed=42):
        return [e.reset() for e in self.envs]

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


def test_emulation(binding, steps=1000, num_workers=1, envs_per_worker=1):
    raw_env = MockVecEnv(binding, num_workers, envs_per_worker)
    puf_env = pufferlib.vectorization.serial.VecEnv(binding, num_workers, envs_per_worker)

    puf_ob = puf_env.reset()
    raw_ob = raw_env.reset()

    flat_atn_space = pufferlib.emulation._flatten_space(
        binding.single_action_space)

    for i in range(steps):
        # Reconstruct original obs format from puffer env and compare to raw
        orig_puf_ob = puf_ob
        puf_ob = binding.unpack_batched_obs(puf_ob)
        idx = 0
        for r_ob in raw_ob:
            for k in raw_env.possible_agents:
                if k not in r_ob:
                    assert not np.count_nonzero(orig_puf_ob[idx])
                else:
                    assert pufferlib.utils._compare_observations(
                        r_ob[k], puf_ob, idx=idx)
                idx += 1

        atn = [{a: binding.single_action_space.sample() for a in agents}
               for agents in raw_env.agents]
        raw_ob, raw_reward, raw_done, _ = raw_env.step(atn)


        # Convert actions to puffer format
        actions = []
        dummy = binding.single_action_space.sample()
        dummy = pufferlib.emulation._flatten_to_array(dummy, flat_atn_space)
        for agent in raw_env.possible_agents:
            if agent in atn:
                actions.append(pufferlib.emulation._flatten_to_array(atn[agent]), flat_atn_space)
            else:
                actions.append(dummy)
        puf_ob, puf_reward, puf_done, _ = puf_env.step(actions)

        idx = 0 
        for r_env, r_reward, r_done in zip(raw_env.envs, raw_reward, raw_done):
            for k in raw_env.possible_agents:
                if k not in r_reward:
                    assert puf_reward[idx] == 0
                    if r_env.agents:
                        assert puf_done[idx] == False
                    else:
                        assert puf_done[idx] == True
                else:
                    assert puf_reward[idx] == r_reward[k]
                    assert puf_done[idx] == r_done[k]
                idx += 1

class FeatureExtractor:
    def __init__(self, teams, team_id):
        pass

    def reset(self, obs):
        return

    def __call__(self, obs, step):
        assert len(obs) > 0
        return list(obs.values())

if __name__ == '__main__':
    import mock_environments
    for teams in mock_environments.MOCK_TEAMS[0:1]:
        for env in mock_environments.MOCK_ENVIRONMENTS:
            binding = pufferlib.emulation.Binding(
                    env_cls=env)#, teams=teams, featurizer_cls=FeatureExtractor)

            test_emulation(binding)
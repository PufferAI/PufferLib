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


def test_emulation(binding, teams, steps=100, num_workers=1, envs_per_worker=2):
    raw_env = MockVecEnv(binding, num_workers, envs_per_worker)
    puf_env = pufferlib.vectorization.serial.VecEnv(binding, num_workers, envs_per_worker)

    puf_ob = puf_env.reset()
    raw_ob = raw_env.reset()

    flat_atn_space = pufferlib.emulation._flatten_space(
        binding.raw_single_action_space)

    for i in range(steps):
        # Reconstruct original obs format from puffer env and compare to raw
        orig_puf_ob = puf_ob
        puf_ob = binding.unpack_batched_obs(puf_ob)
        idx = 0
        for r_ob in raw_ob:
            for team in teams.values():
                orig_team_ob = np.split(orig_puf_ob[idx], len(team))
                for i, k in enumerate(team):
                    if k not in r_ob:
                        assert not np.count_nonzero(orig_team_ob[i])
                    else:
                        assert pufferlib.utils._compare_observations(
                            r_ob[k], puf_ob[i], idx=idx)
                idx += 1

        atn = [{a: binding.raw_single_action_space.sample() for a in agents}
               for agents in raw_env.agents]
        raw_ob, raw_reward, raw_done, _ = raw_env.step(atn)

        # Convert actions to puffer format
        actions = []
        dummy = binding.raw_single_action_space.sample()
        dummy = pufferlib.emulation._flatten_to_array(dummy, flat_atn_space)
        for a in atn:
            for agent in raw_env.possible_agents:
                if agent in a:
                    actions.append(pufferlib.emulation._flatten_to_array(a[agent], flat_atn_space))
                else:
                    actions.append(dummy)

        # Flatten actions of each team
        idx = 0
        team_actions = []
        for team in teams.values():
            t_actions = []
            for agent in team:
                t_actions.append(actions[idx])
                idx += 1
            team_actions.append(np.concatenate(t_actions))

        puf_ob, puf_reward, puf_done, _ = puf_env.step(team_actions)

        idx = 0 
        for r_env, r_reward, r_done in zip(raw_env.envs, raw_reward, raw_done):
            for team in teams.values():
                team_done = [r_done[a] for a in team if a in r_done]
                team_reward = [r_reward[a] for a in team if a in r_reward]

                if len(team_reward) > 0:
                    assert puf_reward[idx] == sum(team_reward)
                else:
                    assert puf_reward[idx] == 0

                if len(team_done) > 0:
                    assert puf_done[idx] == any(team_done)
                else:
                    assert puf_done[idx] != bool(r_env.agents)

                idx += 1


if __name__ == '__main__':
    import mock_environments
    teams = mock_environments.MOCK_TEAMS['pairs']
    for env in mock_environments.MOCK_ENVIRONMENTS:
        binding = pufferlib.emulation.Binding(env_cls=env, teams=teams)
        test_emulation(binding, teams)
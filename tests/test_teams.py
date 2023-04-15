from pdb import set_trace as T
import nmmo

import pufferlib.vectorization
import pufferlib.emulation
import pufferlib.registry.nmmo

from mock_environments import MOCK_ENVIRONMENTS


class FeatureExtractor:
    def __init__(self, teams, team_id):
        pass

    def reset(self, obs):
        return

    def __call__(self, obs, step):
        return obs[list(obs)[0]]

def test_mock_envs():
    for env_cls in MOCK_ENVIRONMENTS:
        binding = pufferlib.emulation.Binding(
            env_cls=env_cls,
            env_name=env_cls.__name__,
            teams = {
                'team_1': ['agent_1', 'agent_2'],
                'team_2': ['agent_3', 'agent_4', 'agent_5', 'agent_6'],
                'team_3': ['agent_7', 'agent_8', 'agent_9'],
                'team_4': ['agent_10', 'agent_11', 'agent_12', 'agent_13', 'agent_14'],
                'team_5': ['agent_15', 'agent_16'],
            },
            featurizer_cls=FeatureExtractor,
        )

        env = binding.env_creator()
        obs = env.reset()

        while not env.done:
            actions = {}
            for agent, ob in obs.items():
                assert env.observation_space(agent).contains(ob)
                actions[agent] = env.action_space(agent).sample()

            obs, rewards, dones, infos = env.step(actions)


def test_nmmo():
    binding = pufferlib.emulation.Binding(
        env_cls=nmmo.Env,
        env_name='Neural MMO',
        teams = {i+1: [i*8+j+1 for j in range(8)] for i in range(16)},
        featurizer_cls=FeatureExtractor,
    )

    binding = pufferlib.emulation.Binding(
        env_cls=TestEnv,
        env_name='Test Environment',
        default_kwargs=dict(
            initial_agents=8,
            max_agents=8,
            spawn_attempts_per_tick=0,
            death_per_tick=1
        ),
        teams = {
            'team_1': ['agent_1', 'agent_2'],
            'team_2': ['agent_3', 'agent_4', 'agent_5', 'agent_6'],
            'team_3': ['agent_7', 'agent_8'],
        },
        featurizer_cls=FeatureExtractor,
    )


    env = binding.env_creator()
    obs = env.reset()

    actions = {}
    for agent, ob in obs.items():
        assert env.observation_space(agent).contains(ob)
        actions[agent] = env.action_space(agent).sample()

    obs, rewards, dones, infos = env.step(actions)

if __name__ == '__main__':
    test_mock_envs()

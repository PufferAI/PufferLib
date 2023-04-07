import nmmo

import pufferlib.emulation
import pufferlib.registry.nmmo

class FeatureExtractor:
    def __init__(self, team):
        pass

    def reset(self, obs):
        pass

    def __call__(self, obs, step):
        key = list(obs.keys())[0]
        return obs[key]


def test_nmmo():
    binding = pufferlib.emulation.Binding(
        env_cls=nmmo.Env,
        env_name='Neural MMO',
        feature_parser=FeatureExtractor,
    )

    env = binding.env_creator()
    obs = env.reset()

    actions = {}
    for agent, ob in obs.items():
        assert env.observation_space(agent).contains(ob)
        actions[agent] = env.action_space(agent).sample()

    obs, rewards, dones, infos = env.step(actions)

if __name__ == '__main__':
    test_nmmo()
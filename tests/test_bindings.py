from pdb import set_trace as T

import pufferlib
from environments import bindings

def test_envs():
    for binding in bindings.values():
        env = binding.env_creator()
        obs = env.reset()

        actions = {}
        for agent, ob in obs.items():
            assert env.observation_space(agent).contains(ob)
            actions[agent] = env.action_space(agent).sample()

        obs, rewards, dones, infos = env.step(actions)

if __name__ == '__main__':
    test_envs()
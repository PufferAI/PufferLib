from pdb import set_trace as T

import pufferlib
import env_defs


def test_envs():
    for binding in env_defs.bindings:
        env = binding.env_creator()
        obs = env.reset()

        actions = {}
        for agent, ob in obs.items():
            assert env.observation_space(agent).contains(ob)
            actions[agent] = env.action_space(agent).sample()

        obs, rewards, dones, infos = env.step(actions)

if __name__ == '__main__':
    test_envs()
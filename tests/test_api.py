from pdb import set_trace as T

import pufferlib
import pufferlib.emulation
import pufferlib.vector
from pufferlib.environments import test
import functools

def print_if(e, print_errors):
    if print_errors:
        print(type(e).__name__ + ':', e)
        print('#################')
        print()

def test_gymnasium_api(print_errors=False):
    env = pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=test.GymnasiumTestEnv)

    try:
        env.step({})
    except pufferlib.APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.close()
    except pufferlib.APIUsageError as e:
        print_if(e, print_errors)

    env.observation_space.sample()
    env.action_space.sample()
    ob = env.reset()

    try:
        bad_action = env.observation_space.sample()
        env.step(bad_action)
    except pufferlib.APIUsageError as e:
        print_if(e, print_errors)

    action = env.action_space.sample()
    obs, rewards, terminals, truncateds, infos = env.step(action)

def test_pettingzoo_api_usage(print_errors=False):
    env = pufferlib.emulation.PettingZooPufferEnv(
        env_creator=test.PettingZooTestEnv)

    try:
        env.step({})
    except pufferlib.APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.close()
    except pufferlib.APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.observation_space('foo')
    except pufferlib.InvalidAgentError as e:
        print_if(e, print_errors)

    try:
        env.action_space('foo')
    except pufferlib.InvalidAgentError as e:
        print_if(e, print_errors)

    env.observation_space('agent_1')
    env.action_space('agent_1')
    obs = env.reset()

    try:
        bad_actions = {agent: env.observation_space(agent).sample() for agent in env.agents}
        env.step(bad_actions)
    except pufferlib.APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.step({'foo': None})
    except pufferlib.InvalidAgentError as e:
        print_if(e, print_errors)


    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminals, truncateds, infos = env.step(actions)

def test_vectorization_api(print_errors=False):
    def gymnasium_creator(*args, buf = None, **kwargs):
        env = test.GymnasiumTestEnv(*args)
        return pufferlib.emulation.GymnasiumPufferEnv(*args, env=env, buf=buf)

    def pettingzoo_creator(*args, buf = None, **kwargs):
        env = test.PettingZooTestEnv(*args)
        return pufferlib.emulation.PettingZooPufferEnv(*args, env=env, buf=buf)
    
    for backend in [
        pufferlib.vector.Serial,
        pufferlib.vector.Multiprocessing,
        pufferlib.vector.Ray
    ]:
            
        for creator in [gymnasium_creator, pettingzoo_creator]:
            vec = pufferlib.vector.make(creator, num_envs=6,
                num_workers=3, backend=backend)

            # Sync API
            _, _ = vec.reset()
            actions = vec.action_space.sample()
            _, _, _, _, _ = vec.step(actions)
            vec.close()

            # Async API
            vec = pufferlib.vector.make(creator, num_envs=8,
                num_workers=4, batch_size=4, backend=backend)
            vec.async_reset()
            actions = vec.action_space.sample()
            _, _, _, _, _, _, _ = vec.recv()
            vec.send(actions)
            vec.close()

        try:
            vec = pufferlib.vector.make(test.GymnasiumTestEnv)
        except pufferlib.APIUsageError as e:
            print_if(e, print_errors)

        try:
            vec = pufferlib.vector.make(gymnasium_creator,
                num_envs=3, num_workers=2)
        except pufferlib.APIUsageError as e:
            print_if(e, print_errors)

        try:
            vec = pufferlib.vector.make(gymnasium_creator,
                num_envs=4, num_workers=2, batch_size=3)
        except pufferlib.APIUsageError as e:
            print_if(e, print_errors)


if __name__ == '__main__':
    test_gymnasium_api()
    test_pettingzoo_api_usage()
    test_vectorization_api()

from pdb import set_trace as T

import pufferlib
import pufferlib.emulation
import pufferlib.vectorization
from pufferlib.exceptions import APIUsageError, InvalidAgentError
from pufferlib.environments import test

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
    except APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.close()
    except APIUsageError as e:
        print_if(e, print_errors)

    env.observation_space.sample()
    env.action_space.sample()
    ob = env.reset()

    try:
        bad_action = env.observation_space.sample()
        env.step(bad_action)
    except APIUsageError as e:
        print_if(e, print_errors)

    action = env.action_space.sample()
    obs, rewards, terminals, truncateds, infos = env.step(action)

def test_pettingzoo_api_usage(print_errors=False):
    env = pufferlib.emulation.PettingZooPufferEnv(
        env_creator=test.PettingZooTestEnv)

    try:
        env.step({})
    except APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.close()
    except APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.observation_space('foo')
    except InvalidAgentError as e:
        print_if(e, print_errors)

    try:
        env.action_space('foo')
    except InvalidAgentError as e:
        print_if(e, print_errors)

    env.observation_space('agent_1')
    env.action_space('agent_1')
    obs = env.reset()

    try:
        bad_actions = {agent: env.observation_space(agent).sample() for agent in env.agents}
        env.step(bad_actions)
    except APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.step({'foo': None})
    except InvalidAgentError as e:
        print_if(e, print_errors)


    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminals, truncateds, infos = env.step(actions)

def test_vectorization_api(print_errors=False):
    gymnasium_creator = lambda: pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=test.GymnasiumTestEnv)
    pettingzoo_creator = lambda: pufferlib.emulation.PettingZooPufferEnv(
        env_creator=test.PettingZooTestEnv)

    for backend in [
        pufferlib.vectorization.Serial,
        pufferlib.vectorization.Multiprocessing,
        pufferlib.vectorization.Ray]:
            
        for creator in [gymnasium_creator, pettingzoo_creator]:
            # Sync API
            vec = backend(env_creator=creator,
                num_envs=6, envs_per_worker=2)
            _, _, _, _ = vec.reset()
            actions = [vec.single_action_space.sample()
                for _ in range(vec.agents_per_batch)]
            _, _, _, _, _, _, _ = vec.step(actions)
            vec.close()

            # Async API
            vec = backend(env_creator=creator,
                num_envs=6, envs_per_worker=2, envs_per_batch=4)
            vec.async_reset()
            actions = [vec.single_action_space.sample()
                for _ in range(vec.agents_per_batch)]
            _, _, _, _, _, _, _ = vec.recv()
            vec.send(actions)
            vec.close()

        try:
            vec = backend(env_creator=test.GymnasiumTestEnv)
        except TypeError as e:
            print_if(e, print_errors)

        try:
            vec = backend(env_creator=gymnasium_creator,
                num_envs=3, envs_per_worker=2)
        except APIUsageError as e:
            print_if(e, print_errors)

        try:
            vec = backend(env_creator=pettingzoo_creator,
                num_envs=4, envs_per_worker=2, envs_per_batch=3)
        except APIUsageError as e:
            print_if(e, print_errors)



if __name__ == '__main__':
    test_gymnasium_api()
    test_pettingzoo_api_usage()
    test_vectorization_api()

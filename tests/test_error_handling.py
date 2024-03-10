from pdb import set_trace as T

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions

def print_if(e, print_errors):
    if print_errors:
        print(type(e).__name__ + ':', e)
        print('#################')
        print()

def test_gym_api_usage(env_cls, print_errors=True):
    env = pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_cls)

    try:
        env.step({})
    except pufferlib.exceptions.APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.close()
    except pufferlib.exceptions.APIUsageError as e:
        print_if(e, print_errors)

    env.observation_space.sample()
    env.action_space.sample()
    ob = env.reset()

    try:
        bad_action = env.observation_space.sample()
        env.step(bad_action)
    except ValueError as e:
        print_if(e, print_errors)

    action = env.action_space.sample()
    obs, rewards, terminals, truncateds, infos = env.step(action)

def test_pettingzoo_api_usage(env_cls, print_errors=True):
    env = pufferlib.emulation.PettingZooPufferEnv(env_creator=env_cls)

    try:
        env.step({})
    except pufferlib.exceptions.APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.close()
    except pufferlib.exceptions.APIUsageError as e:
        print_if(e, print_errors)

    try:
        env.observation_space('foo')
    except pufferlib.exceptions.InvalidAgentError as e:
        print_if(e, print_errors)

    try:
        env.action_space('foo')
    except pufferlib.exceptions.InvalidAgentError as e:
        print_if(e, print_errors)

    env.observation_space('agent_1')
    env.action_space('agent_1')
    obs = env.reset()

    try:
        bad_actions = {agent: env.observation_space(agent).sample() for agent in env.agents}
        env.step(bad_actions)
    except ValueError as e:
        print_if(e, print_errors)

    try:
        env.step({'foo': None})
    except pufferlib.exceptions.InvalidAgentError as e:
        print_if(e, print_errors)


    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminals, truncateds, infos = env.step(actions)

if __name__ == '__main__':
    from pufferlib.environments import test
    first = True

    
    test_gym_api_usage(test.GymnasiumTestEnv)
    test_pettingzoo_api_usage(test.PettingZooTestEnv)

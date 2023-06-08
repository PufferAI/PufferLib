from pdb import set_trace as T

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions

def print_if(e, print_errors):
    if print_errors:
        print(type(e).__name__ + ':', e)
        print('#################')
        print()

def test_env_api_usage(binding, print_errors=True):
    env = binding.env_creator()

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
    obs, rewards, dones, infos = env.step(actions)

if __name__ == '__main__':
    import mock_environments
    first = True
    for env in mock_environments.MOCK_ENVIRONMENTS:
        binding = pufferlib.emulation.Binding(env_cls=env)
        test_env_api_usage(binding, print_errors=first)
        first = False
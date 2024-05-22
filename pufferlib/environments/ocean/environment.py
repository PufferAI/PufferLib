import pufferlib.emulation
import pufferlib.postprocess

from . import ocean

def env_creator(name='squared'):
    if name == 'squared':
        return make_squared
    elif name == 'bandit':
        return make_bandit
    elif name == 'memory':
        return make_memory
    elif name == 'password':
        return make_password
    elif name == 'stochastic':
        return make_stochastic
    elif name == 'multiagent':
        return make_multiagent
    elif name == 'spaces':
        return make_spaces
    elif name == 'performance':
        return make_performance
    elif name == 'performance_empiric':
        return make_performance_empiric
    else:
        raise ValueError('Invalid environment name')

def make_squared(distance_to_target=3, num_targets=1, **kwargs):
    env = ocean.Squared(distance_to_target=distance_to_target, num_targets=num_targets)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, **kwargs)

def make_bandit(num_actions=10, reward_scale=1, reward_noise=1):
    env = ocean.Bandit(num_actions=num_actions, reward_scale=reward_scale,
        reward_noise=reward_noise)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_memory(mem_length=2, mem_delay=2):
    env = ocean.Memory(mem_length=mem_length, mem_delay=mem_delay)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_password(password_length=5):
    env = ocean.Password(password_length=password_length)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_performance(delay_mean=0, delay_std=0, bandwidth=1):
    env = ocean.Performance(delay_mean=delay_mean, delay_std=delay_std, bandwidth=bandwidth)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_performance_empiric(count_n=0, count_std=0, bandwidth=1):
    env = ocean.PerformanceEmpiric(count_n=count_n, count_std=count_std, bandwidth=bandwidth)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_stochastic(p=0.7, horizon=100):
    env = ocean.Stochastic(p=p, horizon=100)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_spaces(**kwargs):
    env = ocean.Spaces()
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, **kwargs)

def make_multiagent():
    env = ocean.Multiagent()
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

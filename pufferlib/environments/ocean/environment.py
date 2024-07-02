import pufferlib.emulation
import pufferlib.postprocess

from . import ocean
from .grid import grid
from .snake import snake

def env_creator(name='squared'):
    if name == 'grid':
        return make_grid
    elif name == 'snake':
        return make_snake
    elif name == 'squared':
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

def make_grid(map_size=512, num_agents=1024, horizon=512, render_mode='rgb_array'):
    #env = grid.PufferGrid(map_size, num_agents, horizon, render_mode=render_mode)
    env = grid.PufferGrid(64, 64, 64, render_mode=render_mode)
    return env
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    env = pufferlib.postprocess.MeanOverAgents(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

def make_snake(width=1024, height=1024, snakes=1024, food=1024, render_mode='ansi'):
    return snake.Snake(
        widths=[80],
        heights=[45],
        num_snakes=[8],
        num_food=[64],
        leave_corpse_on_death=True,
        teleport_at_edge=True,
        render_mode='rgb_array',
    )
 
    return snake.Snake(
        widths=[2560],
        heights=[1440],
        num_snakes=[4096],
        num_food=[16384],
        leave_corpse_on_death=True,
        render_mode=render_mode,
    )
 

    '''
    return snake.Snake(
        widths=[1024],
        heights=[1024],
        num_snakes=[1024],
        num_food=[1024],
        leave_corpse_on_death=True,
        render_mode=render_mode
    )
    '''
 
    return snake.Snake(
        widths=4096*[48],
        heights=4096*[48],
        num_snakes=4096*[1],
        num_food=4096*[16],
        leave_corpse_on_death=False,
        render_mode=render_mode
    )
 
    curriculum = 1024*[16] + 512*[21] + 512*[26]
    #curriculum = 512*[16] + 512*[4] + 1024*[1]
    return snake.Snake(
        widths=curriculum,
        heights=curriculum,
        num_snakes=2048*[1],
        num_food=2048*[1],
        leave_corpse_on_death=False,
        render_mode=render_mode
    )
 
    '''
    return snake.Snake(
        widths=[26],
        heights=[26],
        num_snakes=[1],
        num_food=[1],
        leave_corpse_on_death=True,
        render_mode=render_mode
    )
    '''
 
    return snake.Snake(
        widths=1024*[26],
        heights=1024*[26],
        num_snakes=1024*[1],
        num_food=1024*[1],
        leave_corpse_on_death=True,
        render_mode=render_mode
    )
 
    return snake.Snake(
        widths=[1024],
        heights=[1024],
        num_snakes=[1024],
        num_food=[1024],
        render_mode=render_mode
    )
    return snake.Snake(width=width, height=height, snakes=snakes,
        food=food, render_mode=render_mode)

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

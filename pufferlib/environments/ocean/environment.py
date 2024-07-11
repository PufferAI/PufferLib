import pufferlib.emulation
import pufferlib.postprocess

from . import ocean
from .grid import grid
from .grid_continuous import grid_continuous
from .snake import snake
from .continuous import continuous

def env_creator(name='squared'):
    if name == 'grid':
        return make_grid
    elif name == 'grid_continuous':
        return make_grid_continuous
    elif name == 'snake':
        return make_snake
    elif name == 'continuous':
        return make_continuous
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
    env = grid.PufferGrid(map_size, num_agents, horizon, render_mode=render_mode)
    #env = grid.PufferGrid(64, 64, 64, render_mode=render_mode)
    return env
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    env = pufferlib.postprocess.MeanOverAgents(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

def make_grid_continuous(width=1080, height=720, num_agents=4096,
        horizon=1024, render_mode='rgb_array'):
    env = grid_continuous.PufferGrid(width, height, num_agents,
        horizon, discretize=True, render_mode=render_mode)
    return env

def make_snake(widths=None, heights=None, num_snakes=None, num_food=None, vision=5,
        leave_corpse_on_death=None, preset='1440p-4096', render_mode=None):
    if preset is None:
        render_mode = render_mode or 'rgb_array'
    elif preset == '1440p-4096':
        widths = widths or [2560]
        heights = heights or [1440]
        num_snakes = num_snakes or [4096]
        num_food = num_food or [65536]
        leave_corpse_on_death = leave_corpse_on_death or True
        render_mode = render_mode or 'human'
    elif preset == '720p-1024':
        widths = widths or 4*[1280]
        heights = heights or 4*[720]
        num_snakes = num_snakes or 4*[1024]
        num_food = num_food or 4*[16384]
        leave_corpse_on_death = leave_corpse_on_death or True
        render_mode = render_mode or 'rgb_array'
    elif preset == '40p-4':
        widths = widths or 1024*[40]
        heights = heights or 1024*[40]
        num_snakes = num_snakes or 1024*[4]
        num_food = num_food or 1024*[16]
        leave_corpse_on_death = leave_corpse_on_death or True
        render_mode = render_mode or 'ansi'
    elif preset == 'classic':
        widths = widths or 4096*[26]
        heights = heights or 4096*[26]
        num_snakes = num_snakes or 4096*[1]
        num_food = num_food or 4096*[1]
        leave_corpse_on_death = leave_corpse_on_death or False
        render_mode = render_mode or 'ansi'
    else:
        raise ValueError(
            f'Preset: {preset} must be 1440p-4096, 720p-1024, 40p-4, or classic')
    
    return snake.Snake(
        widths=widths,
        heights=heights,
        num_snakes=num_snakes,
        num_food=num_food,
        leave_corpse_on_death=leave_corpse_on_death,
        render_mode=render_mode,
        vision=vision,
    )

def make_continuous(discretize=False):
    env = continuous.Continuous(discretize=discretize)
    if not discretize:
        env = pufferlib.postprocess.ClipAction(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

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

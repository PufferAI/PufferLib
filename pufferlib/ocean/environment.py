import pufferlib.emulation
import pufferlib.postprocess

from .snake.snake import Snake
from .squared.squared import Squared
from .squared.pysquared import PySquared
from .pong.pong import Pong
from .breakout.breakout import Breakout
from .enduro.enduro import Enduro
from .connect4.connect4 import Connect4
from .tripletriad.tripletriad import TripleTriad
from .tactical.tactical import Tactical
from .moba.moba import Moba
from .nmmo3.nmmo3 import NMMO3
from .go.go import Go
from .rware.rware import Rware
from .grid.grid import PufferGrid
#from .rocket_lander import rocket_lander
from .trash_pickup.trash_pickup import TrashPickupEnv
from .tower_climb.tower_climb import TowerClimb

def make_foraging(width=1080, height=720, num_agents=4096, horizon=512,
        discretize=True, food_reward=0.1, render_mode='rgb_array'):
    from .grid import grid
    init_fn = grid.init_foraging
    reward_fn = grid.reward_foraging
    return grid.PufferGrid(width, height, num_agents,
        horizon, discretize=discretize, food_reward=food_reward, init_fn=init_fn, reward_fn=reward_fn, render_mode=render_mode)

def make_predator_prey(width=1080, height=720, num_agents=4096, horizon=512,
        discretize=True, food_reward=0.1, render_mode='rgb_array'):
    from .grid import grid
    init_fn = grid.init_predator_prey
    reward_fn = grid.reward_predator_prey
    return grid.PufferGrid(width, height, num_agents,
        horizon, discretize=discretize, food_reward=food_reward,
        init_fn=init_fn, reward_fn=reward_fn,
        render_mode=render_mode)

def make_group(width=1080, height=720, num_agents=4096, horizon=512,
        discretize=True, food_reward=0.1, render_mode='rgb_array'):
    from .grid import grid
    init_fn = grid.init_group
    reward_fn = grid.reward_group
    return grid.PufferGrid(width, height, num_agents,
        horizon, discretize=discretize, food_reward=food_reward,
        init_fn=init_fn, reward_fn=reward_fn,
        render_mode=render_mode)

def make_puffer(width=1080, height=720, num_agents=4096, horizon=512,
        discretize=True, food_reward=0.1, render_mode='rgb_array'):
    from .grid import grid
    init_fn = grid.init_puffer
    reward_fn = grid.reward_puffer
    return grid.PufferGrid(width, height, num_agents,
        horizon, discretize=discretize, food_reward=food_reward,
        init_fn=init_fn, reward_fn=reward_fn,
        render_mode=render_mode)

def make_puffergrid(render_mode='raylib', vision_range=5,
        num_envs=4096, num_maps=1000, max_map_size=9,
        report_interval=128, buf=None):
    return PufferGrid(render_mode, vision_range, num_envs,
        num_maps, max_map_size, report_interval, buf)

def make_continuous(discretize=False, buf=None, **kwargs):
    from . import sanity
    env = sanity.Continuous(discretize=discretize)
    if not discretize:
        env = pufferlib.postprocess.ClipAction(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

def make_squared(distance_to_target=3, num_targets=1, buf=None, **kwargs):
    from . import sanity
    env = sanity.Squared(distance_to_target=distance_to_target, num_targets=num_targets, **kwargs)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, **kwargs)

def make_bandit(num_actions=10, reward_scale=1, reward_noise=1, buf=None):
    from . import sanity
    env = sanity.Bandit(num_actions=num_actions, reward_scale=reward_scale,
        reward_noise=reward_noise)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

def make_memory(mem_length=2, mem_delay=2, buf=None, **kwargs):
    from . import sanity
    env = sanity.Memory(mem_length=mem_length, mem_delay=mem_delay)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

def make_password(password_length=5, buf=None, **kwargs):
    from . import sanity
    env = sanity.Password(password_length=password_length)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

def make_performance(delay_mean=0, delay_std=0, bandwidth=1, buf=None, **kwargs):
    from . import sanity
    env = sanity.Performance(delay_mean=delay_mean, delay_std=delay_std, bandwidth=bandwidth)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

def make_performance_empiric(count_n=0, count_std=0, bandwidth=1, buf=None, **kwargs):
    from . import sanity
    env = sanity.PerformanceEmpiric(count_n=count_n, count_std=count_std, bandwidth=bandwidth)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

def make_stochastic(p=0.7, horizon=100, buf=None, **kwargs):
    from . import sanity
    env = sanity.Stochastic(p=p, horizon=100)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

def make_spaces(buf=None, **kwargs):
    from . import sanity
    env = sanity.Spaces()
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, **kwargs)

def make_multiagent(buf=None, **kwargs):
    from . import sanity
    env = sanity.Multiagent()
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env, buf=buf)

MAKE_FNS = {
    'breakout': Breakout,
    'pong': Pong,
    'enduro': Enduro,
    'moba': Moba,
    'nmmo3': NMMO3,
    'snake': Snake,
    'squared': Squared,
    'pysquared': PySquared,
    'connect4': Connect4,
    'tripletriad': TripleTriad,
    'tactical': Tactical,
    'go': Go,
    'rware': Rware,
    'trash_pickup': TrashPickupEnv,
    'tower_climb': TowerClimb,
    #'rocket_lander': rocket_lander.RocketLander,
    'foraging': make_foraging,
    'predator_prey': make_predator_prey,
    'group': make_group,
    'puffer': make_puffer,
    'puffer_grid': make_puffergrid,
    'continuous': make_continuous,
    'bandit': make_bandit,
    'memory': make_memory,
    'password': make_password,
    'stochastic': make_stochastic,
    'multiagent': make_multiagent,
    'spaces': make_spaces,
    'performance': make_performance,
    'performance_empiric': make_performance_empiric,
}

# Alias puffer_ to all names
MAKE_FNS = {**MAKE_FNS, **{'puffer_' + k: v for k, v in MAKE_FNS.items()}}

def env_creator(name='squared'):
    if name in MAKE_FNS:
        return MAKE_FNS[name]
    else:
        raise ValueError(f'Invalid environment name: {name}')



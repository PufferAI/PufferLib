import pufferlib.emulation
import pufferlib.postprocess

def lazy_import(module_path, attr):
    """
    Returns a callable that, when called with any arguments, will
    import the module, retrieve the attribute (usually a class or factory)
    and then call it with the given arguments.
    """
    return lambda *args, **kwargs: getattr(__import__(module_path, fromlist=[attr]), attr)(*args, **kwargs)

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
    'breakout':      lambda: lazy_import('pufferlib.ocean.breakout.breakout', 'Breakout'),
    'blastar':       lambda: lazy_import('pufferlib.ocean.blastar.blastar', 'Blastar'),
    'pong':          lambda: lazy_import('pufferlib.ocean.pong.pong', 'Pong'),
    'enduro':        lambda: lazy_import('pufferlib.ocean.enduro.enduro', 'Enduro'),
    'cartpole':      lambda: lazy_import('pufferlib.ocean.cartpole.cartpole', 'Cartpole'),
    'moba':          lambda: lazy_import('pufferlib.ocean.moba.moba', 'Moba'),
    'nmmo3':         lambda: lazy_import('pufferlib.ocean.nmmo3.nmmo3', 'NMMO3'),
    'snake':         lambda: lazy_import('pufferlib.ocean.snake.snake', 'Snake'),
    'squared':       lambda: lazy_import('pufferlib.ocean.squared.squared', 'Squared'),
    'pysquared':     lambda: lazy_import('pufferlib.ocean.squared.pysquared', 'PySquared'),
    'connect4':      lambda: lazy_import('pufferlib.ocean.connect4.connect4', 'Connect4'),
    'tripletriad':   lambda: lazy_import('pufferlib.ocean.tripletriad.tripletriad', 'TripleTriad'),
    'tactical':      lambda: lazy_import('pufferlib.ocean.tactical.tactical', 'Tactical'),
    'go':            lambda: lazy_import('pufferlib.ocean.go.go', 'Go'),
    'rware':         lambda: lazy_import('pufferlib.ocean.rware.rware', 'Rware'),
    'trash_pickup':  lambda: lazy_import('pufferlib.ocean.trash_pickup.trash_pickup', 'TrashPickupEnv'),
    'tower_climb':   lambda: lazy_import('pufferlib.ocean.tower_climb.tower_climb', 'TowerClimb'),
    'grid':          lambda: lazy_import('pufferlib.ocean.grid.grid', 'Grid'),
    'cpr':           lambda: lazy_import('pufferlib.ocean.cpr.cpr', 'PyCPR'),
    'impulse_wars':  lambda: lazy_import('pufferlib.ocean.impulse_wars.impulse_wars', 'ImpulseWars'),
    'gpudrive':      lambda: lazy_import('pufferlib.ocean.gpudrive.gpudrive', 'GPUDrive'),
    #'rocket_lander': rocket_lander.RocketLander,
    'foraging': make_foraging,
    'predator_prey': make_predator_prey,
    'group': make_group,
    'puffer': make_puffer,
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

def env_creator(name='squared', *args, **kwargs):
    if name in MAKE_FNS:
        return MAKE_FNS[name](*args, **kwargs)
    else:
        raise ValueError(f'Invalid environment name: {name}')



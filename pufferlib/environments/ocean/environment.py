import pufferlib.emulation
import pufferlib.postprocess

def make_moba(num_envs=200, reward_death=-1.0, reward_xp=0.006,
        reward_distance=0.05, reward_tower=3, render_mode='rgb_array'):
    from .moba import moba
    return moba.PufferMoba(num_envs=num_envs, reward_death=reward_death,
        reward_xp=reward_xp, reward_distance=reward_distance,
        reward_tower=reward_tower, render_mode=render_mode)

def make_pong(num_envs=1):
    from .pong import pong
    return pong.MyPong(num_envs=num_envs)

def make_breakout(num_envs=1):
    from .breakout import breakout
    return breakout.MyBreakout(num_envs=num_envs)

def make_tripletriad(num_envs=1):
    from .tripletriad import tripletriad
    return tripletriad.MyTripleTriad(num_envs=num_envs)

def make_foraging(width=1080, height=720, num_agents=4096, horizon=512,
        discretize=True, food_reward=0.1, render_mode='rgb_array'):
    from .grid import grid
    init_fn = grid.init_foraging
    reward_fn = grid.reward_foraging
    return grid.PufferGrid(width, height, num_agents,
        horizon, discretize=discretize, food_reward=food_reward,
        init_fn=init_fn, reward_fn=reward_fn,
        render_mode=render_mode)

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

def make_snake(widths=None, heights=None, num_snakes=None, num_food=None, vision=5,
        leave_corpse_on_death=None, preset='1440p-4096', render_mode=None):
    # TODO: Fix render_mode
    if preset is None:
        render_mode = render_mode or 'rgb_array'
    elif preset == '1440p-4096':
        widths = widths or [2560]
        heights = heights or [1440]
        num_snakes = num_snakes or [4096]
        num_food = num_food or [65536]
        leave_corpse_on_death = leave_corpse_on_death or True
        render_mode = render_mode or 'rgb_array'
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
    
    from .snake import csnake
    return csnake.Snake(
        widths=widths,
        heights=heights,
        num_snakes=num_snakes,
        num_food=num_food,
        leave_corpse_on_death=leave_corpse_on_death,
        render_mode=render_mode,
        vision=vision,
    )

def make_continuous(discretize=False, **kwargs):
    from . import sanity
    env = sanity.Continuous(discretize=discretize)
    if not discretize:
        env = pufferlib.postprocess.ClipAction(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_squared(distance_to_target=3, num_targets=1, **kwargs):
    from . import sanity
    env = sanity.Squared(distance_to_target=distance_to_target, num_targets=num_targets, **kwargs)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, **kwargs)

def make_bandit(num_actions=10, reward_scale=1, reward_noise=1):
    from . import sanity
    env = sanity.Bandit(num_actions=num_actions, reward_scale=reward_scale,
        reward_noise=reward_noise)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_memory(mem_length=2, mem_delay=2, **kwargs):
    from . import sanity
    env = sanity.Memory(mem_length=mem_length, mem_delay=mem_delay)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_password(password_length=5, **kwargs):
    from . import sanity
    env = sanity.Password(password_length=password_length)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_performance(delay_mean=0, delay_std=0, bandwidth=1, **kwargs):
    from . import sanity
    env = sanity.Performance(delay_mean=delay_mean, delay_std=delay_std, bandwidth=bandwidth)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_performance_empiric(count_n=0, count_std=0, bandwidth=1, **kwargs):
    from . import sanity
    env = sanity.PerformanceEmpiric(count_n=count_n, count_std=count_std, bandwidth=bandwidth)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_stochastic(p=0.7, horizon=100, **kwargs):
    from . import sanity
    env = sanity.Stochastic(p=p, horizon=100)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_spaces(**kwargs):
    from . import sanity
    env = sanity.Spaces()
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, **kwargs)

def make_multiagent(**kwargs):
    from . import sanity
    env = sanity.Multiagent()
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

MAKE_FNS = {
    'moba': make_moba,
    'my_pong': make_pong,
    'foraging': make_foraging,
    'predator_prey': make_predator_prey,
    'group': make_group,
    'puffer': make_puffer,
    'snake': make_snake,
    'continuous': make_continuous,
    'squared': make_squared,
    'bandit': make_bandit,
    'memory': make_memory,
    'password': make_password,
    'stochastic': make_stochastic,
    'multiagent': make_multiagent,
    'spaces': make_spaces,
    'performance': make_performance,
    'performance_empiric': make_performance_empiric,
    'my_breakout': make_breakout,
    'my_tripletriad': make_tripletriad,
}

def env_creator(name='squared'):
    if name in MAKE_FNS:
        return MAKE_FNS[name]
    else:
        raise ValueError(f'Invalid environment name: {name}')



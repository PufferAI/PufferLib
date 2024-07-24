from pufferlib.environments import ocean
import numpy as np

def test_performance(env, actions, num_envs, timeout=20):
    n_actions = actions.shape[0]

    tick = 0
    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % n_actions]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', 10 * num_envs * tick / (time.time() - start))

if __name__ == '__main__':
    # Run with c profile
    #from cProfile import run
    #run('test_puffer_performance(10)', sort='tottime')
    #exit(0)

    num_envs = 10
    make_env = ocean.env_creator('moba')
    env = make_env(num_envs=num_envs)
    env.reset()

    actions = np.random.randint(0, 2, (1024, num_envs*10, 6))

    #test_performance(10)
    import cProfile
    cProfile.run('test_performance(env, actions, num_envs, timeout=10)', 'stats.profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('stats.profile')
    p.sort_stats(SortKey.TIME).print_stats(25)

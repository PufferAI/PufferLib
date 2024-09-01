import numpy as np

import pufferlib
import pufferlib.emulation
from pufferlib.environments.gpudrive.environment import PufferCPUDrive

def test_performance(timeout=20):
    import time
    N = actions.shape[0]
    idx = 0
    dones = {1: True}
    start = time.time()
    while time.time() - start < timeout:
        _, _, dones, _, _, _, mask = env.step(actions[idx%N])
        idx += np.sum(mask)

    sps = idx // timeout
    print(f'SPS: {sps}')

if __name__ == '__main__':
    # Run with c profile
    env = PufferCPUDrive()
    env.reset()
    actions = np.random.randint(0, 9, (1024, env.num_agents))
    test_performance()
    exit(0)

    from cProfile import run
    run('test_performance(20)', 'stats.profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('stats.profile')
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)

 


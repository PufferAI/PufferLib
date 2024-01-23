from pdb import set_trace as T
import numpy as np
import time

from pufferlib.vectorization import Multiprocessing
from pufferlib.environments import pokemon_red

def test_envpool(num_envs, envs_per_worker, envs_per_batch, steps=1000, env_pool=True):
    pool = Multiprocessing(pokemon_red.env_creator(), num_envs=num_envs,
        envs_per_worker=envs_per_worker, envs_per_batch=envs_per_batch,
        env_pool=True,
    )
    pool.async_reset()

    a = np.array([pool.single_action_space.sample() for _ in range(envs_per_batch)])
    start = time.time()
    for s in range(steps):
        o, r, d, t, i, mask, env_id = pool.recv()
        pool.send(a)
    end = time.time()
    print('Steps per second: ', envs_per_batch * steps / (end - start))
    pool.close()


if __name__ == '__main__':
    # 225 sps
    #test_envpool(num_envs=1, envs_per_worker=1, envs_per_batch=1, env_pool=False)

    # 600 sps
    #test_envpool(num_envs=6, envs_per_worker=1, envs_per_batch=6, env_pool=False)

    # 645 sps
    #test_envpool(num_envs=24, envs_per_worker=4, envs_per_batch=24, env_pool=False)

    # 755 sps 
    # test_envpool(num_envs=24, envs_per_worker=4, envs_per_batch=24)

    # 1050 sps
    # test_envpool(num_envs=48, envs_per_worker=4, envs_per_batch=24)

    # 1300 sps
    test_envpool(num_envs=48, envs_per_worker=4, envs_per_batch=12)

from pdb import set_trace as T
import numpy as np
import time

from pufferlib.vectorization import Multiprocessing
from pufferlib.registry import pokemon_red

def test_envpool(workers=24, steps=100):
    pool = Multiprocessing(pokemon_red.make_env, num_workers=workers)
    pool.async_reset()

    start = time.time()
    for s in range(steps):
        o, r, d, t, i, mask = pool.recv()
        a = np.array([pool.single_action_space.sample() for _ in mask])
        pool.send(a)
    end = time.time()
    print('Steps per second: ', steps / (end - start))
    pool.close()


if __name__ == '__main__':
    test_envpool()

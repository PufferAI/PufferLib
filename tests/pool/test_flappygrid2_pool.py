from pufferlib.vector import Multiprocessing
from pufferlib.environments import flappygrid2
import numpy as np
import time


def test_flappygrid2_multiprocessing():
    """Test FlappyGrid2 with multiprocessing vectorization."""
    num_envs = 4
    
    def env_factory(*args, **kwargs):
        kwargs.pop('buf', None)
        kwargs.pop('seed', None)
        factory = flappygrid2.env_creator()
        return factory(*args, **kwargs)
    
    env_factories = [env_factory for _ in range(num_envs)]
    
    pool = Multiprocessing(
        env_factories,
        env_args=[() for _ in range(num_envs)],
        env_kwargs=[{} for _ in range(num_envs)],
        num_envs=num_envs,
        envs_per_worker=1,
        envs_per_batch=2,
        env_pool=True,
    )
    
    pool.async_reset()
    
   # Even better - let the pool tell you the right shape
    a = pool.action_space.sample()
    start = time.time()
    
    steps = 1000
    for _ in range(steps):
        o, r, d, t, i, mask, env_id = pool.recv()
        pool.send(a)
    
    end = time.time()
    steps_per_second = 2 * steps / (end - start)
    print(f'Multiprocessing FPS: {steps_per_second:,.0f}')
    
    pool.close()
    assert steps_per_second > 1000
    print("✅ Multiprocessing test passed")


if __name__ == "__main__":
    test_flappygrid2_multiprocessing()
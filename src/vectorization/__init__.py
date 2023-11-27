import gym

from pufferlib.vectorization import vec_env, serial_vec_env, multiprocessing_vec_env, ray_vec_env


class Serial:
    '''Runs environments in serial on the main process
    
    Use this vectorization module for debugging environments
    '''
    __init__ = serial_vec_env.init
    single_observation_space = property(vec_env.single_observation_space)
    single_action_space = property(vec_env.single_action_space)
    structured_observation_space = property(vec_env.structured_observation_space)
    flat_observation_space = property(vec_env.flat_observation_space)
    unpack_batched_obs = vec_env.unpack_batched_obs
    send = serial_vec_env.send
    recv = serial_vec_env.recv
    async_reset = serial_vec_env.async_reset
    profile = serial_vec_env.profile
    reset = serial_vec_env.reset
    step = serial_vec_env.step
    put = serial_vec_env.put
    get = serial_vec_env.get
    close = serial_vec_env.close

class Multiprocessing:
    '''Runs environments in parallel using multiprocessing

    Use this vectorization module for most applications
    '''
    __init__ = multiprocessing_vec_env.init
    single_observation_space = property(vec_env.single_observation_space)
    single_action_space = property(vec_env.single_action_space)
    structured_observation_space = property(vec_env.structured_observation_space)
    flat_observation_space = property(vec_env.flat_observation_space)
    unpack_batched_obs = vec_env.unpack_batched_obs
    send = multiprocessing_vec_env.send
    recv = multiprocessing_vec_env.recv
    async_reset = multiprocessing_vec_env.async_reset
    profile = multiprocessing_vec_env.profile
    reset = multiprocessing_vec_env.reset
    step = multiprocessing_vec_env.step
    put = multiprocessing_vec_env.put
    get = multiprocessing_vec_env.get
    close = multiprocessing_vec_env.close

class Ray:
    '''Runs environments in parallel on multiple processes using Ray

    Use this module for distributed simulation on a cluster. It can also be
    faster than multiprocessing on a single machine for specific environments.
    '''
    __init__ = ray_vec_env.init
    single_observation_space = property(vec_env.single_observation_space)
    single_action_space = property(vec_env.single_action_space)
    structured_observation_space = property(vec_env.structured_observation_space)
    flat_observation_space = property(vec_env.flat_observation_space)
    unpack_batched_obs = vec_env.unpack_batched_obs
    send = ray_vec_env.send
    recv = ray_vec_env.recv
    async_reset = ray_vec_env.async_reset
    profile = ray_vec_env.profile
    reset = ray_vec_env.reset
    step = ray_vec_env.step
    put = ray_vec_env.put
    get = ray_vec_env.get
    close = ray_vec_env.close

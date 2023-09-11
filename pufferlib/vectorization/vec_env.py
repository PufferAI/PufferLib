from pdb import set_trace as T

import numpy as np
from itertools import chain

from pufferlib.utils import namespace
from pufferlib.emulation import GymPufferEnv, PettingZooPufferEnv
from pufferlib.vectorization.multi_env import create_precheck
from pufferlib.vectorization.gym_multi_env import GymMultiEnv
from pufferlib.vectorization.pettingzoo_multi_env import PettingZooMultiEnv


RESET = 0
SEND = 1
RECV = 2

space_error_msg = 'env {env} must be an instance of GymPufferEnv or PettingZooPufferEnv'


def setup(env_creator, env_args, env_kwargs, num_workers, envs_per_worker):
    env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)

    driver_env = env_creator(*env_args, **env_kwargs)
    if isinstance(driver_env, GymPufferEnv):
        multi_env_cls = GymMultiEnv
        num_agents = 1
    elif isinstance(driver_env, PettingZooPufferEnv):
        multi_env_cls = PettingZooMultiEnv
        num_agents = len(driver_env.possible_agents)
    else:
        raise TypeError(
            'env_creator must return an instance '
            'of GymPufferEnv or PettingZooPufferEnv'
        )

    obs_space = _single_observation_space(driver_env)
    preallocated_obs = np.empty((
        num_agents * num_workers * envs_per_worker, *obs_space.shape,
    ), dtype=obs_space.dtype)

    return driver_env, multi_env_cls, num_agents, preallocated_obs

def _single_observation_space(env):
    if isinstance(env, GymPufferEnv):
        return env.observation_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_observation_space
    else:
        raise TypeError(space_error_msg.format(env=env))

def single_observation_space(state):
    return _single_observation_space(state.driver_env)

def _single_action_space(env):
    if isinstance(env, GymPufferEnv):
        return env.action_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_action_space
    else:
        raise TypeError(space_error_msg.format(env=env))

def single_action_space(state):
    return _single_action_space(state.driver_env)

def structured_observation_space(state):
    return state.driver_env.structured_observation_space

def flat_observation_space(state):
    return state.driver_env.flat_observation_space

def unpack_batched_obs(state, obs):
    return state.driver_env.unpack_batched_obs(obs)

def recv_precheck(state):
    assert state.flag == RECV, 'Call reset before stepping'
    state.flag = SEND

def send_precheck(state):
    assert state.flag == SEND, 'Call reset + recv before send'
    state.flag = RECV

def reset_precheck(state):
    assert state.flag == RESET, 'Call reset only once on initialization'
    state.flag = RECV
 
def aggregate_recvs(state, returns):
    obs, rewards, dones, infos = list(zip(*returns))

    for i, o in enumerate(obs):
        total_agents = state.num_agents * state.envs_per_worker
        state.preallocated_obs[i*total_agents:(i+1)*total_agents] = o

    rewards = list(chain.from_iterable(rewards))
    dones = list(chain.from_iterable(dones))
    orig = infos
    infos = [i for ii in infos for i in ii]

    assert len(rewards) == len(dones) == total_agents * state.num_workers
    assert len(infos) == state.num_workers * state.envs_per_worker
    return state.preallocated_obs, rewards, dones, infos

def split_actions(state, actions, env_id=None):
    if type(actions) == list:
        actions = np.array(actions)

    assert isinstance(actions, np.ndarray), 'Actions must be a numpy array'
    assert len(actions) == state.num_agents * state.num_workers * state.envs_per_worker

    return np.array_split(actions, state.num_workers)

def aggregate_profiles(profiles):
    return list(chain.from_iterable([profiles]))

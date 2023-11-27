from pdb import set_trace as T

import numpy as np
from itertools import chain

from pufferlib import namespace
from pufferlib.emulation import GymnasiumPufferEnv, PettingZooPufferEnv
from pufferlib.vectorization.multi_env import create_precheck
from pufferlib.vectorization.gym_multi_env import GymMultiEnv
from pufferlib.vectorization.pettingzoo_multi_env import PettingZooMultiEnv


RESET = 0
SEND = 1
RECV = 2

space_error_msg = 'env {env} must be an instance of GymnasiumPufferEnv or PettingZooPufferEnv'


def setup(env_creator, env_args, env_kwargs, num_envs, envs_per_worker, batch_size):
    assert num_envs % envs_per_worker == 0
    num_workers = num_envs // envs_per_worker

    if batch_size is None:
        batch_size = num_envs

    assert batch_size > 0
    assert batch_size % envs_per_worker == 0
    assert batch_size <= num_envs

    env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
    driver_env = env_creator(*env_args, **env_kwargs)

    if isinstance(driver_env, GymnasiumPufferEnv):
        multi_env_cls = GymMultiEnv
        num_agents = 1
    elif isinstance(driver_env, PettingZooPufferEnv):
        multi_env_cls = PettingZooMultiEnv
        num_agents = len(driver_env.possible_agents)
    else:
        raise TypeError(
            'env_creator must return an instance '
            'of GymnasiumPufferEnv or PettingZooPufferEnv'
        )

    obs_space = _single_observation_space(driver_env)
    return driver_env, multi_env_cls, num_agents, batch_size

def _single_observation_space(env):
    if isinstance(env, GymnasiumPufferEnv):
        return env.observation_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_observation_space
    else:
        raise TypeError(space_error_msg.format(env=env))

def single_observation_space(state):
    return _single_observation_space(state.driver_env)

def _single_action_space(env):
    if isinstance(env, GymnasiumPufferEnv):
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
    assert len(state.data) == state.batch_size // state.envs_per_worker
    assert state.flag == RECV, 'Call reset before stepping'
    state.flag = SEND

def send_precheck(state):
    assert state.flag == SEND, 'Call reset + recv before send'
    state.flag = RECV

def reset_precheck(state):
    assert state.flag == RESET, 'Call reset only once on initialization'
    state.flag = RECV
 
def aggregate_recvs(state, recvs):
    obs, rewards, dones, truncateds, infos, env_ids = list(zip(*recvs))
    agents_per_worker = state.num_agents * state.envs_per_worker
    batch_agents = state.batch_size * agents_per_worker

    obs = np.stack(list(chain.from_iterable(obs)), 0)
    rewards = list(chain.from_iterable(rewards))
    dones = list(chain.from_iterable(dones))
    truncateds = list(chain.from_iterable(truncateds))
    infos = [i for ii in infos for i in ii]

    # TODO: Masking will break for 1-agent PZ envs
    if state.num_agents == 1:
        mask = [e['mask'] for e in infos]
    else:
        mask = [e['mask'] for ee in infos for e in ee.values()]

    env_ids = np.concatenate([np.arange( # Per-agent env indexing
        i*agents_per_worker, (i+1)*agents_per_worker) for i in env_ids])

    assert all(len(e) == batch_agents for e in
        [obs, rewards, dones, truncateds, env_ids, mask])
    assert len(infos) == state.batch_size * state.envs_per_worker
    return obs, rewards, dones, truncateds, infos, env_ids, mask

def split_actions(state, actions, env_id=None):
    if type(actions) == list:
        actions = np.array(actions)

    assert isinstance(actions, np.ndarray), 'Actions must be a numpy array'
    assert len(actions) == state.batch_size * state.num_agents * state.envs_per_worker

    return np.array_split(actions, state.batch_size)

def aggregate_profiles(profiles):
    return list(chain.from_iterable([profiles]))

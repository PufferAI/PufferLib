from pdb import set_trace as T

import numpy as np
import warnings

import gymnasium
import inspect

import pufferlib
import pufferlib.spaces
from pufferlib import utils, exceptions
from pufferlib.environment import set_buffers
from pufferlib.spaces import Discrete, Tuple, Dict
import pufferlib.environment


def emulate(struct, sample):
    if isinstance(sample, dict):
        for k, v in sample.items():
            emulate(struct[k], v)
    elif isinstance(sample, tuple):
        for i, v in enumerate(sample):
            emulate(struct[f'f{i}'], v)
    else:
        struct[()] = sample

def make_buffer(arr_dtype, struct_dtype, struct, n=None):
    '''None instead of 1 makes it work for 1 agent PZ envs'''
    '''
    if n is None:
        struct = np.zeros(1, dtype=struct_dtype)
    else:
        struct = np.zeros(n, dtype=struct_dtype)
    '''

    arr = struct.view(arr_dtype)

    if n is None:
        arr = arr.ravel()
    else:
        arr = arr.reshape(n, -1)

    return arr

def _nativize(struct, space):
    if isinstance(space, Discrete):
        return struct.item()
    elif isinstance(space, Tuple):
        return tuple(_nativize(struct[f'f{i}'], elem)
            for i, elem in enumerate(space))
    elif isinstance(space, Dict):
        return {k: _nativize(struct[k], value)
            for k, value in space.items()}
    else:
        return struct

def nativize(arr, space, struct_dtype):
    struct = np.asarray(arr).view(struct_dtype)[0]
    return _nativize(struct, space)

'''
try:
    from pufferlib.extensions import emulate, nativize
except ImportError:
    warnings.warn('PufferLib Cython extensions not installed. Using slow Python versions')
'''

def dtype_from_space(space):
    if isinstance(space, pufferlib.spaces.Tuple):
        dtype = []
        for i, elem in enumerate(space):
            dtype.append((f'f{i}', dtype_from_space(elem)))
    elif isinstance(space, pufferlib.spaces.Dict):
        dtype = []
        for k, value in space.items():
            dtype.append((k, dtype_from_space(value)))
    elif isinstance(space, (pufferlib.spaces.Discrete)):
        dtype = (np.int32, ())
    elif isinstance(space, (pufferlib.spaces.MultiDiscrete)):
        dtype = (np.int32, (len(space.nvec),))
    else:
        dtype = (space.dtype, space.shape)

    return np.dtype(dtype, align=True)

def flatten_space(space):
    if isinstance(space, pufferlib.spaces.Tuple):
        subspaces = []
        for e in space:
            subspaces.extend(flatten_space(e))
        return subspaces
    elif isinstance(space, pufferlib.spaces.Dict):
        subspaces = []
        for e in space.values():
            subspaces.extend(flatten_space(e))
        return subspaces
    else:
        return [space]

def emulate_observation_space(space):
    emulated_dtype = dtype_from_space(space)

    if isinstance(space, pufferlib.spaces.Box):
        return space, emulated_dtype

    leaves = flatten_space(space)
    dtypes = [e.dtype for e in leaves]
    if dtypes.count(dtypes[0]) == len(dtypes):
        dtype = dtypes[0]
    else:
        dtype = np.dtype(np.uint8)

    mmin, mmax = utils._get_dtype_bounds(dtype)
    numel = emulated_dtype.itemsize // dtype.itemsize
    emulated_space = gymnasium.spaces.Box(low=mmin, high=mmax, shape=(numel,), dtype=dtype)
    return emulated_space, emulated_dtype

def emulate_action_space(space):
    if isinstance(space, pufferlib.spaces.Box):
        return space, space.dtype
    elif isinstance(space, (pufferlib.spaces.Discrete, pufferlib.spaces.MultiDiscrete)):
        return space, np.int32

    emulated_dtype = dtype_from_space(space)
    leaves = flatten_space(space)
    emulated_space = gymnasium.spaces.MultiDiscrete([e.n for e in leaves])
    return emulated_space, emulated_dtype


class GymnasiumPufferEnv(gymnasium.Env):
    def __init__(self, env=None, env_creator=None, env_args=[], env_kwargs={}, buf=None):
        self.env = make_object(env, env_creator, env_args, env_kwargs)

        self.initialized = False
        self.done = True

        self.is_observation_checked = False
        self.is_action_checked = False

        self.observation_space, self.obs_dtype = emulate_observation_space(
            self.env.observation_space)
        self.action_space, self.atn_dtype = emulate_action_space(
            self.env.action_space)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1

        self.is_obs_emulated = self.single_observation_space is not self.env.observation_space
        self.is_atn_emulated = self.single_action_space is not self.env.action_space
        self.emulated = pufferlib.namespace(
            observation_dtype = self.observation_space.dtype,
            emulated_observation_dtype = self.obs_dtype,
        )

        self.render_modes = 'human rgb_array'.split()

        set_buffers(self, buf)
        if isinstance(self.env.observation_space, pufferlib.spaces.Box):
            self.obs_struct = self.observations
        else:
            self.obs_struct = self.observations.view(self.obs_dtype)
 
    @property
    def render_mode(self):
        return self.env.render_mode

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self, seed=None):
        self.initialized = True
        self.done = False

        ob, info = _seed_and_reset(self.env, seed)
        if not self.is_observation_checked:
            self.is_observation_checked = check_space(
                ob, self.env.observation_space)

        if self.is_obs_emulated:
            emulate(self.obs_struct, ob)
        else:
            self.observations[:] = ob

        self.rewards[0] = 0
        self.terminals[0] = False
        self.truncations[0] = False
        self.masks[0] = True
 
        return self.observations, info
 
    def step(self, action):
        '''Execute an action and return (observation, reward, done, info)'''
        if not self.initialized:
            raise exceptions.APIUsageError('step() called before reset()')
        if self.done:
            raise exceptions.APIUsageError('step() called after environment is done')

        # Unpack actions from multidiscrete into the original action space
        if self.is_atn_emulated:
            action = nativize(action, self.env.action_space, self.atn_dtype)
        elif isinstance(action, np.ndarray):
            action = action.ravel()
            # TODO: profile or speed up
            if isinstance(self.action_space, pufferlib.spaces.Discrete):
                action = action[0]

        if not self.is_action_checked:
            self.is_action_checked = check_space(
                action, self.env.action_space)

        ob, reward, done, truncated, info = self.env.step(action)

        if self.is_obs_emulated:
            emulate(self.obs_struct, ob)
        else:
            self.observations[:] = ob

        self.rewards[0] = reward
        self.terminals[0] = done
        self.truncations[0] = truncated
        self.masks[0] = True
                  
        self.done = done or truncated
        return self.observations, reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

class PettingZooPufferEnv:
    def __init__(self, env=None, env_creator=None, env_args=[], buf=None, env_kwargs={}, to_puffer=False):
        self.env = make_object(env, env_creator, env_args, env_kwargs)
        self.to_puffer = to_puffer
        self.initialized = False
        self.all_done = True

        self.is_observation_checked = False
        self.is_action_checked = False

        # Compute the observation and action spaces
        single_agent = self.possible_agents[0]
        self.env_single_observation_space = self.env.observation_space(single_agent)
        self.env_single_action_space = self.env.action_space(single_agent)
        self.single_observation_space, self.obs_dtype = (
            emulate_observation_space(self.env_single_observation_space))
        self.single_action_space, self.atn_dtype = (
            emulate_action_space(self.env_single_action_space))
        self.is_obs_emulated = self.single_observation_space is not self.env_single_observation_space
        self.is_atn_emulated = self.single_action_space is not self.env_single_action_space
        self.emulated = pufferlib.namespace(
            observation_dtype = self.single_observation_space.dtype,
            emulated_observation_dtype = self.obs_dtype,
        )

        self.num_agents = len(self.possible_agents)

        set_buffers(self, buf)
        if isinstance(self.env_single_observation_space, pufferlib.spaces.Box):
            self.obs_struct = self.observations
        else:
            self.obs_struct = self.observations.view(self.obs_dtype)

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def done(self):
        return len(self.agents) == 0 or self.all_done

    def observation_space(self, agent):
        '''Returns the observation space for a single agent'''
        if agent not in self.possible_agents:
            raise pufferlib.exceptions.InvalidAgentError(agent, self.possible_agents)

        return self.single_observation_space

    def action_space(self, agent):
        '''Returns the action space for a single agent'''
        if agent not in self.possible_agents:
            raise pufferlib.exceptions.InvalidAgentError(agent, self.possible_agents)

        return self.single_action_space

    def reset(self, seed=None):
        if not self.initialized:
            self.dict_obs = {agent: self.observations[i] for i, agent in enumerate(self.possible_agents)}

        self.initialized = True
        self.all_done = False
        self.mask = {k: False for k in self.possible_agents}

        obs, info = self.env.reset(seed=seed)

        if not self.is_observation_checked:
            for k, ob in obs.items():
                self.is_observation_checked = check_space(
                    ob, self.env.observation_space(k))

        # Call user featurizer and flatten the observations
        self.observations[:] = 0
        for i, agent in enumerate(self.possible_agents):
            if agent not in obs:
                continue

            ob = obs[agent]
            self.mask[agent] = True
            if self.is_obs_emulated:
                emulate(self.obs_struct[i], ob)
            else:
                self.observations[i] = ob

        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True
        return self.dict_obs, info

    def step(self, actions):
        '''Step the environment and return (observations, rewards, dones, infos)'''
        if not self.initialized:
            raise exceptions.APIUsageError('step() called before reset()')
        if self.done:
            raise exceptions.APIUsageError('step() called after environment is done')

        if isinstance(actions, np.ndarray):
            if not self.is_action_checked and len(actions) != self.num_agents:
                raise exceptions.APIUsageError(
                    f'Actions specified as len {len(actions)} but environment has {self.num_agents} agents')

            actions = {agent: actions[i] for i, agent in enumerate(self.possible_agents)}

        # Postprocess actions and validate action spaces
        if not self.is_action_checked:
            for agent in actions:
                if agent not in self.possible_agents:
                    raise exceptions.InvalidAgentError(agent, self.possible_agents)

            self.is_action_checked = check_space(
                next(iter(actions.values())),
                self.single_action_space
            )

        # Unpack actions from multidiscrete into the original action space
        unpacked_actions = {}
        for agent, atn in actions.items():
            if agent not in self.possible_agents:
                raise exceptions.InvalidAgentError(agent, self.agents)

            if agent not in self.agents:
                continue

            if self.is_atn_emulated:
                atn = nativize(atn, self.env_single_action_space, self.atn_dtype)

            unpacked_actions[agent] = atn

        obs, rewards, dones, truncateds, infos = self.env.step(unpacked_actions)
        # TODO: Can add this assert once NMMO Horizon is ported to puffer
        # assert all(dones.values()) == (len(self.env.agents) == 0)
        self.mask = {k: False for k in self.possible_agents}
        self.rewards[:] = 0
        self.terminals[:] = True
        self.truncations[:] = False
        for i, agent in enumerate(self.possible_agents):
            # TODO: negative padding buf
            if agent not in obs:
                self.observations[i] = 0
                self.rewards[i] = 0
                self.terminals[i] = True
                self.truncations[i] = False
                self.masks[i] = False
                continue

            ob = obs[agent] 
            self.mask[agent] = True
            if self.is_obs_emulated:
                emulate(self.obs_struct[i], ob)
            else:
                self.observations[i] = ob

            self.rewards[i] = rewards[agent]
            self.terminals[i] = dones[agent]
            self.truncations[i] = truncateds[agent]
            self.masks[i] = True
     
        self.all_done = all(dones.values()) or all(truncateds.values())
        rewards = pad_agent_data(rewards, self.possible_agents, 0)
        dones = pad_agent_data(dones, self.possible_agents, True) # You changed this from false to match api test... is this correct?
        truncateds = pad_agent_data(truncateds, self.possible_agents, False)
        return self.dict_obs, rewards, dones, truncateds, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def pad_agent_data(data, agents, pad_value):
    return {agent: data[agent] if agent in data else pad_value
        for agent in agents}
 
def make_object(object_instance=None, object_creator=None, creator_args=[], creator_kwargs={}):
    if (object_instance is None) == (object_creator is None):
        raise ValueError('Exactly one of object_instance or object_creator must be provided')

    if object_instance is not None:
        if callable(object_instance) or inspect.isclass(object_instance):
            raise TypeError('object_instance must be an instance, not a function or class')
        return object_instance

    if object_creator is not None:
        if not callable(object_creator):
            raise TypeError('object_creator must be a callable')
        
        if creator_args is None:
            creator_args = []

        if creator_kwargs is None:
            creator_kwargs = {}

        return object_creator(*creator_args, **creator_kwargs)

def check_space(data, space):
    try:
        contains = space.contains(data)
    except:
        raise exceptions.APIUsageError(
            f'Error checking space {space} with sample :\n{data}')

    if not contains:
        raise exceptions.APIUsageError(
            f'Data:\n{data}\n not in space:\n{space}')
    
    return True

def _seed_and_reset(env, seed):
    if seed is None:
        # Gym bug: does not reset env correctly
        # when seed is passed as explicit None
        return env.reset()

    try:
        obs, info = env.reset(seed=seed)
    except:
        try:
            env.seed(seed)
            obs, info = env.reset()
        except:
            obs, info = env.reset()
            warnings.warn('WARNING: Environment does not support seeding.', DeprecationWarning)

    return obs, info

class GymnaxPufferEnv(pufferlib.environment.PufferEnv):
    def __init__(self, env, env_params, num_envs=1, buf=None):
        from gymnax.environments.spaces import gymnax_space_to_gym_space

        gymnax_obs_space = env.observation_space(env_params)
        self.single_observation_space = gymnax_space_to_gym_space(gymnax_obs_space)

        gymnax_act_space = env.action_space(env_params)
        self.single_action_space = gymnax_space_to_gym_space(gymnax_act_space)

        self.num_agents = num_envs

        super().__init__(buf)
        self.env_params = env_params
        self.env = env

        import jax
        self.reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
        self.step_fn = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
        self.rng = jax.random.PRNGKey(0)

    def reset(self, rng, params=None):
        import jax
        self.rng, _rng = jax.random.split(self.rng)
        self.rngs = jax.random.split(_rng, self.num_agents)
        obs, self.state = self.reset_fn(self.rngs, params)
        from torch.utils import dlpack as torch_dlpack
        self.observations = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(obs))
        return self.observations, []

    def step(self, action):
        import jax
        #self.rng, _rng = jax.random.split(self.rng)
        #rngs = jax.random.split(_rng, self.num_agents)
        obs, self.state, reward, done, info = self.step_fn(self.rngs, self.state, action, self.env_params)

        # Convert JAX array to DLPack, then to PyTorch tensor
        from torch.utils import dlpack as torch_dlpack
        self.observations = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(obs))
        self.rewards = np.asarray(reward)
        self.terminals = np.asarray(done)
        infos = [{k: v.mean().item() for k, v in info.items()}]
        return self.observations, self.rewards, self.terminals, self.terminals, infos

import os
import sys
import warnings

from contextlib import redirect_stdout, redirect_stderr, contextmanager
from types import SimpleNamespace
from collections.abc import Mapping
from io import StringIO
from functools import wraps

import numpy as np
import gymnasium

import pufferlib.spaces

ENV_ERROR = '''
Environment missing required attribute {}. The most common cause is
calling super() before you have assigned the attribute.
'''


def set_buffers(env, buf=None):
    if buf is None:
        obs_space = env.single_observation_space
        env.observations = np.zeros((env.num_agents, *obs_space.shape), dtype=obs_space.dtype)
        env.rewards = np.zeros(env.num_agents, dtype=np.float32)
        env.terminals = np.zeros(env.num_agents, dtype=bool)
        env.truncations = np.zeros(env.num_agents, dtype=bool)
        env.masks = np.ones(env.num_agents, dtype=bool)

        # TODO: Major kerfuffle on inferring action space dtype. This needs some asserts?
        atn_space = pufferlib.spaces.joint_space(env.single_action_space, env.num_agents)
        if isinstance(env.single_action_space, pufferlib.spaces.Box):
            env.actions = np.zeros(atn_space.shape, dtype=atn_space.dtype)
        else:
            env.actions = np.zeros(atn_space.shape, dtype=np.int32)
    else:
        env.observations = buf['observations']
        env.rewards = buf['rewards']
        env.terminals = buf['terminals']
        env.truncations = buf['truncations']
        env.masks = buf['masks']
        env.actions = buf['actions']

class PufferEnv:
    def __init__(self, buf=None):
        if not hasattr(self, 'single_observation_space'):
            raise APIUsageError(ENV_ERROR.format('single_observation_space'))
        if not hasattr(self, 'single_action_space'):
            raise APIUsageError(ENV_ERROR.format('single_action_space'))
        if not hasattr(self, 'num_agents'):
            raise APIUsageError(ENV_ERROR.format('num_agents'))
        if self.num_agents < 1:
            raise APIUsageError('num_agents must be >= 1')

        if hasattr(self, 'observation_space'):
            raise APIUsageError('PufferEnvs must define single_observation_space, not observation_space')
        if hasattr(self, 'action_space'):
            raise APIUsageError('PufferEnvs must define single_action_space, not action_space')
        if not isinstance(self.single_observation_space, pufferlib.spaces.Box):
            raise APIUsageError('Native observation_space must be a Box')
        if (not isinstance(self.single_action_space, pufferlib.spaces.Discrete)
                and not isinstance(self.single_action_space, pufferlib.spaces.MultiDiscrete)
                and not isinstance(self.single_action_space, pufferlib.spaces.Box)):
            raise APIUsageError('Native action_space must be a Discrete, MultiDiscrete, or Box')

        set_buffers(self, buf)

        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.observation_space = pufferlib.spaces.joint_space(self.single_observation_space, self.num_agents)
        self.agent_ids = np.arange(self.num_agents)

    @property
    def agent_per_batch(self):
        return self.num_agents

    @property
    def agents_per_batch(self):
        return self.num_agents

    @property
    def emulated(self):
        '''Native envs do not use emulation'''
        return False

    @property
    def done(self):
        '''Native envs handle resets internally'''
        return False

    @property
    def driver_env(self):
        '''For compatibility with Multiprocessing'''
        return self

    def reset(self, seed=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def async_reset(self, seed=None):
        _, self.infos = self.reset(seed)
        assert isinstance(self.infos, list), 'PufferEnvs must return info as a list of dicts'

    def send(self, actions):
        _, _, _, _, self.infos = self.step(actions)
        assert isinstance(self.infos, list), 'PufferEnvs must return info as a list of dicts'

    def recv(self):
        return (self.observations, self.rewards, self.terminals,
            self.truncations, self.infos, self.agent_ids, self.masks)

### Postprocessing
class ResizeObservation(gymnasium.Wrapper):
    '''Fixed downscaling wrapper. Do NOT use gym.wrappers.ResizeObservation
    It uses a laughably slow OpenCV resize. -50% on Atari just from that.'''
    def __init__(self, env, downscale=2):
        super().__init__(env)
        self.downscale = downscale
        y_size, x_size = env.observation_space.shape
        assert y_size % downscale == 0 and x_size % downscale == 0
        y_size = env.observation_space.shape[0] // downscale
        x_size = env.observation_space.shape[1] // downscale
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(y_size, x_size), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[::self.downscale, ::self.downscale], info

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        return obs[::self.downscale, ::self.downscale], reward, terminal, truncated, info

class ClipAction(gymnasium.Wrapper):
    '''Wrapper for Gymnasium environments that clips actions'''
    def __init__(self, env):
        self.env = env
        assert isinstance(env.action_space, gymnasium.spaces.Box)
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        dtype_info = np.finfo(env.action_space.dtype)
        self.action_space = gymnasium.spaces.Box(
            low=dtype_info.min,
            high=dtype_info.max,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def step(self, action):
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return self.env.step(action)


class EpisodeStats(gymnasium.Wrapper):
    '''Wrapper for Gymnasium environments that stores
    episodic returns and lengths in infos'''
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset()

    def reset(self, seed=None, options=None):
        self.info = dict(episode_return=[], episode_length=0)
        # TODO: options
        return self.env.reset(seed=seed)#, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        for k, v in unroll_nested_dict(info):
            if k not in self.info:
                self.info[k] = []

            self.info[k].append(v)

        self.info['episode_return'].append(reward)
        self.info['episode_length'] += 1

        info = {}
        if terminated or truncated:
            for k, v in self.info.items():
                try:
                    info[k] = sum(v)
                    continue
                except TypeError:
                    pass

                if isinstance(v, str):
                    info[k] = v
                    continue

                try:
                    x = int(v) # probably a value
                    info[k] = v
                    continue
                except TypeError:
                    pass

        return observation, reward, terminated, truncated, info

class PettingZooWrapper:
    '''PettingZoo does not provide a ParallelEnv wrapper. This code is adapted from
    their AEC wrapper, to prevent unneeded conversions to/from AEC'''
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        '''Returns an attribute with ``name``, unless ``name`` starts with an underscore.'''
        if name.startswith('_') and name != '_cumulative_rewards':
            raise AttributeError(f'accessing private attribute "{name}" is prohibited')
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def reset(self, seed=None, options=None):
        try:
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            return self.env.reset(seed=seed)

    def observe(self, agent):
        return self.env.observe(agent)

    def state(self):
        return self.env.state()

    def step(self, action):
        return self.env.step(action)

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def __str__(self) -> str:
        '''Returns a name which looks like: "max_observation<space_invaders_v1>".'''
        return f'{type(self).__name__}<{str(self.env)}>'

class MeanOverAgents(PettingZooWrapper):
    '''Averages over agent infos'''
    def _mean(self, infos):
        list_infos = {}
        for agent, info in infos.items():
            for k, v in info.items():
                if k not in list_infos:
                    list_infos[k] = []

                list_infos[k].append(v)

        mean_infos = {}
        for k, v in list_infos.items():
            try:
                mean_infos[k] = np.mean(v)
            except:
                pass

        return mean_infos

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed, options)
        infos = self._mean(infos)
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        infos = self._mean(infos)
        return observations, rewards, terminations, truncations, infos

class MultiagentEpisodeStats(PettingZooWrapper):
    '''Wrapper for PettingZoo environments that stores
    episodic returns and lengths in infos'''
    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        self.infos = {
            agent: dict(episode_return=[], episode_length=0)
            for agent in self.possible_agents
        }
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)

        all_infos = {}
        for agent in infos:
            agent_info = self.infos[agent]
            for k, v in unroll_nested_dict(infos[agent]):
                if k not in agent_info:
                    agent_info[k] = []

                agent_info[k].append(v)

            # Saved to self. TODO: Clean up
            agent_info['episode_return'].append(rewards[agent])
            agent_info['episode_length'] += 1

            agent_info = {}
            all_infos[agent] = agent_info
            if terminations[agent] or truncations[agent]:
                for k, v in self.infos[agent].items():
                    try:
                        agent_info[k] = sum(v)
                        continue
                    except TypeError:
                        pass

                    if isinstance(v, str):
                        agent_info[k] = v
                        continue

                    try:
                        x = int(v) # probably a value
                        agent_info[k] = v
                        continue
                    except TypeError:
                        pass

        return observations, rewards, terminations, truncations, all_infos
### Exceptions
class EnvironmentSetupError(RuntimeError):
    def __init__(self, e, package):
        super().__init__(self.message)

class APIUsageError(RuntimeError):
    """Exception raised when the API is used incorrectly."""

    def __init__(self, message="API usage error."):
        self.message = message
        super().__init__(self.message)

class InvalidAgentError(ValueError):
    """Exception raised when an invalid agent key is used."""

    def __init__(self, agent_id, agents):
        message = (
            f'Invalid agent/team ({agent_id}) specified. '
            f'Valid values:\n{agents}'
        )
        super().__init__(message)

class GymToGymnasium:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render = env.render
        self.metadata = env.metadata

    def reset(self, seed=None, options=None):
        if seed is not None:
            ob = self.env.reset(seed=seed)
        else:
            ob = self.env.reset()
        return ob, {}

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, False, info

    def close(self):
        self.env.close()

### Wrappers
class PettingZooTruncatedWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render = env.render

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def agents(self):
        return self.env.agents

    def reset(self, seed=None):
        if seed is not None:
            ob, info = self.env.reset(seed=seed)
        else:
            ob, info = self.env.reset()
        info = {k: {} for k in ob}
        return ob, info

    def step(self, actions):
        observations, rewards, terminals, truncations, infos = self.env.step(actions)
        return observations, rewards, terminals, truncations, infos

    def close(self):
        self.env.close()

### Misc
def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v

def silence_warnings(original_func, category=DeprecationWarning):
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=category)
            return original_func(*args, **kwargs)
    return wrapper

class Suppress():
    def __init__(self):
        self.f = StringIO()
        self.null_1 = os.open(os.devnull, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
        self.null_2 = os.open(os.devnull, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)

    def __enter__(self):
        # Suppress C library outputs
        self.orig_stdout = os.dup(1)
        self.orig_stderr = os.dup(2)
        os.dup2(self.null_1, 1)
        os.dup2(self.null_2, 2)

        # Suppress Python outputs
        self._stdout_redirector = redirect_stdout(self.f)
        self._stderr_redirector = redirect_stderr(self.f)
        self._stdout_redirector.__enter__()
        self._stderr_redirector.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Enable C library outputs
        os.dup2(self.orig_stdout, 1)
        os.dup2(self.orig_stderr, 2)
        os.close(self.orig_stdout)
        os.close(self.orig_stderr)
        os.close(self.null_1)
        os.close(self.null_2)

        # Enable Python outputs
        self._stdout_redirector.__exit__(exc_type, exc_val, exc_tb)
        self._stderr_redirector.__exit__(exc_type, exc_val, exc_tb)

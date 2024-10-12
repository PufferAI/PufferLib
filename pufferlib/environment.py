import numpy as np

from pufferlib.exceptions import APIUsageError
import pufferlib.spaces

ERROR = '''
Environment missing required attribute {}. The most common cause is
calling super() before you have assigned the attribute.
'''

class PufferEnv:
    def __init__(self, buf=None):
        if not hasattr(self, 'single_observation_space'):
            raise APIUsageError(ERROR.format('single_observation_space'))
        if not hasattr(self, 'single_action_space'):
            raise APIUsageError(ERROR.format('single_action_space'))
        if not hasattr(self, 'num_agents'):
            raise APIUsageError(ERROR.format('num_agents'))

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

        if buf is None:
            obs_space = self.single_observation_space
            self.observations = np.zeros((self.num_agents, *obs_space.shape), dtype=obs_space.dtype)
            self.rewards = np.zeros(self.num_agents, dtype=np.float32)
            self.terminals = np.zeros(self.num_agents, dtype=bool)
            self.truncations = np.zeros(self.num_agents, dtype=bool)
            self.masks = np.ones(self.num_agents, dtype=bool)

            # TODO: Major kerfuffle on inferring action space dtype. This needs some asserts?
            atn_space = self.single_action_space
            if isinstance(self.single_action_space, pufferlib.spaces.Box):
                self.actions = np.zeros((self.num_agents, *atn_space.shape), dtype=atn_space.dtype)
            else:
                self.actions = np.zeros((self.num_agents, *atn_space.shape), dtype=np.int32)
        else:
            self.observations = buf.observations
            self.rewards = buf.rewards
            self.terminals = buf.terminals
            self.truncations = buf.truncations
            self.masks = buf.masks
            self.actions = buf.actions

        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.observation_space = pufferlib.spaces.joint_space(self.single_observation_space, self.num_agents)
        self.agent_ids = np.arange(self.num_agents)

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

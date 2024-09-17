from pdb import set_trace as T
import numpy as np
import gymnasium

import pufferlib.utils

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

        for k, v in pufferlib.utils.unroll_nested_dict(info):
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
            for k, v in pufferlib.utils.unroll_nested_dict(infos[agent]):
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

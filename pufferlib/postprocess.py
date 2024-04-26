import gymnasium


class EpisodeStats(gymnasium.Wrapper):
    '''Wrapper for Gymnasium environments that stores
    episodic returns and lengths in infos'''
    def __init__(self, env):
        self.env = env
        self.reset()

    # TODO: Fix options. Maybe reimplement gymnasium.Wrapper with better compatibility
    def reset(self, seed=None):
        self.episode_return = 0
        self.episode_length = 0
        return self.env.reset(seed=seed)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.episode_return += reward
        self.episode_length += 1

        if terminated or truncated:
            info['episode_return'] = self.episode_return
            info['episode_length'] = self.episode_length

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

class MultiagentEpisodeStats(PettingZooWrapper):
    '''Wrapper for PettingZoo environments that stores
    episodic returns and lengths in infos'''
    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        self.episode_returns = {agent: 0 for agent in self.agents}
        self.episode_lengths = {agent: 0 for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)

        for agent in self.agents:
            self.episode_returns[agent] += rewards[agent]
            self.episode_lengths[agent] += 1

            if terminations[agent] or truncations[agent]:
                infos[agent]['episode_return'] = self.episode_returns[agent]
                infos[agent]['episode_length'] = self.episode_lengths[agent]

        return observations, rewards, terminations, truncations, infos

from pdb import set_trace as T

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
            ob = self.env.reset(seed=seed)
        else:
            ob = self.env.reset()
        info = {k: {} for k in ob}
        return ob, info

    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        truncated = {k: False for k in observations}
        return observations, rewards, dones, truncated, infos

    def close(self):
        self.env.close()

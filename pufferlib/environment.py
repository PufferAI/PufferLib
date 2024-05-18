class PufferEnv:
    #@property
    #def num_agents(self):
    #    raise NotImplementedError

    #@property
    #def observation_space(self):
    #    raise NotImplementedError

    #@property
    #def action_space(self):
    #    raise NotImplementedError

    def reset(self, seed=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def close(self):
        pass


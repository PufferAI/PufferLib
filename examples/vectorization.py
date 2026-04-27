import gymnasium
import pufferlib.emulation
import pufferlib.vector

class SamplePufferEnv(pufferlib.PufferEnv):
    def __init__(self, foo=0, bar=1, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.num_agents = 2
        super().__init__(buf)

        # Sample args and kwargs
        self.foo = foo
        self.bar = bar

    def reset(self, seed=0):
        self.observations[:] = self.observation_space.sample()
        return self.observations, []

    def step(self, action):
        self.observations[:] = self.observation_space.sample()
        infos = [{'infos': 'is a list of dictionaries'}]
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        pass

if __name__ == '__main__':
    serial_vecenv = pufferlib.vector.make(
        SamplePufferEnv, num_envs=2, backend=pufferlib.vector.Serial)
    observations, infos = serial_vecenv.reset()
    actions = serial_vecenv.action_space.sample()
    o, r, d, t, i = serial_vecenv.step(actions)
    print('Serial VecEnv:')
    print('Observations:', o)
    print('Rewards:', r)
    print('Terminals:', t)
    print('Truncations:', d)

    # Pass arguments to all environments like this
    serial_vecenv = pufferlib.vector.make(
        SamplePufferEnv, num_envs=2, backend=pufferlib.vector.Serial,
        env_args=[3], env_kwargs={'bar': 4}
    )
    print('Foo: ', [env.foo for env in serial_vecenv.envs])
    print('Bar: ', [env.bar for env in serial_vecenv.envs])

    # Or to each environment like this
    serial_vecenv = pufferlib.vector.make(
        [SamplePufferEnv, SamplePufferEnv], num_envs=2, backend=pufferlib.vector.Serial,
        env_args=[[3], [4]], env_kwargs=[{'bar': 4}, {'bar': 5}]
    )
    print('Foo: ', [env.foo for env in serial_vecenv.envs])
    print('Bar: ', [env.bar for env in serial_vecenv.envs])

    vecenv = pufferlib.vector.make(SamplePufferEnv,
        num_envs=2, num_workers=2, batch_size=1, backend=pufferlib.vector.Multiprocessing)
    vecenv.async_reset() # You can also use the synchronous API with Multiprocessing
    o, r, d, t, i, env_ids, masks = vecenv.recv()
    actions = vecenv.action_space.sample()
    print('Policy computes actions for all agents in batch_size=1 of the total num_envs=2 environments')
    print('Actions:', actions)
    vecenv.send(actions)

    # New observations are ready while the other envs are running in the background
    o, r, d, t, i, env_ids, masks = vecenv.recv()
    print('Observations:', o)

    # Make sure to close the vecenv when you're done
    vecenv.close()

    try:
        vecenv = pufferlib.vector.make(SamplePufferEnv,
            num_envs=1, num_workers=2, batch_size=3, backend=pufferlib.vector.Multiprocessing)
    except (AssertionError, ValueError):
        #Make sure num_envs divides num_workers, and both num_envs and num_workers should divide batch_size
        pass
    

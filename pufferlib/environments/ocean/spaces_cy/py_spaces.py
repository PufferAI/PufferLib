import numpy as np
import gymnasium
import pufferlib
from .cy_spaces_cy import CSpacesCy

class SpacesCyEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs=1):
        super().__init__()

        self.num_envs = num_envs
        
        self.buf = pufferlib.namespace(
            image_observations=np.zeros((self.num_envs, 5, 5), dtype=np.float32),
            flat_observations=np.zeros((self.num_envs, 5), dtype=np.int8),
            actions=np.zeros((self.num_envs, 2), dtype=np.uint32),
            rewards=np.zeros((self.num_envs, 1), dtype=np.float32),
            dones=np.zeros((self.num_envs, 1), dtype=np.uint8),
            scores=np.zeros((self.num_envs, 1), dtype=np.int32)
        )

        # Create the observation and action spaces
        self.observation_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.float32),
            'flat': gymnasium.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.int8),
        })
        self.action_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Discrete(2),
            'flat': gymnasium.spaces.Discrete(2),
        })

        self.c_envs = [CSpacesCy(self.buf.image_observations[i:i+1], 
                                self.buf.flat_observations[i:i+1],
                                self.buf.actions[i:i+1],
                                self.buf.rewards[i:i+1],
                                self.buf.dones[i:i+1],
                                self.buf.scores[i:i+1],
                                num_agents=1)
                       for i in range(self.num_envs)]

    def reset(self, seed=None):
        for env in self.c_envs:
            env.reset()

        observations = {
            'image': self.buf.image_observations,
            'flat': self.buf.flat_observations
        }

        return observations, {}

    def step(self, actions):
        if isinstance(actions, dict):
            self.buf.actions[0][0] = actions['image']
            self.buf.actions[0][1] = actions['flat']
            self.c_envs[0].step()

        terminated = self.buf.dones.copy()
        truncated = np.zeros_like(terminated)
        
        return (
            {'image': self.buf.image_observations, 
             'flat': self.buf.flat_observations},
            self.buf.rewards,
            terminated,
            truncated,
            {}
        )

def make_spaces_cy(num_envs=1):
    return SpacesCyEnv(num_envs=num_envs)


def test_performance(num_envs=1, timeout=10, atn_cache=1024):
    import time
    env = make_spaces_cy(num_envs=num_envs)

    env.reset()
    
    tick = 0
    actions_image = np.random.randint(0, 2, (atn_cache, num_envs))
    actions_flat = np.random.randint(0, 2, (atn_cache, num_envs))

    start = time.time()

    while time.time() - start < timeout:
        atn_image = actions_image[tick % atn_cache]
        atn_flat = actions_flat[tick % atn_cache]
        
        actions = {'image': atn_image, 'flat': atn_flat}
        env.step(actions)
        tick += 1

    elapsed_time = time.time() - start
    sps = num_envs * tick / elapsed_time
    print(f"SPS: {sps:.2f}")

if __name__ == '__main__':
    test_performance()
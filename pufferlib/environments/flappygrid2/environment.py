import functools
import numpy as np
import pufferlib
from gymnasium import spaces


class FlappyGrid2(pufferlib.PufferEnv):
    """PufferLib-native FlappyBird/Grid hybrid."""
    
    def __init__(self):
        self.grid_size = 10
        self.bird_pos = 5
        self.obstacle_x = 9
        self.obstacle_gap_y = 5
        self.score = 0
        
        # PufferLib required attributes (set BEFORE super().__init__())
        self.num_agents = 1
        self.single_observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(3,), dtype=np.int32
        )
        self.single_action_space = spaces.Discrete(2)
        
        super().__init__()
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.bird_pos = 5
        self.obstacle_x = 9
        self.obstacle_gap_y = np.random.randint(2, 8)
        self.score = 0
        
        obs = np.array([self.bird_pos, self.obstacle_x, self.obstacle_gap_y], dtype=np.int32)
        return obs, {}
    
    def step(self, action):
        if action == 1:
            self.bird_pos = max(0, self.bird_pos - 1)
        else:
            self.bird_pos = min(self.grid_size - 1, self.bird_pos + 1)
        
        self.obstacle_x -= 1
        done = False
        reward = 0.1
        
        if self.obstacle_x == 0:
            if abs(self.bird_pos - self.obstacle_gap_y) <= 1:
                reward = 1.0
                self.score += 1
            else:
                done = True
                reward = -1.0
            
            self.obstacle_x = 9
            self.obstacle_gap_y = np.random.randint(2, 8)
        
        obs = np.array([self.bird_pos, self.obstacle_x, self.obstacle_gap_y], dtype=np.int32)
        return obs, reward, done, False, {'score': self.score}
    
    def close(self):
        pass


def make(name='flappygrid2'):
    return FlappyGrid2()


def env_creator(name='flappygrid2'):
    return functools.partial(make, name)


if __name__ == "__main__":
    import time
    
    print("FlappyGrid2 Performance Test")
    print("=" * 50)
    
    factory = env_creator()
    env = factory()
    
    obs, info = env.reset()
    start = time.time()
    
    steps = 100_000
    episodes = 0
    
    for i in range(steps):
        action = env.single_action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            obs, info = env.reset()
            episodes += 1
    
    elapsed = time.time() - start
    fps = steps / elapsed
    
    print(f"Steps:    {steps:,}")
    print(f"Episodes: {episodes:,}")
    print(f"Time:     {elapsed:.2f}s")
    print(f"FPS:      {fps:,.0f}")
    print("✅ Test complete")
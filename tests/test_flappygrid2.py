import pytest
from pufferlib.environments import flappygrid2


def test_flappygrid2_basic():
    """Test basic env functionality."""
    factory = flappygrid2.env_creator()
    env = factory()
    
    # Test reset
    obs, info = env.reset()
    assert obs.shape == (3,), f"Expected obs shape (3,), got {obs.shape}"
    
    # Test step
    for _ in range(100):
        action = env.single_action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert obs.shape == (3,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        
        if done:
            obs, info = env.reset()
    
    env.close()
    print("✅ FlappyGrid2 basic test passed")


def test_flappygrid2_episodes():
    """Test multiple episodes complete correctly."""
    factory = flappygrid2.env_creator()
    env = factory()
    
    obs, info = env.reset()
    episodes_completed = 0
    
    for _ in range(10000):
        action = env.single_action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            episodes_completed += 1
            obs, info = env.reset()
    
    assert episodes_completed > 0, "No episodes completed"
    print(f"✅ Completed {episodes_completed} episodes")
    env.close()


if __name__ == "__main__":
    test_flappygrid2_basic()
    test_flappygrid2_episodes()
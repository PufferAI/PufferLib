"""Test script to prove Action0 env can be beaten."""
import numpy as np
from pufferlib.ocean.action0.action0 import Action0

# Create environment
env = Action0(num_envs=1, horizon=128)

# Test 1: Agent that always picks action 0 on step 1
print("=== Test 1: Always pick action 0 on step 1 ===")
wins = 0
episodes = 1000
for ep in range(episodes):
    obs, _ = env.reset()
    for step in range(128):
        if step == 0:
            action = np.array([0])  # Pick action 0 on first step
        else:
            action = np.array([1])  # Any action after
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal[0]:
            if reward[0] > 0:
                wins += 1
            break
print(f"Win rate: {wins}/{episodes} = {wins/episodes:.2%}")

# Test 2: Agent that picks randomly
print("\n=== Test 2: Random agent ===")
wins = 0
for ep in range(episodes):
    obs, _ = env.reset()
    for step in range(128):
        action = np.random.randint(0, 2, size=(1,))
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal[0]:
            if reward[0] > 0:
                wins += 1
            break
print(f"Win rate: {wins}/{episodes} = {wins/episodes:.2%}")
print(f"Expected random: 1/2 = 50.00%")

# Test 3: Agent that always picks wrong action
print("\n=== Test 3: Always pick action 1 on step 1 ===")
wins = 0
for ep in range(episodes):
    obs, _ = env.reset()
    for step in range(128):
        action = np.array([1])  # Always pick 1 (wrong)
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal[0]:
            if reward[0] > 0:
                wins += 1
            break
print(f"Win rate: {wins}/{episodes} = {wins/episodes:.2%}")

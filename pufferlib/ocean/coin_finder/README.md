# CoinFinder Environment

A simple, high-performance reinforcement learning environment. The main goal creating this environment was learning RL and the Pufferlib library.

## Overview

CoinFinder is a 2D grid-based environment where an agent must navigate to collect randomly placed coins.

### Environment Dynamics

- **Grid Size**: 10x10
- **Agent**: Single point agent that can move in 4 directions
- **Coins**: 5 coins randomly placed on the grid
- **Goal**: Collect coins by moving within collection radius (0.1 units)
- **Episode Length**: 100 steps maximum
- **Reward**: +1 for each coin collected, 0 otherwise

### Observation Space
12-dimensional vector

Structure:
  1. Agent position (2 values):
  2. 5 Coins positions (10 values [x,y]):

  Value ranges:
  - Agent position: [0.0, 0.9] (since positions are 0-9 divided by GRID_SIZE=10)
  - Uncollected coins: [0.0, 0.9]
  - Collected coins: -1.0

### Action Space

Discrete(4):
- 0: Move UP
- 1: Move DOWN
- 2: Move LEFT
- 3: Move RIGHT

## Configuration

Configuration file: `pufferlib/config/ocean/coin_finder.ini`

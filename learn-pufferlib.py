"""
LEARN_V2.PY - RL with PufferLib (The Right Way)
================================================

PURPOSE: Learn reinforcement learning using PufferLib's patterns and infrastructure.

This is the "full PufferLib" version of learn.py. Instead of implementing PPO
from scratch, we use PufferLib's pufferl.PuffeRL trainer which handles:
- Rollout collection
- GAE advantage computation
- PPO loss calculation
- Gradient updates
- Logging and metrics

HOW TO USE:
1. Read each section's comments (the WHY and WHAT)
2. Fill in the TODO sections
3. Run and test after each section: python learn_v2.py
4. Only move to next section when current one works

The environment is the same as learn.py:
- 2D arena where an agent must reach a target
- Agent can move UP/DOWN/LEFT/RIGHT or stay still
- Episode ends when: agent reaches target, hits wall, or 200 steps pass

DEPENDENCIES:
    pip install pufferlib torch numpy gymnasium
"""

import os
import numpy as np
import gymnasium
import torch
import torch.nn as nn
import pufferlib
import pufferlib.vector
import pufferlib.pytorch
from pufferlib import pufferl


# =============================================================================
# SECTION 1: PUFFERLIB ENVIRONMENT
# =============================================================================
"""
WHY inherit from pufferlib.PufferEnv?
-------------------------------------
PufferLib provides optimized environment vectorization. When you inherit from
PufferEnv, you get:

1. AUTOMATIC BUFFER MANAGEMENT: PufferLib creates shared memory buffers for
   observations, rewards, terminals, truncations. You just write to them.

2. MULTI-AGENT SUPPORT: The same pattern works for 1 agent or 100 agents.
   You define `num_agents` and PufferLib handles the rest.

3. VECTORIZATION COMPATIBILITY: Your env works with pufferlib.vector.make()
   which can run multiple copies in parallel (Serial or Multiprocessing).

KEY DIFFERENCES from Gymnasium:
-------------------------------
- Define `single_observation_space` and `single_action_space` (not plural)
- Set `self.num_agents` (1 for single-agent)
- Call `super().__init__(buf)` which creates self.observations, self.rewards, etc.
- Update arrays IN-PLACE: `self.observations[:] = ...` not `return obs`
- reset() and step() still return values, but also update internal buffers
"""


class MoveToTargetEnv(pufferlib.PufferEnv):
    """
    A simple environment where an agent navigates to a target position.

    This is identical to learn.py's MoveToTargetEnv, but adapted to PufferLib's
    patterns. The game logic is the same, only the interface changes.

    GAME RULES:
    - Agent starts at random position in [-0.8, 0.8] x [-0.8, 0.8]
    - Target is at random position (at least 0.3 units away from agent)
    - Agent can: NOOP (0), UP (1), DOWN (2), LEFT (3), RIGHT (4)
    - Episode ends when: agent reaches target, hits wall (|x|>1 or |y|>1), or 200 steps
    - Reward: -0.01/step + distance shaping + terminal bonuses
    """

    # Type hints for attributes created by super().__init__()
    observations: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray

    def __init__(self, buf=None, seed=0):
        """
        WHY these parameters?
        ---------------------
        - buf: Optional shared memory buffer from PufferLib's vectorization.
               When running multiple envs, they share memory for efficiency.
               If None, PufferLib creates a buffer automatically.

        - seed: Random seed for reproducibility. Essential for debugging!

        WHAT to do in __init__:
        1. Define single_observation_space (what ONE agent sees)
        2. Define single_action_space (what actions ONE agent can take)
        3. Set self.num_agents (1 for single-agent env)
        4. Call super().__init__(buf) - THIS CREATES self.observations, etc.
        5. Initialize game state variables
        6. Set up random number generator
        """
        # -----------------------------------------------------------------
        # TODO 1.1: Define the observation space
        # -----------------------------------------------------------------
        # WHAT the agent sees: [agent_x, agent_y, target_x, target_y, dx, dy]
        # - Positions are in [-1, 1] (arena bounds)
        # - dx, dy (direction to target) can be in [-2, 2]
        #
        # WHY "single_observation_space" not "observation_space"?
        # PufferLib distinguishes single-agent spaces from joint spaces.
        # For multi-agent, observation_space would be (num_agents, obs_dim).
        # We define the SINGLE agent's view, PufferLib handles batching.
        #
        # YOUR CODE: Create self.single_observation_space as gymnasium.spaces.Box
        # Hint: Box(low=-2.0, high=2.0, shape=(6,), dtype=np.float32)

        self.single_observation_space = gymnasium.spaces.Box(
            low=-2.0, high=2.0, shape=(6,), dtype=np.float32
        )

        # -----------------------------------------------------------------
        # TODO 1.2: Define the action space
        # -----------------------------------------------------------------
        # WHAT actions are available: 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        #
        # YOUR CODE: Create self.single_action_space as gymnasium.spaces.Discrete(5)

        self.single_action_space = gymnasium.spaces.Discrete(5)

        # -----------------------------------------------------------------
        # TODO 1.3: Set the number of agents
        # -----------------------------------------------------------------
        # For single-agent environments, num_agents = 1.
        # PufferLib uses this to allocate the right buffer sizes.
        #
        # YOUR CODE: Set self.num_agents = 1

        self.num_agents = 1

        # -----------------------------------------------------------------
        # CRITICAL: Call super().__init__(buf)
        # -----------------------------------------------------------------
        # This MUST come after defining spaces and num_agents!
        # It creates:
        #   - self.observations: array of shape (num_agents, *obs_shape)
        #   - self.rewards: array of shape (num_agents,)
        #   - self.terminals: array of shape (num_agents,)
        #   - self.truncations: array of shape (num_agents,)
        #
        # These are the buffers you'll update in reset() and step().
        super().__init__(buf)

        # -----------------------------------------------------------------
        # TODO 1.4: Initialize game state variables
        # -----------------------------------------------------------------
        # Track the actual game state (not observations, those are derived).
        # For single-agent, these are simple arrays of shape (2,) for positions.
        #
        # WHAT to initialize:
        # - self.agent_pos: np.zeros(2, dtype=np.float32) - agent's [x, y]
        # - self.target_pos: np.zeros(2, dtype=np.float32) - target's [x, y]
        # - self.tick: 0 - step counter within episode
        #
        # Also initialize constants:
        # - self.max_steps = 200
        # - self.target_radius = 0.1 (how close to count as "reached")
        # - self.move_speed = 0.05 (movement per action)
        # - self.arena_size = 1.0 (arena is [-1, 1] x [-1, 1])
        #
        # YOUR CODE: Initialize game state

        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.tick = 0

        self.max_steps = 200
        self.target_radius = 0.1
        self.move_speed = 0.05
        self.arena_size = 1.0

        # Set up random number generator for reproducibility
        self.rng = np.random.default_rng(seed=seed)

        # Track previous distance for reward shaping
        self.prev_dist = 0.0

    def reset(self, seed=None):
        """
        WHY reset()?
        ------------
        Start a fresh episode. Called at the beginning and after each episode ends.

        WHAT to do:
        1. Randomize agent position
        2. Randomize target position (not too close to agent!)
        3. Reset step counter
        4. Compute initial distance (for reward shaping)
        5. Fill self.observations[:] with initial state

        WHY update self.observations[:] in-place?
        PufferLib uses shared memory buffers. By updating in-place, we avoid
        copying data. The [:] syntax means "update the existing array contents".

        RETURNS:
        - self.observations: the observation buffer (now filled with initial state)
        - []: empty list of infos (PufferLib expects a list)
        """
        # -----------------------------------------------------------------
        # TODO 2.1: Implement reset()
        # -----------------------------------------------------------------
        # Step 1: Randomize agent position
        #         self.agent_pos[:] = self.rng.uniform(-0.8, 0.8, size=2)
        #
        # Step 2: Randomize target position
        #         self.target_pos[:] = self.rng.uniform(-0.8, 0.8, size=2)
        #
        # Step 3: Ensure target is far enough from agent (at least 0.3 units)
        #         while np.linalg.norm(self.agent_pos - self.target_pos) < 0.3:
        #             self.target_pos[:] = self.rng.uniform(-0.8, 0.8, size=2)
        #
        # Step 4: Reset step counter
        #         self.tick = 0
        #
        # Step 5: Compute initial distance
        #         self.prev_dist = np.linalg.norm(self.agent_pos - self.target_pos)
        #
        # Step 6: Fill observations buffer
        #         self.observations[0, 0] = self.agent_pos[0]  # agent_x
        #         self.observations[0, 1] = self.agent_pos[1]  # agent_y
        #         self.observations[0, 2] = self.target_pos[0]  # target_x
        #         self.observations[0, 3] = self.target_pos[1]  # target_y
        #         self.observations[0, 4] = self.target_pos[0] - self.agent_pos[0]  # dx
        #         self.observations[0, 5] = self.target_pos[1] - self.agent_pos[1]  # dy
        #
        # Note: We index [0, :] because num_agents=1, so observations has shape (1, 6)
        #
        # YOUR CODE:

        self.agent_pos[:] = self.rng.uniform(-0.8, 0.8, size=2)
        self.target_pos[:] = self.rng.uniform(-0.8, 0.8, size=2)

        while np.linalg.norm(self.agent_pos - self.target_pos) < 0.3:
            self.target_pos[:] = self.rng.uniform(-0.8, 0.8, size=2)

        self.tick = 0

        self.prev_dist = np.linalg.norm(self.agent_pos - self.target_pos)

        self.observations[0, 0] = self.agent_pos[0]
        self.observations[0, 1] = self.agent_pos[1]
        self.observations[0, 2] = self.target_pos[0]
        self.observations[0, 3] = self.target_pos[1]
        self.observations[0, 4] = self.target_pos[0] - self.agent_pos[0]
        self.observations[0, 5] = self.target_pos[1] - self.agent_pos[1]

        return self.observations, []

    def step(self, actions):
        """
        WHY step()?
        -----------
        The core game loop. Called every timestep with the agent's chosen action.

        WHAT to do:
        1. Apply the action (move agent)
        2. Compute reward (time penalty + distance shaping + terminal bonus)
        3. Check terminal conditions (reached target? hit wall? timeout?)
        4. Update buffers (observations, rewards, terminals, truncations)
        5. Auto-reset if episode ended

        PARAMETERS:
        - actions: numpy array of shape (num_agents,) = (1,) for us
                   Each value is an integer 0-4

        RETURNS:
        - self.observations: updated observation buffer
        - self.rewards: updated reward buffer
        - self.terminals: updated terminal buffer
        - self.truncations: updated truncation buffer
        - infos: list of dicts with episode stats for finished episodes
        """
        # -----------------------------------------------------------------
        # TODO 2.2: Implement step()
        # -----------------------------------------------------------------
        # Step 1: Get the action (we only have 1 agent)
        #         action = actions[0]
        #
        # Step 2: Convert action to movement
        #         dx, dy = 0.0, 0.0
        #         if action == 1: dy = self.move_speed   # UP
        #         elif action == 2: dy = -self.move_speed  # DOWN
        #         elif action == 3: dx = -self.move_speed  # LEFT
        #         elif action == 4: dx = self.move_speed   # RIGHT
        #
        # Step 3: Apply movement
        #         self.agent_pos[0] += dx
        #         self.agent_pos[1] += dy
        #         self.tick += 1
        #
        # Step 4: Compute distance and rewards
        #         distance = np.linalg.norm(self.agent_pos - self.target_pos)
        #         reward = -0.01  # Time penalty
        #         reward += 2.0 * (self.prev_dist - distance)  # Distance shaping
        #         self.prev_dist = distance
        #
        # Step 5: Check terminal conditions
        #         reached_target = distance < self.target_radius
        #         hit_wall = (abs(self.agent_pos[0]) > self.arena_size or
        #                     abs(self.agent_pos[1]) > self.arena_size)
        #         timed_out = self.tick >= self.max_steps
        #
        # Step 6: Apply terminal rewards
        #         if reached_target: reward += 1.0
        #         if hit_wall: reward -= 0.5
        #
        # Step 7: Set terminal and truncation flags
        #         terminal = reached_target or hit_wall
        #         truncation = timed_out and not terminal
        #
        # Step 8: Update buffers
        #         self.rewards[0] = reward
        #         self.terminals[0] = terminal
        #         self.truncations[0] = truncation
        #
        # Step 9: Build info dict for finished episodes
        #         infos = []
        #         if terminal or truncation:
        #             infos.append({
        #                 'episode_length': self.tick,
        #                 'reached_target': reached_target,
        #                 'hit_wall': hit_wall,
        #                 'reward': reward,
        #             })
        #             # Auto-reset for next episode
        #             self.reset()
        #
        # Step 10: Update observations (whether reset or not)
        #          self.observations[0, 0] = self.agent_pos[0]
        #          self.observations[0, 1] = self.agent_pos[1]
        #          self.observations[0, 2] = self.target_pos[0]
        #          self.observations[0, 3] = self.target_pos[1]
        #          self.observations[0, 4] = self.target_pos[0] - self.agent_pos[0]
        #          self.observations[0, 5] = self.target_pos[1] - self.agent_pos[1]
        #
        # YOUR CODE:

        action = actions[0]

        dx, dy = 0.0, 0.0
        if action == 1:
            dy = self.move_speed
        elif action == 2:
            dy = -self.move_speed  # DOWN
        elif action == 3:
            dx = -self.move_speed  # LEFT
        elif action == 4:
            dx = self.move_speed  # RIGHT

        self.agent_pos[0] += dx
        self.agent_pos[1] += dy
        self.tick += 1

        distance = np.linalg.norm(self.target_pos - self.agent_pos)
        reward = -0.01
        reward += 2 * (self.prev_dist - distance)
        self.prev_dist = distance

        reached_target = distance < self.target_radius
        hit_wall = (
            abs(self.agent_pos[0]) > self.arena_size
            or abs(self.agent_pos[1]) > self.arena_size
        )
        timed_out = self.tick >= self.max_steps

        if reached_target:
            reward += 1.0
        if hit_wall:
            reward -= 0.5

        terminal = reached_target or hit_wall
        truncation = timed_out and not terminal

        self.rewards[0] = reward
        self.terminals[0] = terminal
        self.truncations[0] = truncation

        infos = []
        if terminal or truncation:
            infos.append(
                {
                    "episode_length": self.tick,
                    "reached_target": reached_target,
                    "hit_wall": hit_wall,
                    "reward": reward,
                }
            )
            self.reset()

        self.observations[0, 0] = self.agent_pos[0]
        self.observations[0, 1] = self.agent_pos[1]
        self.observations[0, 2] = self.target_pos[0]
        self.observations[0, 3] = self.target_pos[1]
        self.observations[0, 4] = self.target_pos[0] - self.agent_pos[0]
        self.observations[0, 5] = self.target_pos[1] - self.agent_pos[1]

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def render(self):
        """
        Simple ASCII rendering for debugging.
        Shows a 20x20 grid with agent (A) and target (T).
        """
        grid_size = 20
        grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

        # Convert positions from [-1, 1] to grid indices [0, grid_size-1]
        def to_grid(pos):
            x = int((pos[0] + 1) / 2 * (grid_size - 1))
            y = int((1 - (pos[1] + 1) / 2) * (grid_size - 1))  # Flip y for display
            return max(0, min(grid_size - 1, x)), max(0, min(grid_size - 1, y))

        tx, ty = to_grid(self.target_pos)
        ax, ay = to_grid(self.agent_pos)

        grid[ty][tx] = "T"
        grid[ay][ax] = "A"

        print(f"\nStep {self.tick}:")
        print("+" + "-" * grid_size + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * grid_size + "+")

    def close(self):
        pass


# =============================================================================
# SECTION 2: TESTING ENVIRONMENT
# =============================================================================
"""
WHY test before training?
-------------------------
If your environment is broken, RL will silently fail to learn.
You'll waste hours wondering why training doesn't work.

ALWAYS verify:
1. Environment creates without errors
2. reset() returns correct shapes
3. step() works with valid actions
4. Episodes actually terminate
5. A simple heuristic can solve it
"""


def test_environment():
    """Run basic sanity checks on the PufferLib environment."""
    print("=" * 60)
    print("TESTING MoveToTargetEnv (PufferLib)")
    print("=" * 60)

    # Test 1: Creation
    print("\n[TEST 1] Creating environment...")
    try:
        env = MoveToTargetEnv(seed=42)
        print(f"  OK: Created env")
        print(f"  Observation space: {env.single_observation_space}")
        print(f"  Action space: {env.single_action_space}")
        print(f"  Num agents: {env.num_agents}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 2: Reset
    print("\n[TEST 2] Testing reset()...")
    try:
        obs, info = env.reset()
        print(f"  OK: reset() returned observations with shape {obs.shape}")
        print(f"  Sample observation: {obs[0]}")
        assert obs.shape == (1, 6), f"Wrong shape: {obs.shape}, expected (1, 6)"
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 3: Step with random actions
    print("\n[TEST 3] Testing step() with random actions...")
    try:
        for i in range(5):
            actions = np.array([np.random.randint(0, 5)])  # Shape (1,)
            obs, rewards, terminals, truncations, infos = env.step(actions)
            print(f"  Step {i + 1}: reward={rewards[0]:.3f}, terminal={terminals[0]}")
        print(f"  OK: step() works")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 4: Run until episode terminates using heuristic
    print("\n[TEST 4] Running until episode terminates...")
    try:
        obs, _ = env.reset()
        total_steps = 0
        episodes_finished = 0

        while episodes_finished < 2 and total_steps < 500:
            # Simple heuristic: move toward target
            dx = obs[0, 4]  # target_x - agent_x
            dy = obs[0, 5]  # target_y - agent_y

            if abs(dx) > abs(dy):
                action = 4 if dx > 0 else 3  # RIGHT or LEFT
            else:
                action = 1 if dy > 0 else 2  # UP or DOWN

            actions = np.array([action])
            obs, rewards, terminals, truncations, infos = env.step(actions)
            total_steps += 1

            if infos:
                for info in infos:
                    episodes_finished += 1
                    print(f"  Episode finished: {info}")

        print(f"  OK: Completed {episodes_finished} episodes in {total_steps} steps")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 5: Test with PufferLib vectorization
    print("\n[TEST 5] Testing with pufferlib.vector.make()...")
    try:
        vecenv = pufferlib.vector.make(
            MoveToTargetEnv,
            num_envs=4,
            backend=pufferlib.vector.Serial,
        )
        obs, _ = vecenv.reset()
        print(f"  OK: Created vectorized env with 4 copies")
        print(f"  Vectorized observation shape: {obs.shape}")

        # Take a few steps
        for i in range(3):
            actions = np.random.randint(0, 5, size=4)
            obs, rewards, terminals, truncations, infos = vecenv.step(actions)
        print(f"  OK: Vectorized stepping works")
        vecenv.close()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("ALL ENVIRONMENT TESTS PASSED!")
    print("=" * 60)
    return True


# =============================================================================
# SECTION 3: POLICY NETWORK
# =============================================================================
"""
WHY this specific architecture?
-------------------------------
PufferLib expects policies to follow certain conventions:

1. forward_eval(observations, state=None) -> (logits, values)
   - This is what the trainer calls during rollout collection
   - Returns action LOGITS (not probabilities) and value estimates
   - The `state` parameter is for RNNs (we return None for feedforward)

2. Use pufferlib.pytorch.layer_init() for weight initialization
   - Proper initialization is crucial for stable learning
   - Different std values for actor vs critic heads

WHY layer_init?
---------------
Neural network initialization matters A LOT for RL:
- Too large weights -> exploding gradients, unstable training
- Too small weights -> vanishing gradients, slow learning
- layer_init uses orthogonal initialization which works well for RL

ARCHITECTURE:
observation (6) -> encoder (64 -> 64) -> actor head (5) + critic head (1)
"""


class Policy(nn.Module):
    """
    Actor-Critic policy network following PufferLib conventions.

    The network has:
    - Shared encoder: processes observations into features
    - Actor head: outputs action logits (5 actions)
    - Critic head: outputs value estimate (1 value)
    """

    def __init__(self, env, hidden_size=64):
        """
        WHY take env as parameter?
        --------------------------
        We extract observation and action sizes from the environment.
        This is more robust than hardcoding dimensions.

        PufferLib's vectorized envs provide:
        - env.single_observation_space: shape of one agent's observation
        - env.single_action_space: the action space for one agent

        For regular Gymnasium envs, these would be observation_space/action_space.
        """
        super().__init__()

        # Get dimensions from environment
        obs_size = env.single_observation_space.shape[0]
        action_size = env.single_action_space.n

        # -----------------------------------------------------------------
        # TODO 3.1: Create the encoder (shared backbone)
        # -----------------------------------------------------------------
        # The encoder processes observations into a feature vector.
        # Both actor and critic will use these features.
        #
        # Architecture: Linear(obs_size, hidden_size) -> ReLU -> Linear(hidden_size, hidden_size) -> ReLU
        #
        # Use pufferlib.pytorch.layer_init() for each Linear layer.
        # Default std works for hidden layers.
        #
        # Example:
        #   self.encoder = nn.Sequential(
        #       pufferlib.pytorch.layer_init(nn.Linear(obs_size, hidden_size)),
        #       nn.ReLU(),
        #       pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
        #       nn.ReLU(),
        #   )
        #
        # YOUR CODE:

        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(obs_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        # -----------------------------------------------------------------
        # TODO 3.2: Create the actor head
        # -----------------------------------------------------------------
        # Outputs action logits. Use std=0.01 for small initial outputs.
        # WHY small std? We want initial actions to be nearly uniform.
        #
        # self.actor = pufferlib.pytorch.layer_init(
        #     nn.Linear(hidden_size, action_size), std=0.01
        # )
        #
        # YOUR CODE:

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, action_size), std=0.01
        )

        # -----------------------------------------------------------------
        # TODO 3.3: Create the critic head
        # -----------------------------------------------------------------
        # Outputs value estimate. Use std=1.0 for reasonable initial values.
        #
        # self.critic = pufferlib.pytorch.layer_init(
        #     nn.Linear(hidden_size, 1), std=1.0
        # )
        #
        # YOUR CODE:

        self.critic = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward_eval(self, observations, state=None):
        """
        WHY forward_eval specifically?
        ------------------------------
        PufferLib's trainer calls forward_eval() during rollout collection.
        It expects (logits, values) as return value.

        The state parameter is for recurrent networks (LSTMs). For feedforward
        networks like ours, we ignore it and return None.

        PARAMETERS:
        - observations: tensor of shape (batch_size, obs_size)
        - state: For RNN/LSTM policies, carries hidden state between steps.
                 For feedforward networks (like ours), always None.

        RETURNS:
        - logits: tensor of shape (batch_size, action_size) - unnormalized action scores
        - values: tensor of shape (batch_size, 1) - value estimates
        """
        # -----------------------------------------------------------------
        # TODO 3.4: Implement forward_eval
        # -----------------------------------------------------------------
        # Step 1: Pass observations through encoder
        #         hidden = self.encoder(observations)
        #
        # Step 2: Get action logits from actor head
        #         logits = self.actor(hidden)
        #
        # Step 3: Get value estimate from critic head
        #         values = self.critic(hidden)
        #
        # Step 4: Return (logits, values)

        hidden = self.encoder(observations)
        logits = self.actor(hidden)
        values = self.critic(hidden)

        return logits, values

    def forward(self, observations, state=None):
        """Standard PyTorch forward - required by PufferLib trainer."""
        return self.forward_eval(observations, state)


# =============================================================================
# SECTION 4: TESTING POLICY
# =============================================================================
"""
WHY test the policy?
--------------------
Verify the network architecture is correct before training.
Common bugs:
- Wrong input/output dimensions
- Missing activations
- NaN in outputs
"""


def test_policy():
    """Run basic sanity checks on the Policy network."""
    print("\n" + "=" * 60)
    print("TESTING Policy Network")
    print("=" * 60)

    # Test 1: Creation
    print("\n[TEST 1] Creating policy...")
    try:
        # Create a dummy env to get dimensions
        env = MoveToTargetEnv()
        env.reset()  # Initialize the env

        policy = Policy(env, hidden_size=64)
        print(f"  OK: Created policy")

        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"  Total parameters: {total_params}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 2: forward_eval
    print("\n[TEST 2] Testing forward_eval()...")
    try:
        # Create batch of observations
        obs = torch.randn(4, 6)  # batch of 4
        logits, values = policy.forward_eval(obs)

        print(f"  Input shape: {obs.shape}")
        print(f"  Logits shape: {logits.shape} (expected: [4, 5])")
        print(f"  Values shape: {values.shape} (expected: [4, 1])")

        assert logits.shape == (4, 5), f"Wrong logits shape: {logits.shape}"
        assert values.shape == (4, 1), f"Wrong values shape: {values.shape}"

        # Check for NaN
        assert not torch.isnan(logits).any(), "NaN in logits!"
        assert not torch.isnan(values).any(), "NaN in values!"

        print("  OK: Shapes correct, no NaN")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 3: Single observation
    print("\n[TEST 3] Testing with single observation...")
    try:
        obs = torch.randn(1, 6)
        logits, values = policy.forward_eval(obs)

        print(f"  Logits: {logits}")
        print(f"  Value: {values}")
        print("  OK: Single observation works")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("ALL POLICY TESTS PASSED!")
    print("=" * 60)
    return True


# =============================================================================
# SECTION 5: TRAINING WITH PUFFERLIB
# =============================================================================
"""
WHY use pufferl.PuffeRL?
------------------------
PufferLib's trainer handles ALL the RL internals:
- Rollout collection (running envs, storing experiences)
- GAE advantage computation
- PPO loss calculation (clipped surrogate, value loss, entropy)
- Gradient updates with clipping
- Logging and metrics

This means our training code is MUCH simpler than learn.py!

THE TRAINING LOOP:
-----------------
1. Create vectorized environment
2. Create policy
3. Create config dict with hyperparameters
4. Create PuffeRL trainer
5. Loop: trainer.evaluate() -> trainer.train()

WHAT trainer.evaluate() does:
- Runs the policy in all environments
- Collects experiences into buffers
- Computes advantages and returns

WHAT trainer.train() does:
- Runs PPO update on collected experiences
- Updates policy weights
- Logs metrics
"""


def train(quick_test=False):
    """
    Main training function using PufferLib's trainer.

    PARAMETERS:
    - quick_test: if True, run short training to verify code works
                  if False, run full training to see actual learning
    """
    # -----------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------
    if quick_test:
        total_timesteps = 10000
        num_envs = 4
    else:
        total_timesteps = 100000
        num_envs = 8

    # Detect device
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"

    print("=" * 60)
    print("TRAINING WITH PUFFERLIB")
    print("=" * 60)
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {device}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Num environments: {num_envs}")
    print("=" * 60)

    # -----------------------------------------------------------------
    # TODO 5.1: Create vectorized environment
    # -----------------------------------------------------------------
    # PufferLib's vector.make() creates multiple environment copies.
    #
    # Backend options:
    # - Serial: Runs envs sequentially. Good for debugging because errors
    #   appear in the main process with full stack traces.
    # - Multiprocessing: Runs envs in parallel. Much faster for many envs,
    #   but errors in subprocesses are harder to debug.
    #
    # Tip: Use Serial until your code works, then switch to Multiprocessing.
    #
    # vecenv = pufferlib.vector.make(
    #     MoveToTargetEnv,
    #     num_envs=num_envs,
    #     backend=pufferlib.vector.Serial,
    # )
    #
    # YOUR CODE:

    vecenv = pufferlib.vector.make(
        MoveToTargetEnv, num_envs=num_envs, backend=pufferlib.vector.Multiprocessing
    )

    # -----------------------------------------------------------------
    # TODO 5.2: Create policy
    # -----------------------------------------------------------------
    # Use vecenv.driver_env to get a reference to one of the environment copies.
    # This lets us access single_observation_space and single_action_space
    # for creating the policy with correct input/output dimensions.
    # Move policy to device for GPU training.
    #
    # policy = Policy(vecenv.driver_env, hidden_size=64).to(device)
    #
    # YOUR CODE:

    policy = Policy(vecenv.driver_env, hidden_size=64).to(device)
    next(policy.parameters()).device

    # -----------------------------------------------------------------
    # TODO 5.3: Create config
    # -----------------------------------------------------------------
    # PufferLib's trainer uses a Config object for hyperparameters.
    # These are standard PPO values that work well.
    #
    # config = pufferl.Config(
    #     total_timesteps=total_timesteps,
    #     learning_rate=3e-4,
    #     num_steps=128,        # Steps per rollout
    #     num_minibatches=4,    # Minibatches per update
    #     update_epochs=4,      # PPO epochs per update
    #     gamma=0.99,           # Discount factor
    #     gae_lambda=0.95,      # GAE parameter
    #     clip_coef=0.2,        # PPO clipping
    #     vf_coef=0.5,          # Value loss coefficient
    #     ent_coef=0.01,        # Entropy bonus coefficient
    #     max_grad_norm=0.5,    # Gradient clipping
    # )
    #
    # YOUR CODE:

    config = {
        "env": "MoveToTarget",
        "total_timesteps": total_timesteps,
        "learning_rate": 3e-4,
        "batch_size": num_envs * 128,
        "bptt_horizon": 128,
        "minibatch_size": 512,
        "max_minibatch_size": 512,
        "update_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "vf_coef": 0.5,
        "vf_clip_coef": 0.2,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "device": device,
        "seed": 42,
        "torch_deterministic": True,
        "cpu_offload": False,
        "use_rnn": False,
        "compile": False,
        "optimizer": "adam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "anneal_lr": True,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "prio_alpha": 0.8,
        "prio_beta0": 0.2,
        "checkpoint_interval": 200,
        "data_dir": "experiments",
        "precision": "float32",
    }

    # -----------------------------------------------------------------
    # TODO 5.4: Create trainer
    # -----------------------------------------------------------------
    # The PuffeRL trainer handles the entire training loop internals.
    #
    # trainer = pufferl.PuffeRL(
    #     config=config,
    #     vecenv=vecenv,
    #     policy=policy,
    #     optimizer=torch.optim.Adam(policy.parameters(), lr=config.learning_rate),
    # )
    #
    # YOUR CODE:

    trainer = pufferl.PuffeRL(config, vecenv, policy)

    # -----------------------------------------------------------------
    # TODO 5.5: Training loop
    # -----------------------------------------------------------------
    # The training loop is very simple with PufferLib:
    # 1. trainer.evaluate() - collect experiences
    # 2. trainer.train() - run PPO update
    # 3. Repeat until done
    #
    # Example:
    # while not trainer.done:
    #     trainer.evaluate()
    #     trainer.train()
    #
    #     # Print progress every 10 epochs
    #     if trainer.epoch % 10 == 0:
    #         # Get metrics from trainer
    #         metrics = trainer.metrics
    #         print(f"Epoch {trainer.epoch} | "
    #               f"reward: {metrics.get('episode_reward', 0):.2f} | "
    #               f"length: {metrics.get('episode_length', 0):.1f}")
    #
    # Or use the built-in dashboard:
    # while not trainer.done:
    #     trainer.evaluate()
    #     trainer.train()
    #     trainer.print_dashboard()  # Pretty-printed metrics
    #
    # YOUR CODE:

    while trainer.global_step < total_timesteps:
        trainer.evaluate()
        trainer.train()

    # Cleanup
    trainer.close()
    vecenv.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return policy


# =============================================================================
# SECTION 6: EVALUATION WITH ASCII RENDERING
# =============================================================================


def eval_policy(num_episodes=3, delay=0.1):
    """
    Run the trained policy and watch it play with ASCII rendering.

    PARAMETERS:
    - num_episodes: number of episodes to run
    - delay: seconds between frames (for watchability)
    """
    import time
    import glob

    print("=" * 60)
    print("EVALUATING TRAINED POLICY")
    print("=" * 60)

    # Find latest checkpoint
    checkpoints = glob.glob("experiments/**/model.pt", recursive=True)
    if not checkpoints:
        print(
            "No checkpoint found in experiments/. Train first with 'python learn_v2.py train'"
        )
        return

    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(x))
    print(f"Loading checkpoint: {latest_checkpoint}")

    # Create environment (single, not vectorized)
    env = MoveToTargetEnv(seed=int(time.time()))

    # Create and load policy
    policy = Policy(env, hidden_size=64)
    checkpoint = torch.load(latest_checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(checkpoint)
    policy.eval()

    print(f"Running {num_episodes} episodes...\n")

    for ep in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"EPISODE {ep + 1}")
        print(f"{'=' * 60}")

        obs, _ = env.reset()
        env.render()
        time.sleep(delay)

        done = False
        total_reward = 0.0

        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float()
                logits, _ = policy(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()

            # Step environment
            obs, rewards, terminals, truncations, infos = env.step(np.array([action]))
            total_reward += rewards[0]
            done = terminals[0] or truncations[0]

            # Render
            env.render()
            action_names = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT"]
            print(f"Action: {action_names[action]}, Reward: {rewards[0]:.3f}")
            time.sleep(delay)

        # Episode summary
        if infos:
            info = infos[0]
            result = (
                "REACHED TARGET!"
                if info.get("reached_target")
                else "Failed (wall/timeout)"
            )
            print(f"\nResult: {result}")
            print(f"Episode length: {info.get('episode_length', 'N/A')}")
        print(f"Total reward: {total_reward:.3f}")

    env.close()
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "test":
            # Run all tests
            env_ok = test_environment()
            if env_ok:
                test_policy()
        elif command == "train":
            # Run full training
            test_environment()
            test_policy()
            train(quick_test=False)
        elif command == "quick":
            # Quick training test
            # test_environment()
            # test_policy()
            train(quick_test=True)
        elif command == "eval":
            # Evaluate trained policy with ASCII rendering
            eval_policy(num_episodes=3, delay=0.1)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python learn_v2.py [test|train|quick|eval]")
    else:
        # Default: run tests only
        print("Running tests... (use 'python learn_v2.py train' for full training)")
        print()
        env_ok = test_environment()
        if env_ok:
            test_policy()

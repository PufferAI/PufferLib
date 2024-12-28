import numpy as np
from tqdm.auto import trange

import pufferlib

try:
    from .cy_codeball import CyCodeBall
except ImportError:
    from cy_codeball import CyCodeBall

import gymnasium
import numpy as np
import pufferlib


class CodeBall(pufferlib.PufferEnv):
    def __init__(self, num_envs=2, n_robots=8, n_nitros=0, max_steps=2000,
                 reward_mul=1.0,
                 goal_scored_reward=1.0, loiter_penalty=0.0, ball_reward=0.0,
                 scripted_opponent_type=None,
                 frame_skip=5, buf=None, render_mode='human'):
        is_single = scripted_opponent_type is not None
        self.is_single = is_single
        self.num_envs = num_envs
        self.num_agents = n_robots * num_envs
        if is_single:
            self.num_agents //= 2
        self.n_robots = n_robots
        self.n_nitros = n_nitros
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.single_observation_space = gymnasium.spaces.Box(
            low=-1, high=1,
            shape=((self.n_robots + 3), 9,)
            , dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

        super().__init__(buf=buf)

        self.c_envs = CyCodeBall(
            self.num_envs, self.n_robots, self.n_nitros, frame_skip, reward_mul, max_steps,
            self.is_single,
            goal_scored_reward, loiter_penalty, ball_reward,
            (None, "zero", "random", "run").index(scripted_opponent_type),
            self.observations,
            self.actions, self.rewards, self.terminals, self.truncations,
        )

    def reset(self, seed=None):
        self.c_envs.reset()
        return self.observations, []
    
    def render(self):
        return self.c_envs.render()

    def step(self, actions):
        self.actions[:] = np.clip(actions, -1.0, 1.0)
        self.c_envs.step()

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            [self.c_envs.log()],
        )

    def close(self):
        self.c_envs.close()

if __name__ == '__main__':
    env = CodeBall(num_envs=128)
    obs, _ = env.reset(seed=42)
    print("Observation Space:", env.single_observation_space)
    print("Action Space:", env.single_action_space)
    print("Initial Observations Shape:", obs.shape)

    # actions = np.array([[0, 0]])
    # actions = np.array([[0]])
    actions = np.array([0.0])
    actions = np.tile(actions, (env.num_agents, 4))
    # actions[:, :] = 3
    # actions[:, :] = 0
    actions[:] = 0
    all_obs = []
    all_rewards = []
    terminals = []
    truncations = []
    for _ in trange(10_000):
        actions[:] = np.random.uniform(-1, 1, size=actions.shape)
        obs, rewards, terminated, truncated, info = env.step(actions)
        all_obs.append(obs.copy())
        all_rewards.append(rewards.copy())
        terminals.append(terminated.copy())
        truncations.append(truncated.copy())
    obs = np.stack(all_obs)
    if obs.ndim == 3:
        obs = obs.reshape(obs.shape[0], env.num_agents, -1, 6)
    rewards = np.stack(all_rewards)
    terminals = np.stack(terminals)
    truncations = np.stack(truncations)
    from matplotlib import pyplot as plt
    plt.plot(obs[:, 0, 0, 0], label="Robot 0 X")
    plt.plot(obs[:, 0, 0, 1], label="Robot 0 Y")
    plt.plot(obs[:, 0, 0, 2], label="Robot 0 Z")
    plt.plot(obs[:, 0, -1, -6], label="Ball X")
    plt.plot(obs[:, 0, -1, -5], label="Ball Y")
    plt.plot(obs[:, 0, -1, -4], label="Ball Z")
    plt.plot(obs[:, 0, -1, -3], label="Ball vX")
    plt.plot(obs[:, 0, -1, -2], label="Ball vY")
    plt.plot(obs[:, 0, -1, -1], label="Ball vZ")
    plt.plot(rewards[:, 0], label="Rewards")
    plt.plot(terminals[:, 0], label="Terminals")
    plt.plot(truncations[:, 0], label="Truncations")
    plt.legend()
    plt.show()
    print("Rewards:", rewards.sum(0))
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Observations shape:", obs.shape)

    env.close()

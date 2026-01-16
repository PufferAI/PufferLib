The One-State World is a simple research testbed designed for studying exploration strategies.

- Structure:

    - The environment has only one state.
    - At each step, the agent can choose between two actions:
        - Action 0: Produces a reward with low expectation and low variance. Concretely, rewards are sampled from a Gaussian distribution with mean 0.1 and variance 0 (i.e., always 0.1).
        - Action 1: Produces a reward with higher expectation but also higher variance. Rewards are sampled from a Gaussian distribution with a larger mean, but with substantial variance.
- Objective: 
    - The environment is meant to challenge exploration algorithms by forcing them to balance between:
        - Exploiting the “safe” but low-payoff option (Action 0).
        - Exploring the “risky” but potentially more rewarding option (Action 1).

The core goal is to evaluate how well different exploration methods can identify and manage stochastic, high-variance rewards in a minimal setting.
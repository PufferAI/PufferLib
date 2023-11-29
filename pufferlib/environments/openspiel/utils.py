from pdb import set_trace as T
import numpy as np

import gymnasium

from pufferlib import namespace


def init(self, 
    env,
    n_rollouts,
    min_simulations,
    max_simulations
    ):
    #state.num_agents = state.env.num_players()
    return namespace(self,
        env=env,
        type=env.get_type(),
        n_rollouts=n_rollouts,
        min_simulations=min_simulations,
        max_simulations=max_simulations,
        state=None,
        agents=[],
        has_reset=False,
    )

def observation_space(state):
    return gymnasium.spaces.Dict({
        'obs': gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state.env.observation_tensor_size(),),
            dtype=np.float32,
        ),
        'action_mask': gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(action_space(state).n,),
            dtype=np.int8
        )
    })

def action_space(state):
    return gymnasium.spaces.Discrete(
        state.env.num_distinct_actions())

def render(state, mode=None) -> None:
    if mode == "human":
        print(state.state)

def close(state):
    pass

def act(state, action):
    solve_chance_nodes(state)
    state.state.apply_action(action)

def get_obs_and_infos(state):
    # Before calculating an observation, there could be chance nodes
    # (that may have an effect on the actual observations).
    # E.g. After reset, figure out initial (random) positions of the
    # agents.
    solve_chance_nodes(state)

    if state.state.is_terminal():
        return (
            state.last_obs, 
            {player: {} for player in range(state.env.num_players())},
        )

    # Sequential game:
    curr_player = state.state.current_player()
    mask = state.state.legal_actions(curr_player)
    np_mask = np.zeros(action_space(state).n)
    np_mask[mask] = 1

    state.last_obs = {player: {
        'obs': np.reshape(state.state.observation_tensor(),
            [-1]).astype(np.float32),
        'action_mask': np_mask.astype(np.int8),
    } for player in range(state.env.num_players())}

    state.last_info = {curr_player: {}}

    return (
        {curr_player: state.last_obs[curr_player]},
        state.last_info,
    )

def solve_chance_nodes(state):
    # Before applying action(s), there could be chance nodes.
    # E.g. if env has to figure out, which agent's action should get
    # resolved first in a simultaneous node.
    # Chance node(s): Sample a (non-player) action and apply.
    while state.state.is_chance_node():
        assert state.state.current_player() == -1
        actions, probs = zip(*state.state.chance_outcomes())
        action = np.random.choice(actions, p=probs)
        state.state.apply_action(action)

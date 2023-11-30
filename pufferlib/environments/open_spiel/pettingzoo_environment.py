from pdb import set_trace as T
import numpy as np

import pufferlib
from pufferlib import namespace

from pufferlib.environments.open_spiel.utils import (
    solve_chance_nodes,
    get_obs_and_infos,
    observation_space,
    action_space,
    init,
    render,
    close,
)

def agents(state):
    return state.agents

def possible_agents(state):
    return list(range(state.env.num_players()))

def pz_observation_space(state, agent):
    return observation_space(state)

def pz_action_space(state, agent):
    return action_space(state)

def reset(state, seed = None, options = None):
    state.state = state.env.new_initial_state()
    obs, infos = get_obs_and_infos(state)
    state.agents = state.possible_agents

    if not state.has_reset:
        state.has_reset = True
        state.seed_value = seed
        np.random.seed(seed)

    return obs, infos

def step(state, actions):
    curr_player = state.state.current_player()
    solve_chance_nodes(state)
    state.state.apply_action(actions[curr_player])
    obs, infos = get_obs_and_infos(state)
    rewards = {ag: r for ag, r in enumerate(state.state.returns())}

    # Are we done?
    is_terminated = state.state.is_terminal()
    terminateds = {a: False for a in obs}
    truncateds = {a: False for a in obs}

    if is_terminated:
        terminateds = {a: True for a in state.possible_agents}
        state.agents = []

    return obs, rewards, terminateds, truncateds, infos

class OpenSpielPettingZooEnvironment:
    __init__ = init
    step = step
    reset = reset
    agents = lambda state: state.agents
    possible_agents = property(possible_agents)
    observation_space = pz_observation_space
    action_space = pz_action_space
    render = render
    close = close

from pdb import set_trace as T
import numpy as np

from open_spiel.python.algorithms import mcts

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


def create_bots(state, seed):
    assert seed is not None, 'seed must be set'
    rnd_state = np.random.RandomState(seed)

    evaluator = mcts.RandomRolloutEvaluator(
        n_rollouts=state.n_rollouts,
        random_state=rnd_state
    )

    return [mcts.MCTSBot(
        game=state.env,
        uct_c=2,
        max_simulations=a,
        evaluator=evaluator,
        random_state=rnd_state, 
        child_selection_fn=mcts.SearchNode.puct_value,
        solve=True,
    ) for a in range(state.min_simulations, state.max_simulations + 1)]
    
def reset(state, seed = None, options = None):
    state.state = state.env.new_initial_state()

    if not state.has_reset:
        state.has_reset = True
        state.seed_value = seed
        np.random.seed(seed)
        state.all_bots = create_bots(state, seed)

    state.bot = np.random.choice(state.all_bots)

    if np.random.rand() < 0.5:
        bot_atn = state.bot.step(state.state)
        state.state.apply_action(bot_atn)
    
    obs, infos = get_obs_and_infos(state)
    player = state.state.current_player()
    return obs[player], infos[player]

def step(state, action):
    player = state.state.current_player()
    solve_chance_nodes(state)
    state.state.apply_action(action)

    # Take other move with a bot
    if not state.state.is_terminal():
        bot_atn = state.bot.step(state.state)
        solve_chance_nodes(state)
        state.state.apply_action(bot_atn)

    # Now that we have applied all actions, get the next obs.
    obs, all_infos = get_obs_and_infos(state)
    reward = state.state.returns()[player]
    info = all_infos[player]

    # Are we done?
    terminated = state.state.is_terminal()
    if terminated:
        key = f'win_mcts_{state.bot.max_simulations}'
        info[key] = int(reward==1)

    return obs[player], reward, terminated, False, info

class OpenSpielGymnasiumEnvironment:
    __init__ = init
    step = step
    reset = reset
    observation_space = property(observation_space)
    action_space = property(action_space)
    render = render
    close = close

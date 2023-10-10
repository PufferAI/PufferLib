from pdb import set_trace as T
import numpy as np

import gym
import gymnasium
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.registry


def env_creator():
    '''OpenSpiel creation function'''
    pyspiel = pufferlib.registry.try_import('pyspiel', 'openspiel')
    return pyspiel.load_game

def make_env(name='connect_four'):
    '''OpenSpiel creation function'''
    env = env_creator()(name)
    env = OpenSpielToPettingZoo(env)
    #env = shimmy.OpenspielCompatibilityV0(env, None)
    #env = turn_based_aec_to_parallel_wrapper(env)
    #env = OpenSpielActionMask(env)
    # TODO: needs custom conversion to parallel
    #env = aec_to_parallel_wrapper(env)
    #env = OpenSpielToPettingZoo(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

import gymnasium

class OpenSpielToPettingZoo:
    def __init__(self, env):
        self.env = env
        # Agent IDs are ints, starting from 0.
        self.num_agents = self.env.num_players()
        # Store the open-spiel game type.
        self.type = self.env.get_type()
        # Stores the current open-spiel game state.
        self.state = None
        self.agents = []

    @property
    def possible_agents(self):
        return list(range(self.num_agents))

    def reset(self, seed = None, options = None):
        self.state = self.env.new_initial_state()
        obs, infos = self._get_obs_and_infos()
        self.agents = self.possible_agents
        return obs, infos

    def step(self, action):
        # Before applying action(s), there could be chance nodes.
        # E.g. if env has to figure out, which agent's action should get
        # resolved first in a simultaneous node.
        self._solve_chance_nodes()

        # Sequential game:
        if str(self.type.dynamics) == "Dynamics.SEQUENTIAL":
            curr_player = self.state.current_player()
            assert curr_player in action
            self.state.apply_action(action[curr_player])

            # Compile rewards dict.
            rewards = {ag: r for ag, r in enumerate(self.state.returns())}
        # Simultaneous game.
        else:
            assert self.state.current_player() == -2
            # Apparently, this works, even if one or more actions are invalid.
            self.state.apply_actions([action[ag] for ag in range(self.num_agents)])

        # Now that we have applied all actions, get the next obs.
        obs, infos = self._get_obs_and_infos()

        # Compile rewards dict
        rewards = {ag: r for ag, r in enumerate(self.state.returns())}

        # Are we done?
        is_terminated = self.state.is_terminal()
        terminateds = dict({ag: is_terminated for ag in range(self.num_agents)})
        truncateds = dict({ag: False for ag in range(self.num_agents)})

        if is_terminated:
            self.agents = []

        return obs, rewards, terminateds, truncateds, infos

    def render(self, mode=None) -> None:
        if mode == "human":
            print(self.state)

    def observation_space(self, agent):
        return gymnasium.spaces.Dict({
            'obs': gymnasium.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.env.observation_tensor_size(),),
                dtype=np.float32,
            ),
            'action_mask': gymnasium.spaces.Box(
                low=0,
                high=1,
                shape=(self.action_space(agent).n,),
                dtype=np.int8
            )
        })

    def action_space(self, agent):
        return gymnasium.spaces.Discrete(
            self.env.num_distinct_actions())

    def _get_obs_and_infos(self):
        # Before calculating an observation, there could be chance nodes
        # (that may have an effect on the actual observations).
        # E.g. After reset, figure out initial (random) positions of the
        # agents.
        self._solve_chance_nodes()

        if self.state.is_terminal():
            return self.last_obs, self.last_infos

        # Sequential game:
        if str(self.type.dynamics) == "Dynamics.SEQUENTIAL":
            curr_player = self.state.current_player()
            mask = self.state.legal_actions(curr_player)
            np_mask = np.zeros(self.action_space(curr_player).n)
            np_mask[mask] = 1
            self.last_obs = {curr_player: {
                'obs': np.reshape(self.state.observation_tensor(), [-1]).astype(np.float32),
                'action_mask': np_mask.astype(np.int8),
            }}
            self.last_infos = {curr_player: {}}
            return self.last_obs, self.last_infos
        # Simultaneous game. NOT YET IMPLEMENTED FOR THIS WRAPPER
        # NEED TO HANDLE INFOS AND LAST
        else:
            assert self.state.current_player() == -2
            return {
                ag: np.reshape(self.state.observation_tensor(ag), [-1])
                for ag in range(self.num_agents)
            }

    def _solve_chance_nodes(self):
        # Chance node(s): Sample a (non-player) action and apply.
        while self.state.is_chance_node():
            assert self.state.current_player() == -1
            actions, probs = zip(*self.state.chance_outcomes())
            action = np.random.choice(actions, p=probs)
            self.state.apply_action(action)

class OpenSpielActionMask:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.render = env.render

    def observation_space(self, agent):
        original_space = self.env.observation_space(agent)
        return gymnasium.spaces.Dict({
            'obs': original_space,
            'action_mask': gymnasium.spaces.Box(
                low=0,
                high=1,
                shape=(self.action_space(agent).n,),
                dtype=np.int8
            )
        })

    def apply_action_mask(self, obs, infos):
        for player in obs:
            obs[player] = {
                'obs': obs[player],
                'action_mask': infos[player]['action_mask']
            }
 
    def reset(self, seed=None):
        obs, infos = self.env.reset(seed=seed)
        self.agents = list(obs)
        self.apply_action_mask(obs, infos)
        return obs, infos

    def step(self, actions):
        obs, rewards, dones, truncateds, infos = self.env.step(actions)
        self.apply_action_mask(obs, infos)
        return obs, rewards, dones, truncateds, infos
 
    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def agents(self):
        return self.env.agents




import copy
import warnings
from collections import defaultdict
from typing import Callable, Dict, Optional

from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType, ParallelEnv
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

class turn_based_aec_to_parallel_wrapper(ParallelEnv):
    def __init__(self, aec_env):
        self.aec_env = aec_env

        try:
            self.possible_agents = aec_env.possible_agents
        except AttributeError:
            pass

        self.metadata = aec_env.metadata

        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = (
                self.aec_env.state_space  # pyright: ignore[reportGeneralTypeIssues]
            )
        except AttributeError:
            pass

        try:
            self.render_mode = (
                self.aec_env.render_mode  # pyright: ignore[reportGeneralTypeIssues]
            )
        except AttributeError:
            warnings.warn(
                f"The base environment `{aec_env}` does not have a `render_mode` defined."
            )

    @property
    def unwrapped(self):
        return self.aec_env.unwrapped

    @property
    def observation_spaces(self):
        warnings.warn(
            "The `observation_spaces` dictionary is deprecated. Use the `observation_space` function instead."
        )
        try:
            return {
                agent: self.observation_space(agent) for agent in self.possible_agents
            }
        except AttributeError as e:
            raise AttributeError(
                "The base environment does not have an `observation_spaces` dict attribute. Use the environments `observation_space` method instead"
            ) from e

    @property
    def action_spaces(self):
        warnings.warn(
            "The `action_spaces` dictionary is deprecated. Use the `action_space` function instead."
        )
        try:
            return {agent: self.action_space(agent) for agent in self.possible_agents}
        except AttributeError as e:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            ) from e

    def observation_space(self, agent):
        return self.aec_env.observation_space(agent)

    def action_space(self, agent):
        return self.aec_env.action_space(agent)

    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed, options=options)
        self.agents = self.aec_env.agents[:]
        observations = {
            agent: self.aec_env.observe(agent)
            for agent in self.aec_env.agents
            if not (self.aec_env.terminations[agent] or self.aec_env.truncations[agent])
        }

        infos = {**self.aec_env.infos}
        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}
        self.aec_env.step(actions[self.aec_env.agent_selection])
        rewards = {**self.aec_env.rewards}
        terminations = {**self.aec_env.terminations}
        truncations = {**self.aec_env.truncations}
        infos = {**self.aec_env.infos}
        observations = {
            agent: self.aec_env.observe(agent) for agent in self.aec_env.agents
        }

        while self.aec_env.agents:
            if (
                self.aec_env.terminations[self.aec_env.agent_selection]
                or self.aec_env.truncations[self.aec_env.agent_selection]
            ):
                self.aec_env.step(None)
            else:
                break
            # no need to update data after null step (nothing should change other than the active agent)

        for agent in self.aec_env.agents:
            infos[agent]["active_agent"] = self.aec_env.agent_selection
        self.agents = self.aec_env.agents
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.aec_env.render()

    def state(self):
        return self.aec_env.state()

    def close(self):
        return self.aec_env.close()
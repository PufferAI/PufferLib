from pdb import set_trace as T
import numpy as np

from collections.abc import MutableMapping
import functools

import gym

def SingleToMultiAgent(Env):
    class MultiAgentWrapper(Env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.agents = [1]

            # Single agent envs use obs/atn properties
            self._observation_space = self.observation_space
            self._action_space = self.action_space

            del self.observation_space
            del self.action_space
 
        def observation_space(self, agent: int):
            return self._observation_space

        def action_space(self, agent: int):
            return self._action_space

        def reset(self):
            ob = super().reset()
            return {1: ob}

        def step(self, action):
            action = action[1]
            ob, reward, done, info = super().step(action)

            obs = {1: ob}
            rewards = {1: reward}
            if done:
                dones = {1: done, '__all__': True}
            else:
                dones = {1: done}
            infos = {1: info}

            return obs, rewards, dones, infos

    return MultiAgentWrapper

def EnvWrapper(Env, *args):
    class Wrapped(Env):
        def __init__(self,
                *args,
                feature_parser=None,
                reward_shaper=None,
                emulate_flat_obs=True,
                emulate_flat_atn=True,
                emulate_const_horizon=1024,
                emulate_const_num_agents=128,
                **kwargs):

            # Infer obs space from first agent
            # Assumes all agents have the same obs space
            self.dummy_obs = {}
            self._step = 0

            self.feature_parser = feature_parser
            self.reward_shaper = reward_shaper

            self.emulate_flat_obs = emulate_flat_obs
            self.emulate_flat_atn = emulate_flat_atn
            self.emulate_const_horizon = emulate_const_horizon
            self.emulate_const_num_agents = emulate_const_num_agents

            super().__init__(**kwargs)

        def action_space(self, agent):
            '''Neural MMO Action Space

            Args:
                agent: Agent ID

            Returns:
                actions: gym.spaces object contained the structured actions
                for the specified agent. Each action is parameterized by a list
                of discrete-valued arguments. These consist of both fixed, k-way
                choices (such as movement direction) and selections from the
                observation space (such as targeting)'''
            atn_space = super().action_space(agent)

            if self.emulate_flat_atn:
                return pack_atn_space(atn_space)

        def structured_observation_space(self, agent: int, dtype=np.float32):
            if self.feature_parser:
                return self.feature_parser.spec
            return self.observation_space(agent)

        def observation_space(self, agent: int, dtype=np.float32):
            obs_space = super().observation_space(agent)

            if agent not in self.dummy_obs:
                # TODO: Zero this obs
                dummy = obs_space.sample()
                self.dummy_obs[agent] = zero(dummy)

            dummy = self.dummy_obs[agent]

            if self.feature_parser:
                dummy = self.feature_parser({agent: dummy}, self._step)[agent]

            if self.emulate_flat_obs:
                dummy = flatten_ob(dummy)

            shape = dummy.shape

            return gym.spaces.Box(
                low=-2**20, high=2**20,
                shape=shape, dtype=dtype
            )

        def _process_obs(self, obs):
            # Faster to have feature parser on env but then 
            # you have to somehow mod the space to pack unpack the modded feat
            # space instead of just using the orig and featurizing in the net
            if self.feature_parser:
                obs = self.feature_parser(obs, self._step)

            if self.emulate_flat_obs:
                obs = pack_obs(obs)

            return obs

        def reset(self):
            self.reset_calls_step = False
            obs = super().reset()

            # Some envs implement reset by calling step
            if not self.reset_calls_step:
                obs = self._process_obs(obs)

            self._step = 0
            return obs

        def step(self, actions, **kwargs):
            # Unpack actions
            #if self.emulate_flat_atn:
            #    for k, v in actions.items():
            #        actions[k] = unflatten(v, super().action_space(k))
            self.reset_calls_step = True

            obs, rewards, dones, infos = super().step(actions)
            self._step += 1

            #TODO: Add some of this to reset
            if self.emulate_const_num_agents:
                pad_const_num_agents(self.dummy_obs, obs, rewards, dones, infos)
                self.possible_agents = [i for i in range(1, self.emulate_const_num_agents + 1)]
           
            obs = self._process_obs(obs)

            if self.reward_shaper:
                rewards = self.reward_shaper(rewards, self._step)

            all_done = True
            for agent in dones:
                if agent == '__all__':
                    continue
                if not dones[agent]:
                    all_done = False

            # TODO: Figure out termination bounds
            if self.emulate_const_horizon:
                assert self._step <= self.emulate_const_horizon
                if self._step == self.emulate_const_horizon or all_done:
                    const_horizon(dones)
                else:
                    for agent in dones:
                        dones[agent] = False

            # Should __all__ only be here if true?
            if '__all__' in dones:
                dones['__all__'] = all_done

            # No agents alive
            if not obs:
                const_horizon(dones)

            return obs, rewards, dones, infos

    return Wrapped

def pad_const_num_agents(dummy_obs, obs, rewards, dones, infos):
    '''Requires constant integer agent keys'''
    for k in dummy_obs:
        if k not in obs:                                                  
            obs[k] = dummy_obs[k]
            rewards[k] = 0                                                 
            infos[k] = {}
            dones[k] = False

def const_horizon(dones):
    for agent in dones:
        dones[agent] = True

    return dones

def zero(nested_dict):
    for k, v in nested_dict.items():
        if type(v) == np.ndarray:
            nested_dict[k] *= 0
        else:
            zero(nested_dict[k])
    return nested_dict

def flatten(nested_dict, parent_key=[]):
    items = []
    if isinstance(nested_dict, MutableMapping) or isinstance(nested_dict, gym.spaces.Dict):
        for k, v in nested_dict.items():
            new_key = parent_key + [k]
            items.extend(flatten(v, new_key).items())
    elif parent_key:
        items.append((parent_key, nested_dict))
    else:
        return nested_dict
 
    return {tuple(k): v for k, v in items}

def unflatten(ary, space, nested_dict={}):
    for k, v in space.items():
        if isinstance(v, MutableMapping) or isinstance(v, gym.spaces.Dict):
            nested_dict[k] = {}
            unflatten(ary, v, nested_dict[k])
        else:
            nested_dict[k] = ary[0]
    return nested_dict

def pack_obs_space(obs_space, dtype=np.float32):
    assert(isinstance(obs_space, gym.Space)), 'Arg must be a gym space'

    if isinstance(obs_space, gym.spaces.Box):
        return obs_space

    flat = flatten(obs_space)

    n = 0
    for e in flat.values():
        n += np.prod(e.shape)

    return gym.spaces.Box(
        low=-2**20, high=2**20,
        shape=(int(n),), dtype=dtype
    )

def pack_atn_space(atn_space):
    assert(isinstance(atn_space, gym.Space)), 'Arg must be a gym space'

    if isinstance(atn_space, gym.spaces.Discrete):
        return atn_space

    flat = flatten(atn_space)

    lens = []
    for e in flat.values():
        lens.append(e.n)

    return gym.spaces.MultiDiscrete(lens) 

def flatten_ob(ob):
    flat = flatten(ob)
    vals = [e.ravel() for e in flat.values()]
    return np.concatenate(vals)

def pack_obs(obs):
    return {k: flatten_ob(v) for k, v in obs.items()}

def pack_and_batch_obs(obs):
    obs = list(pack_obs(obs).values())
    return np.stack(obs, axis=0)

def unpack_batched_obs(obs_space, packed_obs):
    assert(isinstance(obs_space, gym.Space)), 'First arg must be a gym space'

    batch = packed_obs.shape[0]
    obs = {}
    idx = 0

    flat_obs_space = flatten(obs_space)
    for key_list, val in flat_obs_space.items():
        obs_ptr = obs
        for key in key_list[:-1]:
            if key not in obs_ptr:
                obs_ptr = obs_ptr[key]

        key = key_list[-1]
        inc = np.prod(val.shape)
        obs_ptr[key] = packed_obs[:, idx:idx + inc].reshape(batch, *val.shape)
        idx = idx + inc

    return obs
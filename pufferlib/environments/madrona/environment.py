from pdb import set_trace as T
import numpy as np

# import gym
# import shimmy

import math
import torch
import functools
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete

import pufferlib
import pufferlib.emulation
import pufferlib.environments

import gpu_hideseek


def env_creator(name='hide_and_seek'):
    return functools.partial(make, name)

def make(name='hide_and_seek', num_envs=100):
    '''Madrona wrapper. Tested with Hide and Seek'''
    assert int(num_envs) == float(num_envs), "num_envs must be an integer"
    num_envs = int(num_envs)

    envs = make_madrona(mode='cpu', num_worlds=num_envs)

    envs = MadronaPettingZooEnv(envs, num_envs * 5)
    return pufferlib.emulation.PettingZooPufferEnv(
        env=envs,
        postprocessor_cls=MadronaPostprocessor,
    )

class MadronaPettingZooEnv:
    '''Fakes a multiagent interface to Madrona
    where each env is an agent.'''
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
        self.possible_agents = list(range(1, num_envs+1))
        self.agents = self.possible_agents

    def close(self):
        del self.env

    def observation_space(self, agent):
        return self.env.observation_space

    def action_space(self, agent):
        return self.env.action_space

    def reset(self, seed=None):
        _obs, _ = self.env.reset()
        obs = {}
        batch_size = _obs['agent_data'].shape[0]
        assert(batch_size == self.num_envs)
        for i in range(batch_size):
            obs[i+1] = {k: v[i] for k, v in _obs.items()}
        # obs = {i: o for i, o in enumerate(_obs)}
        info = {i: {'mask': True} for i in obs}
        return obs, info

    def step(self, actions):
        actions = np.array(actions)#[actions[i] for i in range(self.num_envs)])
        _obs, _rewards, _dones, _trunc, _infos = self.env.step(actions)
        batch_size = _obs['agent_data'].shape[0]
        obs = {}
        rewards = {}
        dones = {}
        truncateds = {}
        infos = {}
        for i in range(batch_size):
            obs[i+1] = {k: v[i] for k, v in _obs.items()}
            rewards[i+1] = _rewards[i]
            dones[i+1] = _dones[i]
            truncateds[i+1] = False
            infos[i+1] = {'mask': True}
        # obs = {i: o for i, o in enumerate(obs)}
        # rewards = {i: r for i, r in enumerate(rewards)}
        # dones = {i: bool(d) for i, d in enumerate(dones)}
        # truncateds = {i: False for i in range(len(obs))}
        # infos = {i: {'mask': True} for i in range(len(obs))}
        return obs, rewards, dones, truncateds, infos

class MadronaPostprocessor(pufferlib.emulation.Postprocessor):
    def reset(self, obs):
        self.epoch_return = 0
        self.epoch_length = 0

    def reward_done_truncated_info(self, reward, done, truncated, info):
        if isinstance(reward, (list, np.ndarray)):
            reward = sum(reward) #.values())

        self.epoch_length += 1
        self.epoch_return += reward

        if done or truncated:
            info['return'] = self.epoch_return
            info['length'] = self.epoch_length
            self.epoch_return = 0
            self.epoch_length = 0

        return reward, done, truncated, info


class MadronaHideAndSeekWrapper: #gym.Wrapper):
    def __init__(self, sim, nSeekers=3, nHiders=2):
        # super(MadronaHideAndSeekWrapper, self).__init__(sim)
        self.sim = sim

        self.N = sim.agent_data_tensor().to_torch().shape[0] # = num_worlds * (seekers + hiders)
        self.nSeekers = nSeekers
        self.nHiders = nHiders
        self.max_agents_minus_one = 5 # maxSeekers + maxHiders - 1

        # Define observation space components (using provided shapes)
        # [num_worlds * (seekers + hider), [?, ?, ?, ?, ?], [pos.x, pos.y, vel.x, vel.y]]
        prep_counter = sim.prep_counter_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        agent_type = sim.agent_type_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        agent_data = sim.agent_data_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        box_data = sim.box_data_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        ramp_data = sim.ramp_data_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        visible_agents_mask = sim.visible_agents_mask_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        visible_boxes_mask = sim.visible_boxes_mask_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        visible_ramps_mask = sim.visible_ramps_mask_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        lidar_tensor = sim.lidar_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        done_mask = sim.done_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]

        # Add in an agent ID tensor
        id_tensor = self.get_id_tensor()

        agent_data_space = Box(low=-np.inf, high=np.inf, shape=agent_data.shape[1:], dtype=np.float32)
        # self.relative_agent_obs_space = Box(low=-np.inf, high=np.inf, shape=sim.agent_data_tensor().to_torch().shape[1:], dtype=np.float32)
        relative_box_obs_space = Box(low=-np.inf, high=np.inf, shape=box_data.shape[1:], dtype=np.float32)
        relative_ramp_obs_space = Box(low=-np.inf, high=np.inf, shape=ramp_data.shape[1:], dtype=np.float32)
        visible_agents_mask_space = Box(low=0, high=1, shape=visible_agents_mask.shape[1:], dtype=np.float32)
        visible_boxes_mask_space = Box(low=0, high=1, shape=visible_boxes_mask.shape[1:], dtype=np.float32)
        visible_ramps_mask_space = Box(low=0, high=1, shape=visible_ramps_mask.shape[1:], dtype=np.float32)
        lidar_space = Box(low=0, high=np.inf, shape=lidar_tensor.shape[1:], dtype=np.float32)
        prep_counter_space = Box(low=0, high=np.inf, shape=prep_counter.shape[1:], dtype=np.int32)
        agent_type_mask = Box(low=0, high=1, shape=agent_type.shape[1:], dtype=np.int32)
        # done_space = Box(low=0, high=1, shape=done_mask.shape[1:], dtype=np.int32)
        # agent_mask_space = Box(low=0, high=1, shape=sim.agent_mask_tensor().to_torch().shape[1:], dtype=np.float32)
        # global_positions_space = Box(low=-np.inf, high=np.inf, shape=sim.global_positions_tensor().to_torch().shape[1:], dtype=np.float32)
        
        id_tensor_shape = Box(low=0, high=self.nHiders + self.nSeekers, 
                              shape=id_tensor.shape[1:], dtype=np.int32)

        
        obs_tensors = [
                prep_counter,
                agent_type,
                agent_data,
                lidar_tensor,
                id_tensor,
            ]

        self.num_obs_features = 0 # 53
        for tensor in obs_tensors:
            self.num_obs_features += math.prod(tensor.shape[1:])

        ent_tensors = [
            box_data,
            ramp_data,
        ]

        self.num_ent_features = 0 # 73
        for tensor in ent_tensors:
            self.num_ent_features += math.prod(tensor.shape[1:])

        obs_tensors += ent_tensors
        
        obs_tensors += [
            visible_agents_mask,
            visible_boxes_mask,
            visible_ramps_mask,
        ]

        self.num_mask_features = 0
        for tensor in obs_tensors[-3:]:
            self.num_mask_features += math.prod(tensor.shape[1:])
        
        # Create dictionary observation space
        self.observation_space = Dict({
            # observation data
            "agent_data": agent_data_space,
            "lidar": lidar_space,
            "prep_counter": prep_counter_space,
            'agent_type_mask': agent_type_mask,
            'id_tensor': id_tensor_shape,
            
            # entity data
            "relative_box_obs": relative_box_obs_space,
            "relative_ramp_obs": relative_ramp_obs_space,
            
            # visibility masks from env
            "visible_agents_mask": visible_agents_mask_space,
            "visible_boxes_mask": visible_boxes_mask_space,
            "visible_ramps_mask": visible_ramps_mask_space,

        })

        self.action_space = MultiDiscrete([11, 11, 11, 2, 2])
        # action_space[0] = [-5, 5] move amount distritized
        # action_space[1] = [-5, 5] move angle distritized
        # action_space[2] = [-5, 5] rotation amount distritized
        # action_space[3] = [0, 1]  grab yes/no
        # action_space[4] = [0, 1]  lock yes/no

    def get_id_tensor(self):
        prep_counter = self.sim.prep_counter_tensor().to_torch()[0:self.N * self.max_agents_minus_one, ...]
        id_tensor = torch.arange(self.max_agents_minus_one).float()

        id_tensor = id_tensor.to(device=prep_counter.device)
        id_tensor = id_tensor.view(1, self.max_agents_minus_one).expand(prep_counter.shape[0] // self.max_agents_minus_one, 
                                                                    self.max_agents_minus_one).reshape(
                                                                               prep_counter.shape[0], 1)
        return id_tensor
    
    def to_numpy(self, tensor):
        return tensor.cpu().numpy()
    
    def flatten(self, tensor, keepdim=-1):
        return tensor.view(tensor.shape[0], keepdim)

    def get_obs(self):
        obs = {
            "agent_data": self.sim.agent_data_tensor().to_torch(),
            "relative_box_obs": self.sim.box_data_tensor().to_torch(),
            "relative_ramp_obs": self.sim.ramp_data_tensor().to_torch(),
            "visible_agents_mask": self.sim.visible_agents_mask_tensor().to_torch(),
            "visible_boxes_mask": self.sim.visible_boxes_mask_tensor().to_torch(),
            "visible_ramps_mask": self.sim.visible_ramps_mask_tensor().to_torch(),
            "lidar": self.sim.lidar_tensor().to_torch(),
            "prep_counter": self.sim.prep_counter_tensor().to_torch(),
            'agent_type_mask': self.sim.agent_type_tensor().to_torch(),
            'id_tensor': self.get_id_tensor(),
        }

        for k in obs.keys():
            obs[k] = self.to_numpy(obs[k])
        
        return obs
    
    def reset(self, **kwargs):
        # Reset the simulator is usually done inside the sim.step function
        # however, with the split task graph, we can manually reset it as well.
        # self.sim.reset_and_update() # reset ALL of the batched worlds
        
        # Collect observations and return as a dictionary
        obs = self.get_obs()
        return obs, {}


    def step(self, action_mat):
        # Extract actions from the dictionary
        # move_amount = action_dict["move_amount"]
        # move_angle = action_dict["move_angle"]
        # turn = action_dict["turn"]
        # grab = action_dict["grab"]
        # lock = action_dict["lock"]

        # Get the action tensor from the simulator
        action_tensor = self.sim.action_tensor().to_torch()

        # Fill in the action tensor with the extracted actions
        # move amount, angle and turn are all in [-5, 5] and 
        #  the logits are in [0, 10] so we need to center those properly
        action_mat = torch.from_numpy(action_mat).float()
        action_tensor[..., 0:3] = action_mat[..., 0:3] - 5
        action_tensor[..., 4:] = action_mat[..., 4:]
        
        # copy the action tensor to the simulator
        # action_tensor.copy_(action_tensor)

        # Apply the modified action tensor to the simulator
        # self.sim.setAction(action_tensor)
        self.sim.step()
        # self.sim.prestep()
        # self.sim.poststep()

        # Collect observations, rewards, dones, and info
        obs = self.get_obs()
        reward = self.sim.reward_tensor().to_torch().cpu().numpy()
        done = self.sim.done_tensor().to_torch().cpu().numpy()
        info = {}  # Add any additional info if needed

        return obs, reward, done, done, info
    
    def __del__(self):
        del self.sim
    
class MadronaHideAndSeekWrapperSplitTaskGraph(MadronaHideAndSeekWrapper):
    def __init__(self, sim, nSeekers=3, nHiders=2):
        super().__init__(sim, nSeekers, nHiders)
        
        # intial reward function uses the c++ hide and seek calculated reward
        def reward_fn(agent_data, relative_box_obs, relative_ramp_obs, visible_agents_mask, 
                      visible_boxes_mask, visible_ramps_mask, lidar, prep_counter, agent_type_mask, id_tensor):
                        
            return self.sim.reward_tensor().to_torch()
        
        initial_rf = reward_fn

        self.rfs = [initial_rf]
        self.sampled_reward_fn_id = 0
        self.dynamic_reward_fn = self.rfs[self.sampled_reward_fn_id]

    def step(self, action_dict):
        # Extract actions from the dictionary
        # Get the action tensor from the simulator
        action_tensor = self.sim.action_tensor().to_torch()

        # Fill in the action tensor with the extracted actions
        # move amount, angle and turn are all in [-5, 5] and 
        #  the logits are in [0, 10] so we need to center those properly
        action_mat =  np.stack(action_dict.item().values())
        action_mat = torch.from_numpy(action_mat).float()
        action_tensor[..., 0:3] = action_mat[..., 0:3] - 5
        action_tensor[..., 4:] = action_mat[..., 4:]
        
        # copy the action tensor to the simulator
        # action_tensor.copy_(action_tensor)

        # Apply the modified action tensor to the sim
        self.sim.simulate()
        
        # get the state of the environment post action 
        # (env is updated at the end of simulate, but envs are not reset yet)
        obs = self.get_obs()
        
        # calculate the reward using state + action
        reward = self.dynamic_reward_fn(**obs).cpu().numpy()
        
        # update the environment if finished and collect next states
        self.sim.reset_and_update()

        # Collect observations, rewards, dones, and info
        obs = self.get_obs()
        done = self.sim.done_tensor().to_torch().cpu().numpy()
        info = {}
        
        return obs, reward, done, done, info

    def set_reward_function(self, reward_fn_string, sample=False):
        # parse the generated reward function into a callable 
        # make sure that the function signature matches what we expect:
        # def reward_fn(agent_data, relative_box_obs, relative_ramp_obs, visible_agents_mask, 
        #               visible_boxes_mask, visible_ramps_mask, lidar, prep_counter, agent_type_mask, id_tensor): -> reward
        self.rfs.append(reward_fn_string)
        function = exec(reward_fn_string, globals(), locals())
        self.dynamic_reward_fn = function


def make_madrona(mode='cpu', num_worlds=10, num_steps=2500, entities_per_world=2, 
                 reset_chance=0., nSeekers=3, nHiders=2, split_task_graph=True):

    sim_mode = gpu_hideseek.madrona.ExecMode.CPU if mode == 'cpu' else gpu_hideseek.madrona.ExecMode.GPU
    
    sim = gpu_hideseek.HideAndSeekSimulator(
			exec_mode = sim_mode,
			gpu_id = 0,
			num_worlds = num_worlds,
			sim_flags = gpu_hideseek.SimFlags.Default,
			rand_seed = 10,
			min_hiders = nHiders,
			max_hiders = nHiders,
			min_seekers = nSeekers,
			max_seekers = nSeekers,
            num_pbt_policies = 0,
	)
    sim.init()

    if split_task_graph:
        env = MadronaHideAndSeekWrapperSplitTaskGraph(sim)
    else:
        env = MadronaHideAndSeekWrapper(sim)
    return env


if __name__ == '__main__':
    import torch
    import numpy as np
    import sys
    import time
    import PIL
    import PIL.Image
    torch.manual_seed(0)
    import random
    random.seed(0)

    num_envs = 2000
    num_agents_per_env = 5
    num_frames = 1920

    env = make(name='hide_and_seek', num_envs=num_envs)
    # env = make_madrona()
    # T()

    # print(env.observation_space)
    obs, _ = env.reset()
    start = time.time()
    for _ in range(num_frames):
        action_matrix = [
            np.random.randint(0, 11, num_envs * num_agents_per_env), 
            np.random.randint(0, 11, num_envs * num_agents_per_env), 
            np.random.randint(0, 11, num_envs * num_agents_per_env), 
            np.random.randint(0, 2, num_envs * num_agents_per_env),
            np.random.randint(0, 2, num_envs * num_agents_per_env)
        ]
        action = np.array(action_matrix).T
        action_dict = {i: action[i] for i in range(num_envs * num_agents_per_env)}
        ns, rew, done, tr, i = env.step(action_dict)
    duration = time.time() - start
    print(f"{duration} seconds")
    print(f"{num_envs * num_frames * num_agents_per_env} Total Frames")
    print(f"{num_envs * num_frames * num_agents_per_env / duration} SPS")
    
    # print(ns.keys())

    del env

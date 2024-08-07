import os
import numpy as np
from pathlib import Path
import torch
import gymnasium

import pufferlib
import pufferlib.emulation
import pufferlib.postprocess

from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.env_torch import GPUDriveTorchEnv

#from pygpudrive.env.env_jax import GPUDriveJaxEnv
#from pygpudrive.env.env_numpy import GPUDriveNumpyEnv

EPISODE_LENGTH = 90  # Number of steps in each episode
MAX_NUM_OBJECTS = 32 # Maximum number of objects in the scene we control
NUM_WORLDS = 128 # Number of parallel environments
K_UNIQUE_SCENES = 3 # Number of unique scenes

def env_creator(name='gpudrive'):
    return PufferCPUDrive

# TODO? Not a puffer env?
class PufferCPUDrive(pufferlib.PufferEnv):
    def __init__(self, device='cpu'):
        # Set working directory to the base directory 'gpudrive'
        working_dir = os.path.join(Path.cwd(), '../gpudrive')
        os.chdir(working_dir)

        scene_config = SceneConfig(
            path="data", 
            num_scenes=NUM_WORLDS,
            discipline=SelectionDiscipline.K_UNIQUE_N,
            k_unique_scenes=K_UNIQUE_SCENES,
        )

        env_config = EnvConfig(
            steer_actions = torch.round(
                torch.linspace(-1.0, 1.0, 3), decimals=3),
            accel_actions = torch.round(
                torch.linspace(-3, 3, 3), decimals=3
            )
        )

        render_config = RenderConfig(
            resolution=(512, 512), # Quality of the rendered images
        )

        self.env = GPUDriveTorchEnv(
            config=env_config,
            scene_config=scene_config,
            render_config=render_config,
            max_cont_agents=MAX_NUM_OBJECTS, # Maximum number of agents to control per scene
            device="cpu",
        )

        self.obs_size = self.env.observation_space.shape[-1]
        self.env_id = np.array([i for i in range(NUM_WORLDS*MAX_NUM_OBJECTS)])
        self.device = device

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_size,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.done = False
        self.emulated = None
        self.num_agents = NUM_WORLDS * MAX_NUM_OBJECTS
        self.render_mode = 'rgb_array'

        self.reward = torch.zeros(NUM_WORLDS*MAX_NUM_OBJECTS, dtype=torch.float32).to(self.device)
        self.terminal = torch.zeros(NUM_WORLDS*MAX_NUM_OBJECTS, dtype=torch.bool).to(self.device)
        self.truncated = torch.zeros(NUM_WORLDS*MAX_NUM_OBJECTS, dtype=torch.bool).to(self.device)

        self.controlled_agent_mask = self.env.cont_agent_mask.clone()
        self.live_agent_mask = self.controlled_agent_mask.clone().numpy()
        self.mask = self.live_agent_mask.reshape(NUM_WORLDS*MAX_NUM_OBJECTS)

    def _obs_and_mask(self, obs):
        #self.buf.masks[:] = self.env.cont_agent_mask.numpy().ravel() * self.live_agent_mask
        #return np.asarray(obs).reshape(NUM_WORLDS*MAX_NUM_OBJECTS, self.obs_size)
        #return obs.numpy().reshape(NUM_WORLDS*MAX_NUM_OBJECTS, self.obs_size)[:, :6]
        return obs.view(NUM_WORLDS*MAX_NUM_OBJECTS, self.obs_size)

    def close(self):
        self.env.close()

    def reset(self, seed=None, options=None):
        self.episode_returns = torch.zeros((NUM_WORLDS, MAX_NUM_OBJECTS), dtype=torch.float32).to(self.device)

        self.tick = 0
        obs = self.env.reset()
        return self._obs_and_mask(obs), self.reward, self.terminal, self.truncated, [], self.env_id, self.mask

    def step(self, action):
        action = torch.from_numpy(action).reshape(NUM_WORLDS, MAX_NUM_OBJECTS)
        self.env.step_dynamics(action)
        obs = self._obs_and_mask(self.env.get_obs())
        reward = self.env.get_rewards()
        terminal = self.env.get_dones().bool()

        done_worlds = torch.where(
            (terminal.nan_to_num(0) * self.controlled_agent_mask).sum(dim=1)
             == self.controlled_agent_mask.sum(dim=1)
        )[0]

        self.episode_returns += reward

        self.live_agent_mask[terminal.cpu().numpy()] = 0
        reward = reward.view(NUM_WORLDS*MAX_NUM_OBJECTS)
        terminal = terminal.view(NUM_WORLDS*MAX_NUM_OBJECTS)

        info = []
        if done_worlds.any().item():
            info_tensor = self.env.get_infos()[done_worlds]
            info.append({
                'off_road': info_tensor[:, 0].sum().item(),
                'veh_collisions': info_tensor[:, 1].sum().item(),
                'non_veh_collisions': info_tensor[:, 2].sum().item(),
                'goal_achieved': info_tensor[:, 3].sum().item(),
                'num_finished_agents': self.controlled_agent_mask[done_worlds].sum().item(),
                'mean_reward_per_episode': self.episode_returns[done_worlds].mean().item(),
                'data_density': self.mask.sum() / NUM_WORLDS / MAX_NUM_OBJECTS,
            })

            for idx in done_worlds:
                self.env.sim.reset(idx)
                self.episode_returns[idx] = 0
                self.live_agent_mask[idx] = self.controlled_agent_mask[idx]

        self.tick += 1
        return obs, reward, terminal, self.truncated, info, self.env_id, self.mask

    def render(self, world_render_idx=0):
        return self.env.render(world_render_idx=world_render_idx)


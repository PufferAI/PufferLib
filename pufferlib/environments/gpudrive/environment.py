import os
import numpy as np
from pathlib import Path
import torch
import gymnasium

from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.env_torch import GPUDriveTorchEnv

def env_creator(name='gpudrive'):
    return PufferGPUDrive

class PufferGPUDrive:
    def __init__(self, device='cuda', max_cont_agents=64, num_worlds=64, k_unique_scenes=1):
        self.device = device
        self.max_cont_agents = max_cont_agents
        self.num_worlds = num_worlds
        self.k_unique_scenes = k_unique_scenes
        self.total_agents = max_cont_agents * num_worlds

        # Set working directory to the base directory 'gpudrive'
        working_dir = os.path.join(Path.cwd(), '../gpudrive')
        os.chdir(working_dir)

        scene_config = SceneConfig(
            path='biggest_file/',
            num_scenes=num_worlds,
            discipline=SelectionDiscipline.K_UNIQUE_N,
            k_unique_scenes=k_unique_scenes,
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
            max_cont_agents=max_cont_agents,
            device=device,
        )

        self.obs_size = self.env.observation_space.shape[-1]
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_size,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.done = False
        self.emulated = None
        self.render_mode = 'rgb_array'
        self.num_live = []

        self.controlled_agent_mask = self.env.cont_agent_mask.clone()
        self.obs = self.env.reset()[self.controlled_agent_mask]
        self.num_controlled = self.controlled_agent_mask.sum().item()
        self.num_agents = self.obs.shape[0]
        self.env_id = np.array([i for i in range(self.num_agents)])
        self.mask = np.ones(self.num_agents, dtype=bool)
        self.actions = torch.zeros((num_worlds, max_cont_agents),
            dtype=torch.int64).to(self.device)

    def _obs_and_mask(self, obs):
        #self.buf.masks[:] = self.env.cont_agent_mask.numpy().ravel() * self.live_agent_mask
        #return np.asarray(obs).reshape(NUM_WORLDS*MAX_NUM_OBJECTS, self.obs_size)
        #return obs.numpy().reshape(NUM_WORLDS*MAX_NUM_OBJECTS, self.obs_size)[:, :6]
        return obs.view(self.total_agents, self.obs_size)

    def close(self):
        '''There is no point in closing the env because
        Madrona doesn't close correctly anyways. You will want
        to cache this copy for later use. Cuda errors if you don't'''
        pass 
        #self.env.close()
        #del self.env.sim

    def reset(self, seed=None, options=None):
        self.reward = torch.zeros(self.num_agents, dtype=torch.float32).to(self.device)
        self.terminal = torch.zeros(self.num_agents, dtype=torch.bool).to(self.device)
        self.truncated = torch.zeros(self.num_agents, dtype=torch.bool).to(self.device)

        self.episode_returns = torch.zeros(self.num_agents, dtype=torch.float32).to(self.device)
        self.episode_lengths = torch.zeros(self.num_agents, dtype=torch.float32).to(self.device)
        self.live_agent_mask = torch.ones((self.num_worlds, self.max_cont_agents), dtype=bool).to(self.device)
        return self.obs, self.reward, self.terminal, self.truncated, [], self.env_id, self.mask

    def step(self, action):
        action = torch.from_numpy(action).to(self.device)
        self.actions[self.controlled_agent_mask] = action
        self.env.step_dynamics(self.actions)

        obs = self.env.get_obs()[self.controlled_agent_mask]
        reward = self.env.get_rewards()[self.controlled_agent_mask]
        terminal = self.env.get_dones().bool()

        done_worlds = torch.where(
            (terminal.nan_to_num(0) * self.controlled_agent_mask).sum(dim=1)
             == self.controlled_agent_mask.sum(dim=1)
        )[0].cpu()

        self.episode_returns += reward
        self.episode_lengths += 1
        self.mask = self.live_agent_mask[self.controlled_agent_mask].cpu().numpy()
        self.live_agent_mask[terminal] = 0
        terminal = terminal[self.controlled_agent_mask]

        info = []
        self.num_live.append(self.mask.sum())

        if len(done_worlds) > 0:
            controlled_mask = self.controlled_agent_mask[done_worlds]
            info_tensor = self.env.get_infos()[done_worlds][controlled_mask]
            num_finished_agents = controlled_mask.sum().item()
            info.append({
                'off_road': info_tensor[:, 0].sum().item() / num_finished_agents,
                'veh_collisions': info_tensor[:, 1].sum().item() / num_finished_agents,
                'non_veh_collisions': info_tensor[:, 2].sum().item() / num_finished_agents,
                'goal_achieved': info_tensor[:, 3].sum().item() / num_finished_agents,
                'num_finished_agents': num_finished_agents,
                'episode_length': self.episode_lengths[done_worlds].mean().item(),
                'mean_reward_per_episode': self.episode_returns[done_worlds].mean().item(),
                'control_density': self.num_controlled / self.num_agents,
                'data_density': np.mean(self.num_live) / self.num_agents,
            })

            self.num_live = []
            for idx in done_worlds:
                self.env.sim.reset(idx)
                self.episode_returns[idx] = 0
                self.episode_lengths[idx] = 0
                self.live_agent_mask[idx] = self.controlled_agent_mask[idx]

        return obs, reward, terminal, self.truncated, info, self.env_id, self.mask

    def render(self, world_render_idx=0):
        return self.env.render(world_render_idx=world_render_idx)

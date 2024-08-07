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
MAX_NUM_OBJECTS = 128 # Maximum number of objects in the scene we control
NUM_WORLDS = 16 # Number of parallel environments
K_UNIQUE_SCENES = 3 # Number of unique scenes

def env_creator(name='gpudrive'):
    return PufferCPUDrive

class PufferCPUDrive(pufferlib.PufferEnv):
    def __init__(self):
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

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(6,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.done = False
        self.emulated = None
        self.num_agents = NUM_WORLDS * MAX_NUM_OBJECTS

    def _obs_and_mask(self, obs):
        #self.buf.masks[:] = self.env.cont_agent_mask.numpy().ravel() * self.live_agent_mask
        #return np.asarray(obs).reshape(NUM_WORLDS*MAX_NUM_OBJECTS, self.obs_size)
        return obs.numpy().reshape(NUM_WORLDS*MAX_NUM_OBJECTS, self.obs_size)[:, :6]

    def close(self):
        self.env.close()

    def reset(self, seed=None, options=None):
        self.live_agent_mask = np.ones(NUM_WORLDS*MAX_NUM_OBJECTS, dtype=bool)
        obs = self.env.reset()
        return self._obs_and_mask(obs), {}

    def step(self, action):
        action = torch.from_numpy(action).reshape(NUM_WORLDS, MAX_NUM_OBJECTS)
        #import jax.numpy as jnp
        #action = jnp.asarray(action.reshape(NUM_WORLDS, MAX_NUM_OBJECTS))
        self.env.step_dynamics(action)
        obs = self._obs_and_mask(self.env.get_obs())
        #reward = self.env.get_rewards().numpy().ravel()
        #terminal = self.env.get_dones().numpy().ravel()
        reward = np.asarray(self.env.get_rewards()).ravel()
        terminal = np.asarray(self.env.get_dones()).ravel()
        truncated = np.zeros_like(terminal, dtype=bool)
        info = {}
        self.live_agent_mask = 1 - terminal
        return obs, reward, terminal, truncated, info

    def render(self, world_render_idx=0):
        return self.env.render(world_render_idx=world_render_idx)


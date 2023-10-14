import os
import psutil

import pufferlib


@pufferlib.dataclass
class CleanRL:
    exp_name: str = os.path.basename(__file__).rstrip('.py')
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

@pufferlib.dataclass
class CleanPuffeRL:
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    num_cores: int = 4
    num_buffers: int = 1
    verbose: bool = True
    data_dir: str = 'experiments'
    checkpoint_interval: int = 200
    cpu_offload: bool = True
    selfplay_kernel: list = None
    batch_size: int = 1024
    batch_rows: int = 32
    bptt_horizon: int = 16 #8
    vf_clip_coef: float = 0.1

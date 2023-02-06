from pdb import set_trace as T

import pufferlib
import pufferlib.registry

import test_cleanrl


if __name__ == '__main__':
    # TODO: Make this return dict by default
    bindings = pufferlib.registry.make_atari_bindings()
    bindings = {e.env_name: e for e in bindings}
    bindings = {'BreakoutNoFrameskip-v4': bindings['BreakoutNoFrameskip-v4']}

    for name, binding in bindings.items():
        test_cleanrl.train(
            binding,
            num_cores=8, 
            num_envs=512,
            cuda=True,
            total_timesteps=10_000_000,
            track=True,
            wandb_project_name='pufferlib',
            wandb_entity='jsuarez',
        )

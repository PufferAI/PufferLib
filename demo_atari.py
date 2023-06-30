import torch
from torch.nn import functional as F

from nmmo.entity.entity import EntityState
EntityId = EntityState.State.attr_name_to_col["id"]

import pufferlib
import pufferlib.vectorization
from clean_pufferl import CleanPuffeRL

import pufferlib.registry.atari

device = 'cuda' if torch.cuda.is_available() else 'cpu'

USE_LSTM = False

framestack = 4 if not USE_LSTM else 1

bindings = [
    pufferlib.registry.atari.make_binding('BreakoutNoFrameskip-v4', framestack=framestack),
    pufferlib.registry.atari.make_binding('BeamRiderNoFrameskip-v4', framestack=framestack),
    pufferlib.registry.atari.make_binding('PongNoFrameskip-v4', framestack=framestack),
    pufferlib.registry.atari.make_binding('EnduroNoFrameskip-v4', framestack=framestack),
    pufferlib.registry.atari.make_binding('QbertNoFrameskip-v4', framestack=framestack),
    pufferlib.registry.atari.make_binding('SeaquestNoFrameskip-v4', framestack=framestack),
    pufferlib.registry.atari.make_binding('SpaceInvadersNoFrameskip-v4', framestack=framestack),
]

for binding in bindings:
    # TODO: for the LSTM model, hidden size 128
    agent = pufferlib.frameworks.cleanrl.make_policy(
            pufferlib.registry.atari.Policy,
            recurrent_kwargs={'num_layers': 1 if USE_LSTM else 0}
        )(binding, framestack=framestack).to(device)

    trainer = CleanPuffeRL(binding, agent, batch_size=1024,
            num_buffers=2, num_envs=4, num_cores=4, verbose=False,
            vec_backend=pufferlib.vectorization.multiprocessing.VecEnv)

    trainer.init_wandb()

    data = trainer.allocate_storage()

    num_updates = 10000
    for update in range(trainer.update+1, num_updates + 1):
        trainer.evaluate(agent, data)
        trainer.train(agent, data, batch_rows=64)

# TODO: Figure out why this does not exit cleanly
trainer.close()
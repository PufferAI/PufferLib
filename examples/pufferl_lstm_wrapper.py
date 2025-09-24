import torch
import pufferlib.vector
import pufferlib.ocean
import pufferlib.models as models
from pufferlib import pufferl



# Similar to the pufferrl example
if __name__ == '__main__':
    env_name = 'puffer_breakout'
    env_creator = pufferlib.ocean.env_creator(env_name)
    vecenv = pufferlib.vector.make(env_creator, num_envs=2, num_workers=2, batch_size=1,
        backend=pufferlib.vector.Multiprocessing, env_kwargs={'num_envs': 4096})

    # Wrap a default model with the LSTMWrapper (doesn't have to be the default model)
    policy = models.LSTMWrapper(vecenv.driver_env,models.Default(vecenv.driver_env)).cuda()

    args = pufferl.load_config('default')
    args['train']['env'] = env_name

    # IMPORTANT need to let trainer know to pass state when using LSTMWrapper
    args['train']['use_rnn'] = True

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    for epoch in range(50):
        trainer.evaluate()
        logs = trainer.train()

    trainer.print_dashboard()
    trainer.close()

import torch
import pufferlib.vector
import pufferlib.ocean
from pufferlib import pufferl


# Equivalent to running puffer train puffer_breakout
def cli():
    pufferl.train('puffer_breakout')

# Simple trainer based on pufferl functions
def simple_trainer(env_name='puffer_breakout'):
    args = pufferl.load_config(env_name)

    # You can customize the puffer-provided config
    args['vec']['num_envs'] = 2
    args['env']['num_envs'] = 2048
    args['policy']['hidden_size'] = 256
    args['rnn']['input_size'] = 256
    args['rnn']['hidden_size'] = 256
    args['train']['total_timesteps'] = 10_000_000
    args['train']['learning_rate'] = 0.03

    # Or, you can create and use a separate config file
    # args = pufferl.load_config_file(<YOUR_OWN_CONFIG.ini>, fill_in_default=True)

    vecenv = pufferl.load_env(env_name, args)
    policy = pufferl.load_policy(args, vecenv, env_name)

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        logs = trainer.train()

    trainer.print_dashboard()
    trainer.close()

class Policy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(env.single_observation_space.shape[0], 128)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(128, 128)),
        )
        self.action_head = torch.nn.Linear(128, env.single_action_space.n)
        self.value_head = torch.nn.Linear(128, 1)

    def forward_eval(self, observations, state=None):
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

# Managing your own trainer
if __name__ == '__main__':
    env_name = 'puffer_breakout'
    env_creator = pufferlib.ocean.env_creator(env_name)
    vecenv = pufferlib.vector.make(env_creator, num_envs=2, num_workers=2, batch_size=1,
        backend=pufferlib.vector.Multiprocessing, env_kwargs={'num_envs': 4096})
    policy = Policy(vecenv.driver_env).cuda()
    args = pufferl.load_config('default')
    args['train']['env'] = env_name

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    for epoch in range(10):
        trainer.evaluate()
        logs = trainer.train()

    trainer.print_dashboard()
    trainer.close()

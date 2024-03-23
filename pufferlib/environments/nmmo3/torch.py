import pufferlib.models

from torch import nn


class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

'''
class Policy(pufferlib.models.ProcgenResnet):
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__(
            env=env,
            cnn_width=cnn_width,
            mlp_width=mlp_width,
        )
'''

class Policy(pufferlib.models.Policy):
    def __init__(self, env, *args, framestack):
        '''
        , flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''
        super().__init__(env)

        self.num_actions = self.action_space.n
        hidden_size = 256
        output_size = 256

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(1, 32, 5, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 5, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(320, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def encode_observations(self, observations):
        x = pufferlib.emulation.unpack_batched_obs(observations, self.unflatten_context)
        x = x['map'].unsqueeze(1).float()
        return self.network(x), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value


'''
class Policy(pufferlib.models.ProcgenResnet):
    def __init__(self, env, cnn_width=16, mlp_width=512):
        super().__init__(
            env=env,
            cnn_width=cnn_width,
            mlp_width=mlp_width,
        )
'''

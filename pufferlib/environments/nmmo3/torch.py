import pufferlib.models

import torch
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
            pufferlib.pytorch.layer_init(nn.Conv2d(128, 32, 3, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(192, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)


    def decode_map_one_hot(self, codes):
        ob1 = nn.functional.one_hot((codes % 17).long(), 17)
        codes //= 17
        ob2 = nn.functional.one_hot((codes % 81).long(), 81)
        codes //= 81
        ob3 = nn.functional.one_hot((codes % 3).long(), 3)
        codes //= 3
        ob4 = nn.functional.one_hot((codes % 5).long(), 5)
        codes //= 5
        ob5 = nn.functional.one_hot((codes % 5).long(), 5)
        codes //= 5
        ob6 = nn.functional.one_hot((codes % 6).long(), 6)
        codes //= 6
        ob7 = nn.functional.one_hot((codes % 7).long(), 7)
        codes //= 7
        ob8 = nn.functional.one_hot(codes.long(), 4)

        obs = torch.cat([ob1, ob2, ob3, ob4, ob5, ob6, ob7, ob8], dim=-1)
        obs = torch.permute(obs, (0, 3, 1, 2))
        return obs.float()

    def decode_map_fast(self, codes):
        codes = codes.unsqueeze(1).long()
        obs = torch.zeros(codes.shape[0], 128, 11, 15, device='cuda', dtype=int)

        add, div = 0, 1
        factors = [17, 81, 3, 5, 5, 6, 7, 4]
        for mod in factors:
            obs.scatter_(1, add+(codes//div)%mod, 1)
            add += mod
            div *= mod

        return obs.float().detach()
 
    def encode_observations(self, observations):
        x = pufferlib.emulation.unpack_batched_obs(observations, self.unflatten_context)
        ob_map = self.decode_map_fast(x['map'])
        return self.network(ob_map), None

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

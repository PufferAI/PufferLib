import pufferlib.models


class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class Policy(pufferlib.models.ProcgenResnet):
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__(
            env=env,
            cnn_width=cnn_width,
            mlp_width=mlp_width,
        )

'''
class Policy(pufferlib.models.ProcgenResnet):
    def __init__(self, env, cnn_width=16, mlp_width=512):
        super().__init__(
            env=env,
            cnn_width=cnn_width,
            mlp_width=mlp_width,
        )
'''

import pufferlib.models


class Policy(pufferlib.models.Default):
    def __init__(self, env, input_size=64, hidden_size=64):
        super().__init__(env, input_size, hidden_size)

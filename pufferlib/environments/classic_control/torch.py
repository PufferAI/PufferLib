import pufferlib.models


class Policy(pufferlib.models.Default):
    def __init__(self, env, hidden_size=64):
        super().__init__(env, hidden_size)

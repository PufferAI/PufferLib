from pdb import set_trace as T

import torch

import pufferlib.vectorization.serial
import pufferlib.vectorization.multiprocessing

class Default:
    emulate_const_horizon = 1024

    vec_backend = pufferlib.vectorization.serial.VecEnv
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_cores = 4
    num_buffers = 2
    num_envs = 4
    batch_size = 1024
    batch_rows = 256
    bptt_horizon = 1
    seed = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy_args = [128, 128]
    policy_kwargs = {}
    recurrent_args = []
    #recurrent_args = [128, 128]
    recurrent_kwargs = dict(num_layers=0)

    pool_rank_interval=1
    pool_update_policy_interval=1
    pool_add_policy_interval=1

    @property
    def make_binding(self):
        return self.registry.make_binding

    @property
    def Policy(self):
        return self.registry.Policy

class Atari(Default):
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=1, flat_size=64*7*7)
    recurrent_args = [512, 128]

    def __init__(self, framestack):
        import pufferlib.registry.atari
        self.registry = pufferlib.registry.atari
        self.all_bindings = [
            self.make_binding('BreakoutNoFrameskip-v4', framestack=framestack),
            self.make_binding('PongNoFrameskip-v4', framestack=framestack),
        ]

class Avalon(Default):
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=1, flat_size=64*7*7)
    recurrent_args = [512, 128]

    def __init__(self, framestack):
        import pufferlib.registry.avalon
        self.registry = pufferlib.registry.avalon
        self.all_bindings = [self.make_binding()]

class Box2d(Default):
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=1, flat_size=64*7*7)
    recurrent_args = [512, 128]

    def __init__(self):
        import pufferlib.registry.box2d
        self.registry = pufferlib.registry.box2d
        self.all_bindings = [self.make_binding()]

class Butterfly(Default):
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=1, flat_size=64*7*7)
    recurrent_args = [512, 128]

    def __init__(self, framestack):
        import pufferlib.registry.butterfly
        self.registry = pufferlib.registry.butterfly
        self.all_bindings = [
            self.registry.make_cooperative_pong_v5_binding(),
        ]

class Crafter(Default):
    # Framestack 3 is a hack for RGB
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=3, flat_size=64*4*4)

    def __init__(self):
        import pufferlib.registry.crafter
        self.registry = pufferlib.registry.crafter
        self.all_bindings = [self.make_binding()]

class DMControl(Default):
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=3, flat_size=64*4*4)

    def __init__(self):
        import pufferlib.registry.dmc
        self.registry = pufferlib.registry.dmc
        self.all_bindings = [self.make_binding()]

class DMLab(Default):
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=3, flat_size=64*4*4)

    def __init__(self):
        import pufferlib.registry.dm_lab
        self.registry = pufferlib.registry.dm_lab
        self.all_bindings = [self.make_binding()]

class Griddly(Default):
    def __init__(self):
        import pufferlib.registry.griddly
        self.registry = pufferlib.registry.griddly
        self.all_bindings = [self.registry.make_spider_v0_binding()]

class MAgent(Default):
    # Framestack 5 is a hack for obs channels
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=5, flat_size=64*4*4)

    def __init__(self):
        import pufferlib.registry.magent
        self.registry = pufferlib.registry.magent
        self.all_bindings = [self.registry.make_battle_v4_binding()]

class MicroRTS(Default):
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=5, flat_size=64*4*4)

    def __init__(self):
        import pufferlib.registry.microrts
        self.registry = pufferlib.registry.microrts
        self.all_bindings = [self.registry.make_env()]

class MineRL(Default):
    policy_kwargs = dict(input_size=512, hidden_size=128, output_size=128, framestack=5, flat_size=64*4*4)

    def __init__(self):
        import pufferlib.registry.minecraft
        self.registry = pufferlib.registry.minecraft
        self.all_bindings = [self.registry.make_env()]

class NetHack(Default):
    policy_args = []
    policy_kwargs = dict(embedding_dim=32, crop_dim=9, num_layers=5)
    recurrent_args = [512, 512]

    def __init__(self):
        import pufferlib.registry.nethack
        self.registry = pufferlib.registry.nethack
        self.all_bindings = [self.make_binding()]

class NMMO(Default):
    batch_size = 2**14
    batch_rows = 128

    def __init__(self) -> None:
        import pufferlib.registry.nmmo
        self.registry = pufferlib.registry.nmmo
        self.all_bindings = [self.make_binding()]

class SMAC(Default):
    policy_args = []
    policy_kwargs = dict(embedding_dim=32, crop_dim=9, num_layers=5)
    recurrent_args = [512, 512]

    def __init__(self):
        import pufferlib.registry.smac
        self.registry = pufferlib.registry.smac
        self.all_bindings = [self.make_binding()]


all = {
    'atari': Atari,
    'avalon': Avalon,
    'box2d': Box2d,
    'butterfly': Butterfly,
    'crafter': Crafter,
    'dm_control': DMControl,
    'dm_lab': DMLab,
    'griddly': Griddly,
    'magent': MAgent,
    'microrts': MicroRTS,
    'minerl': MineRL,
    'nethack': NetHack,
    'nmmo': NMMO,
    'smac': SMAC,
}

# Possible stuff to add support:
# Deep RTS
# https://github.com/kimbring2/MOBA_RL
# Serpent AI
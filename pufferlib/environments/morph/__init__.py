from .environment import env_creator

try:
    import pufferlib.environments.morph.policy as torch
except ImportError:
    pass
else:
    from pufferlib.environments.morph.policy import PHCPolicy, LSTMCriticPolicy
    try:
        from pufferlib.environments.morph.policy import Recurrent
    except:
        Recurrent = None

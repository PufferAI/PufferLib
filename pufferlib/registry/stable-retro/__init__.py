from .environment import env_creator, make_env

try:
    import torch
except ImportError:
    pass
else:
    from .torch import Policy
    Recurrent = getattr(Policy, 'Recurrent', None)

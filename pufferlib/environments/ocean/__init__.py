from .environment import env_creator

try:
    import torch
except ImportError:
    pass
else:
    from .torch import Policy
    try:
        from .torch import RNN
    except:
        RNN = None

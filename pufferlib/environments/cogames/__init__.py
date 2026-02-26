"""CoGames integration package."""

from .environment import env_creator, make

try:
    import torch
    from .torch import Policy, Recurrent
except ImportError:
    pass

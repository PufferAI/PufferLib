from .environment import env_creator, make_env

try:
    from .torch import Policy
    Recurrent = getattr(Policy, 'Recurrent', None)
except ImportError:
    pass

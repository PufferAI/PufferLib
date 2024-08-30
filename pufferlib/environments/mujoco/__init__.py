from .environment import env_creator

try:
    # NOTE: demo.py looks the policy class from the torch module
    import pufferlib.environments.mujoco.policy as torch
except ImportError:
    pass
else:
    from .policy import Policy
    try:
        from .policy import Recurrent
    except:
        Recurrent = None
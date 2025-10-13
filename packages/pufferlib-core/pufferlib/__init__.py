"""
PufferLib Core - Minimal vectorized environment functionality
"""

import sys

# Import individual modules with delayed loading to avoid circular imports
def _import_modules():
    from . import spaces
    from . import pufferlib

    # Temporarily add pufferlib to the current module namespace to resolve imports
    current_module = sys.modules[__name__]
    current_module.PufferEnv = pufferlib.PufferEnv
    current_module.set_buffers = pufferlib.set_buffers
    current_module.unroll_nested_dict = pufferlib.unroll_nested_dict
    current_module.APIUsageError = pufferlib.APIUsageError

    from . import emulation
    from . import vector

    # Try to import C extensions if available
    try:
        from . import _C
        current_module._C = _C
    except ImportError:
        # C extensions not available, continue without them
        pass

    # Import PyTorch modules (now required dependencies)
    from . import pytorch
    from . import models
    current_module.pytorch = pytorch
    current_module.models = models
    pytorch_modules = [pytorch, models]

    # Import pufferl module
    from . import pufferl
    current_module.pufferl = pufferl

    return spaces, pufferlib, emulation, vector, pytorch_modules, pufferl

# Perform the imports
spaces, pufferlib, emulation, vector, pytorch_modules, pufferl = _import_modules()

__version__ = "3.0.3"
__all__ = ["spaces", "emulation", "vector", "pufferlib", "pytorch", "models", "pufferl"]

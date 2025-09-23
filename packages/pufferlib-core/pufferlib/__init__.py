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

    from . import emulation
    from . import vector

    # Try to import C extensions if available
    try:
        from . import _C
        current_module._C = _C
    except ImportError:
        # C extensions not available, continue without them
        pass

    # Try to import PyTorch modules if available (optional dependency)
    pytorch_modules = []
    try:
        from . import pytorch
        from . import models
        current_module.pytorch = pytorch
        current_module.models = models
        pytorch_modules = [pytorch, models]
    except ImportError:
        # PyTorch not available, continue without it
        pass

    return spaces, pufferlib, emulation, vector, pytorch_modules

# Perform the imports
spaces, pufferlib, emulation, vector, pytorch_modules = _import_modules()

__version__ = "3.0.3"
__all__ = ["spaces", "emulation", "vector", "pufferlib"]

# Add pytorch and models to __all__ if they are available
if pytorch_modules:
    __all__.extend(["pytorch", "models"])

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

    from . import emulation
    from . import vector

    return spaces, pufferlib, emulation, vector

# Perform the imports
spaces, pufferlib, emulation, vector = _import_modules()

__version__ = "3.0.3"
__all__ = ["spaces", "emulation", "vector", "pufferlib"]

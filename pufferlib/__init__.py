__version__ = 3.0

import os
path = __path__[0]
link_to = os.path.join(path, 'resources')
try:
    os.symlink(link_to, 'resources')
except FileExistsError:
    pass

# Silence noisy dependencies
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import torch
    from torch.utils import _triton as _torch_triton
except Exception:
    _torch_triton = None
else:
    if _torch_triton is not None and not getattr(_torch_triton, "_pufferlib_safe_triton_patch", False):
        original_cuda_extra_check = getattr(_torch_triton, "cuda_extra_check", None)
        original_has_triton = getattr(_torch_triton, "has_triton", None)

        if callable(original_cuda_extra_check):
            def _cuda_extra_check_guard(device_interface):
                try:
                    return original_cuda_extra_check(device_interface)
                except IndexError:
                    return False

            _torch_triton.cuda_extra_check = _cuda_extra_check_guard

        if callable(original_has_triton):
            def _has_triton_guard():
                try:
                    return original_has_triton()
                except IndexError:
                    return False

            _torch_triton.has_triton = _has_triton_guard

        _torch_triton._pufferlib_safe_triton_patch = True

# Silence noisy packages
import sys
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
try:
    import gymnasium
    import pygame
except ImportError:
    pass
sys.stdout.close()
sys.stderr.close()
sys.stdout = original_stdout
sys.stderr = original_stderr

from pufferlib.pufferlib import *
from pufferlib import environments

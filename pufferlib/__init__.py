__version__ = 4.0

import os as _os
import sys as _sys

if _sys.platform == 'win32':
    # The _C extension links against CUDA/cuDNN/libomp DLLs that are not on
    # the default search path. Register their directories before any import.
    def _add_dll_dir(path):
        if path and _os.path.isdir(path):
            _os.add_dll_directory(path)

    _cuda = _os.environ.get('CUDA_PATH') or _os.environ.get('CUDA_HOME')
    if _cuda:
        _add_dll_dir(_os.path.join(_cuda, 'bin'))

    # NVIDIA pip wheels (nvidia-cudnn-cu12 etc.) ship runtime DLLs in bin/
    for _mod in ('cudnn', 'cublas', 'cusolver', 'curand'):
        try:
            _m = __import__(f'nvidia.{_mod}', fromlist=['__path__'])
            _add_dll_dir(_os.path.join(_m.__path__[0], 'bin'))
        except ImportError:
            pass

    # libomp.dll (and any DLLs copied next to the extension)
    _add_dll_dir(_os.path.dirname(_os.path.abspath(__file__)))

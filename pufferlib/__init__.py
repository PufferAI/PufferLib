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

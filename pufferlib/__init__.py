from pufferlib import version
__version__ = version.__version__

import os
import sys

# Silence noisy dependencies
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Silence noisy packages
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

from pufferlib.namespace import namespace, dataclass, Namespace
from pufferlib import frameworks, environments
from pufferlib.environment import PufferEnv

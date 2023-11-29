import warnings


# Shut deepmind_lab up
warnings.filterwarnings("ignore", category=DeprecationWarning)#, module="deepmind_lab")
try:
    from deepmind_lab import dmenv_module  # Or whatever the actual module is
except ImportError:
    pass

import os
import sys

# Shut pygame up
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    import pygame
except ImportError:
    pass
sys.stdout.close()
sys.stdout = original_stdout


from pufferlib.namespace import namespace, dataclass
from pufferlib import frameworks, environments

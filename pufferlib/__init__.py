__version__ = 4.0

import os
path = __path__[0]
link_to = os.path.join(path, 'resources')
try:
    os.symlink(link_to, 'resources')
except FileExistsError:
    pass

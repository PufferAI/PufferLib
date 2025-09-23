# PufferLib Core

Minimal PufferLib core functionality with vectorized environments.

This package contains only the essential components:
- `spaces`: Observation/action space handling
- `emulation`: Environment compatibility layer for Gym/Gymnasium/PettingZoo
- `vector`: Vectorized environment implementations

For the full PufferLib with training capabilities and environments, see the main `pufferlib` package.

## Installation

### Basic Installation (Python-only)
```bash
pip install pufferlib-core
```

### Installation with C++/CUDA Extensions
To enable the `pufferlib._C` extensions (required for advanced features like CUDA advantage computation):

```bash
# First install torch
pip install torch

# Then install with extensions
PUFFERLIB_BUILD_EXT=1 pip install pufferlib-core[ext]
```

Or you can install the dependencies and build in separate steps:
```bash
# Install with extra dependencies
pip install pufferlib-core[ext]

# Then rebuild with extensions
PUFFERLIB_BUILD_EXT=1 pip install --upgrade --force-reinstall --no-deps pufferlib-core
```

## Usage

After installation with extensions, you should be able to import the C extensions:

```python
import pufferlib
from pufferlib import _C  # This will only work if extensions were built
```

## Development

To build extensions in development mode:
```bash
PUFFERLIB_BUILD_EXT=1 python setup.py build_ext --inplace
```
"""
Tribal Village Environment PufferLib Integration.

Prefers a local tribal-village checkout for rapid iteration; falls back to the
installed package if not present.
"""

import functools
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pufferlib


def _import_tribal_village_env():
    """Prefer local tribal-village checkout if present; else import package."""
    repo_root = Path(__file__).resolve().parents[2]
    fallback_dir = repo_root.parent / 'tribal-village'
    if fallback_dir.exists():
        if str(fallback_dir) not in sys.path:
            sys.path.insert(0, str(fallback_dir))
        try:
            from tribal_village_env.environment import TribalVillageEnv  # type: ignore
            return TribalVillageEnv
        except ImportError:
            pass

    try:
        from tribal_village_env.environment import TribalVillageEnv  # type: ignore
        return TribalVillageEnv
    except ImportError as exc:
        raise ImportError("""Failed to import tribal-village environment. Install the package with
  pip install pufferlib[tribal-village] --no-build-isolation
or keep a checkout at ../tribal-village containing tribal_village_env/.""") from exc


def env_creator(name='tribal_village'):
    return functools.partial(make, name=name)


def make(name='tribal_village', config=None, buf=None, **kwargs):
    """Create a tribal village PufferLib environment instance."""
    TribalVillageEnv = _import_tribal_village_env()

    # Merge config with kwargs
    if config is None:
        config = {}
    config = {**config, **kwargs}

    # Create the environment
    env = TribalVillageEnv(config=config, buf=buf)
    return env

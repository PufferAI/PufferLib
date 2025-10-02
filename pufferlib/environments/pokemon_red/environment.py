from pdb import set_trace as T

import gymnasium
import functools

from pokegym import Environment

import pufferlib.emulation
import pufferlib.postprocess


def env_creator(name='pokemon_red'):
    return functools.partial(make, name)

def make(name, headless: bool = True, state_path=None, buf=None, seed=0):
    '''Pokemon Red'''
    env = Environment(headless=headless, state_path=state_path)
    # --- infer observation_space from the raw screen ndarray (no reset needed) ---
    try:
        if getattr(env, 'observation_space', None) is None and hasattr(env, 'screen'):
            arr = env.screen.screen_ndarray()
            import numpy as _np
            from gymnasium import spaces as _spaces
            _dtype = arr.dtype if getattr(arr, 'dtype', None) is not None and arr.dtype.kind in 'iu' else _np.uint8
            _low  = 0 if _dtype.kind in 'iu' else _np.finfo(_dtype).min
            _high = 255 if _dtype.kind in 'iu' else _np.finfo(_dtype).max
            env.observation_space = _spaces.Box(low=_low, high=_high, shape=tuple(arr.shape), dtype=_dtype)
    except Exception:
        # best-effort only
        pass
    # --- infer observation_space from raw env BEFORE wrappers ---
    # Some wrappers (e.g., gymnasium RecordEpisodeStatistics) call reset(seed=...) which
    # the raw pokegym.Environment.reset may not accept. Call reset() on the raw env to
    # obtain a sample observation and set observation_space if missing.
    try:
        if getattr(env, 'observation_space', None) is None:
            sample = env.reset()
            if isinstance(sample, tuple) and len(sample) >= 1:
                obs = sample[0]
            else:
                obs = sample
            import numpy as _np
            from gymnasium import spaces as _spaces
            if hasattr(obs, 'shape') and hasattr(obs, 'dtype'):
                if _np.issubdtype(obs.dtype, _np.floating):
                    low = _np.finfo(obs.dtype).min
                    high = _np.finfo(obs.dtype).max
                elif _np.issubdtype(obs.dtype, _np.integer):
                    low = _np.iinfo(obs.dtype).min
                    high = _np.iinfo(obs.dtype).max
                else:
                    low = 0
                    high = 1
                env.observation_space = _spaces.Box(
                    low=low, high=high, shape=tuple(obs.shape), dtype=_np.dtype(obs.dtype)
                )
    except Exception:
        # best-effort only
        pass

    env = RenderWrapper(env)    # Episode statistics wrapper: prefer pufferlib.postprocess.EpisodeStats if available; fallback to Gymnasium
    EpisodeStats = getattr(getattr(pufferlib, "postprocess", object()), "EpisodeStats", None)
    if EpisodeStats is None:
        try:
            from gymnasium.wrappers import RecordEpisodeStatistics as EpisodeStats
        except Exception:
            EpisodeStats = None
    if EpisodeStats is not None:
        env = EpisodeStats(env)

        # Prefer PufferLib's postprocess wrapper if available; otherwise, fall back to Gymnasium
        try:
            EpisodeStats = getattr(pufferlib.postprocess, "EpisodeStats", None) or \
                           getattr(pufferlib.postprocess, "RecordEpisodeStatistics", None)
        except Exception:
            EpisodeStats = None

        if EpisodeStats is not None:
            env = EpisodeStats(env)
        else:
            try:
                from gymnasium.wrappers import RecordEpisodeStatistics
            except Exception as e:
                raise ImportError("Neither pufferlib.postprocess EpisodeStats nor gymnasium.wrappers.RecordEpisodeStatistics is available") from e
            env = RecordEpisodeStatistics(env)
    
    # --- ensure the underlying env defines an observation_space ---
    # Some upstream envs only provide observations after reset(); do a best-effort
    # inference and attach a Box so downstream emulation can proceed.
    try:
        if getattr(env, 'observation_space', None) is None:
            sample = env.reset()
            if isinstance(sample, tuple) and len(sample) >= 1:
                obs = sample[0]
            else:
                obs = sample
            import numpy as _np
            from gymnasium import spaces as _spaces
            if hasattr(obs, 'shape') and hasattr(obs, 'dtype'):
                if _np.issubdtype(obs.dtype, _np.floating):
                    low = _np.finfo(obs.dtype).min
                    high = _np.finfo(obs.dtype).max
                elif _np.issubdtype(obs.dtype, _np.integer):
                    low = _np.iinfo(obs.dtype).min
                    high = _np.iinfo(obs.dtype).max
                else:
                    low = 0
                    high = 1
                env.observation_space = _spaces.Box(low=low, high=high, shape=tuple(obs.shape), dtype=_np.dtype(obs.dtype))
    except Exception:
        # best-effort only — fall back to original behavior if inference fails
        pass

    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, seed=seed)

class RenderWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        # Properly initialize the Gymnasium.Wrapper base class
        super().__init__(env)
        # Forward spaces if the inner env already defines them
        if hasattr(env, 'observation_space'):
            self.observation_space = env.observation_space
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        return self.env.screen.screen_ndarray()

    def reset(self, *args, **kwargs):
        """Accept gym's seed/options and gracefully fall back for pokegym.Environment."""
        try:
            return self.env.reset(*args, **kwargs)
        except TypeError:
            kwargs.pop('seed', None)
            kwargs.pop('options', None)
            try:
                return self.env.reset(*args, **kwargs)
            except TypeError:
                return self.env.reset()

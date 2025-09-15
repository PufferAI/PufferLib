"""
grid_flappy_puffer.py
Initial port stub to PufferLib native Python binding + perf harness.

/* UNVERIFIED: I cannot verify this API exists. Replace the PufferLib constructor
   or import with the actual PufferLib binding after checking their docs. */
"""
import time
import numpy as np

# Try to import PufferLib binding first (UNVERIFIED)
try:
    # /* UNVERIFIED: replace with actual pufferlib python import */
    from puffer import GridFlappy as GridFlappyPuffer  # /* UNVERIFIED: I cannot verify this API exists */
    _USE_PUFFER = True
except Exception:
    _USE_PUFFER = False

# Fallback: import original reference env if present
try:
    from envs.grid_flappy import GridFlappy as GridFlappyOriginal  # keep for parity testing
except Exception:
    GridFlappyOriginal = None

# Minimal wrapper class if puffer binding exists
if _USE_PUFFER:
    class GridEnv:
        def __init__(self, seed: int = 0):
            # /* UNVERIFIED: constructor args may differ */
            try:
                self._env = GridFlappyPuffer(seed=seed)  # /* UNVERIFIED */
            except Exception as e:
                raise RuntimeError("Could not construct Puffer GridFlappy: " + str(e))
        def reset(self):
            return self._env.reset()
        def step(self, a):
            out = self._env.step(a)
            # Accept either Gym or Gymnasium return signatures
            if len(out) == 4:
                obs, reward, done, info = out
                return obs, reward, done, info
            elif len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = terminated or truncated
                return obs, reward, done, info
            else:
                raise RuntimeError("Unexpected step() return signature: " + repr(out))
else:
    class GridEnv:
        def __init__(self, seed: int = 0):
            if GridFlappyOriginal is None:
                raise RuntimeError("Original env not found in envs.grid_flappy")
            # Construct original env (assumed signature)
            self._env = GridFlappyOriginal(seed=seed)
        def reset(self):
            return self._env.reset()
        def step(self, a):
            out = self._env.step(a)
            if len(out) == 4:
                return out
            elif len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = terminated or truncated
                return obs, reward, done, info
            else:
                raise RuntimeError("Unexpected step() return signature: " + repr(out))

# Perf harness at bottom
if __name__ == "__main__":
    n_steps = 200000
    env = None
    try:
        env = GridEnv(seed=1234)
    except Exception as e:
        print("Failed to construct GridEnv:", e)
        raise SystemExit(1)

    obs = env.reset()
    start = time.time()
    for i in range(n_steps):
        # deterministic simple policy: always '0' (replace as needed)
        a = 0
        o, r, done, info = env.step(a)
        if done:
            obs = env.reset()
    elapsed = time.time() - start
    steps_per_sec = n_steps / elapsed if elapsed > 0 else float("inf")
    print(f"Ran {n_steps} steps in {elapsed:.2f}s -> {steps_per_sec:.2f} steps/sec")

# Craftax Full Ocean Port Notes

## Current Implementation

`ocean/craftax/` is wired as a full Craftax Ocean environment with the correct
symbolic observation size (`8268`) and action count (`43`). The C header declares
the full Craftax enum set and an `EnvState`-shaped C struct matching the field
order in `craftax_state.py`.

Reset and step are currently reference-backed. The C env acquires the Python GIL,
calls the installed JAX `Craftax-Symbolic-v1` implementation, and copies the
resulting float32 observation, reward, terminal flag, and terminal achievement
log into PufferLib-owned buffers.

## Deliberate Divergences From The Requested Native Port

- The Craftax game logic is not yet native C. World generation, step logic,
  achievements, rewards, and auto-reset behavior are delegated to the JAX oracle.
- JAX threefry PRNG has not been ported to C. The proxy uses the same JAX key
  schedule as `tests/craftax_parity.py`: `split(PRNGKey(seed))` for reset, then
  one `split` per action.
- Fractal noise has not been ported to C. It is still executed by the JAX world
  generator.
- `c_step` allocates through Python/JAX and serializes on the GIL. This violates
  the final performance target and the intended no-allocation step path.
- `c_close` asks the proxy to drop JAX arrays, then intentionally leaks the small
  Python proxy wrapper objects. DECREFing JAX/XLA-owned wrappers during
  PufferLib shutdown segfaulted in the proxy baseline; the native port removes
  this path.
- Rendering is a no-op.
- `config/ocean/craftax.ini` uses a small proxy-friendly vector size. The native
  port should raise this once step no longer calls Python.

## Known Risks

- Training throughput is expected to be poor. This baseline is for parity and ABI
  validation, not for the Ryzen 9950X3D optimization target.
- `uv run puffer train craftax` currently reaches rollout/train work, but a
  128-step smoke run exits with code 139 during shutdown. The parity harness and
  direct `VecEnv` close path exit cleanly; this appears specific to the GPU
  trainer plus proxy/JAX runtime cleanup.
- The helper forces `JAX_PLATFORM_NAME=cpu` before importing JAX to avoid using
  the shared GPU from inside environment steps.
- `build.sh` now embeds rpaths for wheel-provided CUDA libraries so
  `pufferlib._C` can find `libnccl.so.2`. The parity harness still preloads NCCL
  defensively for older local builds.

## Next Native Port Steps

1. Replace the proxy reset path with native world generation, including
   `util/noise.py` and JAX key-compatible threefry.
2. Replace one step subsystem at a time with native logic and keep the proxy as a
   local oracle until each subsystem matches.
3. Remove Python/JAX calls from `c_step`, restore large vector sizes, then measure
   CPU throughput before optimizing observation encoding, mob updates, and light
   propagation.

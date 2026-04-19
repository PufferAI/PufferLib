# Craftax Full Ocean Port Notes

## 2026-04-18 Standalone Simple Step Subsystems

This phase adds native C ports for the easy step subsystems, but deliberately
does not integrate them into `c_step`. The live Ocean environment still delegates
step to the Python/JAX proxy, so the full parity harness should remain unchanged.

- `step_simple.h` contains standalone in-place helpers for:
  - `move_player`
  - `update_plants`
  - `boss_logic`
  - `level_up_attributes`
  - `clip_inventory_and_intrinsics`
  - `calculate_inventory_achievements`
  - `update_player_intrinsics`
  - `drink_potion`
  - `read_book`
- `tests/craftax_state_fixtures.py` provides test-only pickle payloads for JAX
  `EnvState` values, a ctypes mirror of `CraftaxState`, C-to-JAX conversion, and
  strict state diffing with exact integer/bool checks and `atol=1e-6` float
  checks.
- `tests/craftax_step_subsystem_test.py` builds a temporary C wrapper around the
  inline helpers and compares each subsystem against the JAX function on copied
  reset-plus-step-through states for 16 seeds and targeted stress cases.
- The helpers do not allocate, do not call Python, and keep JAX details that
  matter for these routines, including clamped gather-style indexing, `where` and
  `select` ordering, potion `-1` indexing, and the `read_book` split plus
  probability-choice path.

Native-step roadmap checklist:

- [x] Native reset PRNG, noise, 9-floor world generation, and reset observation.
- [x] Standalone native simple step subsystems with JAX-parity tests.
- [ ] Standalone native ports for hard action subsystems: `do_action`,
  `do_crafting`, `place_block`, `shoot_projectile`, `cast_spell`, `enchant`,
  `change_floor`, `add_items_from_chest`, `update_mobs`, and `spawn_mobs`.
- [ ] Native reward, terminal, timestep, light-level, RNG, and achievement-delta
  bookkeeping around the subsystem calls.
- [ ] Integrate all green subsystem ports into a native `c_step` behind one
  explicit switch, then remove the Python/JAX proxy from the normal step path.
- [ ] Restore production vector sizes in `config/ocean/craftax.ini` after native
  step is the default.
- [ ] Benchmark CPU throughput only after the proxy path is gone.

## 2026-04-18 Native 9-Floor Reset Worldgen

This phase replaces the JAX reset call with native C reset world generation for
the default `Craftax-Symbolic-v1` environment parameters.

- `worldgen.h` now mirrors `generate_world` for all nine floors:
  - floor 0 overworld smoothworld
  - floor 1 dungeon
  - floor 2 gnomish mines smoothworld
  - floor 3 sewers dungeon
  - floor 4 vaults dungeon
  - floor 5 troll mines smoothworld
  - floor 6 fire smoothworld
  - floor 7 ice smoothworld
  - floor 8 boss smoothworld
- Native reset generation covers `map`, `item_map`, `mob_map`, `light_map`,
  ladders, chest flags, `monsters_killed[0] = 10`, empty mob/projectile arrays,
  projectile directions, empty plants, the random `potion_mapping`, `state_rng`,
  and the scalar reset fields used by symbolic observations.
- `craftax_encode_reset_observation` encodes the native reset state into the
  flat symbolic observation, so `c_reset` no longer imports Python or calls JAX.
- `tests/craftax_worldgen_test.py` compares the native C reset state against JAX
  `generate_world` for 16 seeds, with exact map/item/ladder/potion/scalar checks
  and `atol=1e-6` for light and float state.
- The Python/JAX proxy is still used for `c_step`. Because step state is still
  JAX-owned, native `c_reset` marks the proxy dirty and the first delegated step
  lazily calls the proxy reset before applying the action. This keeps reset
  Python-free while preserving current step parity.

Remaining proxy paths:

- All step logic, rewards, achievements, auto-reset behavior after a delegated
  step, mob updates, inventory updates, and logging data still come from the
  Python/JAX proxy.
- `c_step` still allocates through Python/JAX and serializes on the GIL. The
  next porting phase should move gameplay state transitions native and remove
  the lazy step-side proxy reset.
- Rendering remains a no-op.
- `config/ocean/craftax.ini` still uses a small proxy-friendly vector size. The
  native port should raise this once step no longer calls Python.

## 2026-04-18 Native Floor-0 Reset Slice

This phase added the first native C replacement pieces while keeping the JAX
proxy as the oracle for all live game state and step logic.

- `threefry.h` ports JAX's `threefry2x32` PRNG for uint32 seeds, including
  `PRNGKey(seed)`, partitionable `split`/`split_n`, `fold_in`, and
  `uniform_u32`/float32 uniform helpers. `tests/craftax_threefry_test.py`
  compares bitwise against `jax.random.PRNGKey`, `split`, `fold_in`, and
  `bits`.
- `noise.h` ports `craftax/craftax/util/noise.py` for Perlin and fractal 2D
  noise. The test uses soft parity because C `sinf`/`cosf` and XLA
  transcendental lowering can differ by a few ulps; no JAX FFT path is used.
  `tests/craftax_noise_test.py` enforces `atol=rtol=2e-6`.
- `worldgen.h` ports default overworld `generate_smoothworld` for floor 0:
  `map`, `item_map`, `light_map`, `ladder_down`, and `ladder_up`.
  `tests/craftax_worldgen_floor0_test.py` compares these arrays against JAX for
  default reset seeds.
- `c_reset` still calls the JAX proxy to build the full observation and retain
  the JAX-owned state, then overwrites the visible floor-0 map/item/light
  observation channels from native C. Because native floor-0 generation matches
  the JAX reset data for default seeds, end-to-end step parity remains intact.

Remaining proxy paths:

- Floors 1..8 are still generated by JAX.
- The live `EnvState`, all step logic, rewards, achievements, auto-reset, mobs,
  inventory, and logging data still come from the Python/JAX proxy.
- The native floor-0 arrays are not yet installed into the JAX state object;
  this is safe only because the native generator currently matches the JAX
  oracle for the covered default reset path.

## Current Implementation

`ocean/craftax/` is wired as a full Craftax Ocean environment with the correct
symbolic observation size (`8268`) and action count (`43`). The C header declares
the full Craftax enum set and an `EnvState`-shaped C struct matching the field
order in `craftax_state.py`.

Reset is native for the full initial `generate_world` state and symbolic
observation. Step remains reference-backed: the C env acquires the Python GIL,
calls the installed JAX `Craftax-Symbolic-v1` implementation, and copies the
resulting float32 observation, reward, terminal flag, and terminal achievement
log into PufferLib-owned buffers. After a native reset, the first delegated step
performs a proxy reset internally so the JAX-owned step state starts from the
same seed and remains aligned with the native reset observation.

## Deliberate Divergences From The Requested Native Port

- The Craftax game logic is not yet native C. Step logic, achievements, rewards,
  auto-reset behavior after delegated steps, mobs, inventory updates, and other
  transition logic are delegated to the JAX oracle.
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

1. Replace one step subsystem at a time with native logic and keep the proxy as a
   local oracle until each subsystem matches.
2. Remove Python/JAX calls from `c_step`, restore large vector sizes, then measure
   CPU throughput before optimizing observation encoding, mob updates, and light
   propagation.

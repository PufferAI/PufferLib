# Craftax Full Ocean Port Notes

## Verification coverage

The standalone parity harness now supports deterministic action policies beyond
uniform random exploration:

- `uniform`: the original random action stream.
- `combat`: biases toward `DO`, arrows, fireballs, and iceballs when mobs and
  resources make those actions meaningful, otherwise moves toward live mobs.
- `descend`: uses the mirrored state to push toward down ladders, clear blocked
  levels through combat, and exercise placement and crafting actions.
- `suicide`: steers into adjacent lava, water, mob-occupied, or projectile-heavy
  danger and otherwise paths toward the nearest known hazard.
- `boss`: warms up with downward navigation and then repeatedly attempts
  descent while continuing to route toward ladders.
- `mixed`: round-robins the above every 500 steps.

`tests/craftax_parity.py` now reports the policy, seed, step, action, reward
delta, terminal delta, first symbolic-observation field, suspected subsystem,
and the last 10 actions on any divergence. With `--reset-on-done` enabled, the
harness tracks terminal counts and mean episode length by seed. JAX stepping is
run through the no-auto-reset path with the same per-step key split used by the
native env; when a terminal is observed, the mirrored state is advanced through
the native reset helper keyed by the same auto-reset key, and that reset state
and observation are checked field-by-field before continuing.

The stress battery in `tests/craftax_parity_stress.py` runs:

- 64 seeds times 10000 steps with `mixed`.
- 16 seeds times 30000 steps with `descend`.
- 32 seeds times 5000 steps with `suicide`.
- 16 seeds times 5000 steps with `combat`.

All stress cases use `atol=1e-5` for observations and rewards and exact terminal
matching. The phase-10a run completed with zero divergences in 1033.0 seconds:
2883 terminals in `mixed`, 2498 in `descend`, 622 in `suicide`, and 355 in
`combat`.

Residual caveats:

- The harness observes live C step state through the public vector API, so step
  diagnostics identify the first differing observation field and subsystem class
  rather than dumping the entire private C state after every step.
- CPU XLA can fuse reset worldgen noise normalization differently from
  materialized JAX by one ULP on exact threshold cells. Materialized JAX
  worldgen and native reset agree on the targeted sand-threshold keys covered by
  `tests/craftax_worldgen_test.py`, so terminal continuation uses the native
  reset helper after explicit reset-state verification.

## 2026-04-18 Native Step Integration and Proxy Removal

This phase wires the green native reset and all green native step subsystems
into the live Ocean `c_step` path. The Python/JAX proxy has been fully removed:
`c_init`, `c_reset`, `c_step`, and `c_close` are now 100% native.

- `c_step_native` now mirrors the installed `craftax_step` subsystem order:
  floor changes, crafting, action, placement, projectiles, spells, potions,
  books, enchantment, boss logic, attributes, movement, mobs, spawning, plants,
  intrinsics, clipping, inventory achievements, reward, timestep, light level,
  terminal, and symbolic observation encoding.
- The live env keeps the same outer RNG schedule as the old auto-reset proxy:
  reset uses the reset key's inner worldgen split, each step splits the external
  key once, then splits the per-step key into gameplay and auto-reset keys.
- Step observations reuse the native symbolic encoder, now with mob channels and
  boss-vulnerable special value populated for non-reset states.
- `tests/craftax_step_full_test.py` adds the full side-by-side parity check for
  16 seeds times 2000 random-action steps. `tests/craftax_parity.py` remains as
  the standalone harness.

Native-step roadmap checklist:

- [x] Native reset PRNG, noise, 9-floor world generation, and reset observation.
- [x] Standalone native simple step subsystems with JAX-parity tests.
- [x] Standalone native medium step subsystems with JAX-parity tests.
- [x] Standalone native crafting and placement subsystems with JAX-parity tests.
- [x] Standalone native `do_action` subsystem with JAX-parity tests.
- [x] Standalone native `spawn_mobs` subsystem with JAX-parity tests.
- [x] Standalone native `update_mobs` subsystem with JAX-parity tests.
- [x] Native reward, terminal, timestep, light-level, RNG, and achievement-delta
  bookkeeping around the subsystem calls.
- [x] Integrate all green subsystem ports into native `c_step` and remove all
  Python/JAX proxy code paths.

Remaining proxy paths:

- None. The Craftax Ocean env no longer loads CPython symbols, constructs a JAX
  env, or delegates reset/step/close through Python.

Next phase:

- Optimize the native path after correctness is locked down. Likely targets are
  SIMD-friendly loops, cache-tiled symbolic observation encoding, and mob update
  hot paths. Performance claims need measurement.

## 2026-04-18 Standalone Update Mobs Step Subsystem

This phase adds a native C port for the `update_mobs` subsystem, still
deliberately without integrating it into `c_step`. The live Ocean environment
continues to delegate step to the Python/JAX proxy.

- `step_update_mobs.h` contains the standalone in-place helper for:
  - `update_mobs`
- The helper mirrors the installed JAX update order for melee mobs, passive
  mobs, ranged mobs, mob projectiles, and player projectiles. It preserves the
  scan-level Threefry threading, including the melee loop's final right-key
  carry, and the top-level split before each mob class.
- Mob movement and collision use the installed collision tables for land,
  flying, aquatic, and amphibian mobs, including JAX-style clamped reads,
  scatter-drop writes, mob-map exclusion, water/lava/solid checks, despawn
  distance, boss-floor despawn suppression, and sequential mob-map updates.
- Combat covers melee player attacks, ranged projectile spawning, projectile
  movement, player damage with armour and enchantment defenses, sleeping/resting
  wakeups, player projectile damage scaling, first-target mob attacks, kill
  achievements, mob-map clearing, and `monsters_killed` updates.
- `tests/craftax_step_update_mobs_test.py` builds a temporary C wrapper around
  the inline helper and compares full copied states against the installed JAX
  function for 16 reset-plus-RNG-action-stepped states. Targeted coverage
  includes every mob class on every floor, melee attacks, ranged projectile
  firing, mob projectiles hitting the player, walls, and out-of-bounds, player
  projectile mob kills, despawn, cooldown decrement, and empty-mask live-effect
  checks.

Native-step roadmap checklist:

- [x] Native reset PRNG, noise, 9-floor world generation, and reset observation.
- [x] Standalone native simple step subsystems with JAX-parity tests.
- [x] Standalone native medium step subsystems with JAX-parity tests.
- [x] Standalone native crafting and placement subsystems with JAX-parity tests.
- [x] Standalone native `do_action` subsystem with JAX-parity tests.
- [x] Standalone native `spawn_mobs` subsystem with JAX-parity tests.
- [x] Standalone native `update_mobs` subsystem with JAX-parity tests.
- [ ] Native reward, terminal, timestep, light-level, RNG, and achievement-delta
  bookkeeping around the subsystem calls.
- [ ] Integrate all green subsystem ports into a native `c_step` behind one
  explicit switch, then remove the Python/JAX proxy from the normal step path.
- [ ] Restore production vector sizes in `config/ocean/craftax.ini` after native
  step is the default.
- [ ] Benchmark CPU throughput only after the proxy path is gone.

Remaining proxy paths:

- `c_step` still delegates to the Python/JAX proxy. None of the standalone
  subsystem helpers are wired into the live environment yet.
- All gameplay step subsystems now have standalone native ports with parity
  tests. Reward/terminal bookkeeping, light-level updates, timestep updates,
  RNG threading between subsystems, and achievement-delta logging are still not
  integrated natively.
- Rendering remains a no-op.
- `config/ocean/craftax.ini` still uses a small proxy-friendly vector size. The
  native port should raise this once step no longer calls Python.

## 2026-04-18 Standalone Spawn Mobs Step Subsystem

This phase adds a native C port for the `spawn_mobs` subsystem, still
deliberately without integrating it into `c_step`. The live Ocean environment
continues to delegate step to the Python/JAX proxy.

- `step_spawn_mobs.h` contains the standalone in-place helper for:
  - `spawn_mobs`
- The helper mirrors the installed JAX split order: passive chance, passive
  position, melee chance, melee position, ranged chance, ranged position. It
  also keeps the JAX behavior where the selected slot's `type_id` is written
  even when the spawn gate fails.
- Spawn maps match the installed function's terrain and distance rules,
  including passive distance rejection near the player, monster range gates,
  overworld night-zombie light scaling, deep-thing water spawning, grave-only
  boss-wave spawning, mob-map exclusion, caps, and sequential mob-map updates
  between passive, melee, and ranged attempts.
- `tests/craftax_step_spawn_mobs_test.py` builds a temporary C wrapper around
  the inline helper and compares full copied states against the installed JAX
  function for 16 reset-plus-NOOP-step seeds. Targeted coverage includes all
  nine floors, full mob caps, empty-slot spawns at single candidate positions,
  day versus night overworld melee chances, boss spawn-wave pacing, player-
  adjacent candidate rejection, and land, water, and grave terrain constraints.

Native-step roadmap checklist:

- [x] Native reset PRNG, noise, 9-floor world generation, and reset observation.
- [x] Standalone native simple step subsystems with JAX-parity tests.
- [x] Standalone native medium step subsystems with JAX-parity tests.
- [x] Standalone native crafting and placement subsystems with JAX-parity tests.
- [x] Standalone native `do_action` subsystem with JAX-parity tests.
- [x] Standalone native `spawn_mobs` subsystem with JAX-parity tests.
- [ ] Standalone native `update_mobs` subsystem with JAX-parity tests.
- [ ] Native reward, terminal, timestep, light-level, RNG, and achievement-delta
  bookkeeping around the subsystem calls.
- [ ] Integrate all green subsystem ports into a native `c_step` behind one
  explicit switch, then remove the Python/JAX proxy from the normal step path.
- [ ] Restore production vector sizes in `config/ocean/craftax.ini` after native
  step is the default.
- [ ] Benchmark CPU throughput only after the proxy path is gone.

Remaining proxy paths:

- `c_step` still delegates to the Python/JAX proxy. None of the standalone
  subsystem helpers are wired into the live environment yet.
- The only gameplay step subsystem still without a standalone native port is
  `update_mobs`. Reward/terminal bookkeeping, light-level updates, timestep
  updates, RNG threading between subsystems, and achievement-delta logging are
  also still not integrated natively.
- Rendering remains a no-op.
- `config/ocean/craftax.ini` still uses a small proxy-friendly vector size. The
  native port should raise this once step no longer calls Python.

## 2026-04-18 Standalone Do Action Step Subsystem

This phase adds a native C port for the `do_action` subsystem, still
deliberately without integrating it into `c_step`. The live Ocean environment
continues to delegate step to the Python/JAX proxy.

- `step_do_action.h` contains the standalone in-place helper for:
  - `do_action`
- The helper mirrors the installed JAX ordering: mob attack resolution runs
  before block interaction; block mining/eating/drinking/inventory/achievement
  effects are gated by in-bounds and no mob attack; chest-open flags and boss
  progress keep the JAX side effects that are not part of that gate.
- Chest looting calls the existing native `craftax_add_items_from_chest_native`
  helper after consuming the sapling RNG split, so first-open bow/book rewards
  see the old `chests_opened` value and the chest RNG thread matches JAX.
- Mob attacks cover passive, melee, and ranged mob arrays, including first-match
  target selection, defense mapping, sword enchantment damage, strength and
  intelligence scaling, passive food refill, kill achievements, mob-map updates,
  and monster kill counts.
- `tests/craftax_step_do_action_test.py` builds a temporary C wrapper around the
  inline helper and compares full copied states against the installed JAX
  function for 16 reset-plus-step-through seeds. Coverage includes a seeded
  no-op-then-DO sequence, mining success and missing-pickaxe cases, sapling RNG
  rolls, plant/passive food and water/fountain drink cases, all chest levels,
  all passive/melee/ranged kill achievement mappings, damage modifier cases,
  out-of-bounds targets, no-op target blocks, projectile-occupied targets, and
  mob-on-chest gating.

Native-step roadmap checklist:

- [x] Native reset PRNG, noise, 9-floor world generation, and reset observation.
- [x] Standalone native simple step subsystems with JAX-parity tests.
- [x] Standalone native medium step subsystems with JAX-parity tests.
- [x] Standalone native crafting and placement subsystems with JAX-parity tests.
- [x] Standalone native `do_action` subsystem with JAX-parity tests.
- [ ] Standalone native ports for the remaining mob step subsystems:
  `update_mobs` and `spawn_mobs`.
- [ ] Native reward, terminal, timestep, light-level, RNG, and achievement-delta
  bookkeeping around the subsystem calls.
- [ ] Integrate all green subsystem ports into a native `c_step` behind one
  explicit switch, then remove the Python/JAX proxy from the normal step path.
- [ ] Restore production vector sizes in `config/ocean/craftax.ini` after native
  step is the default.
- [ ] Benchmark CPU throughput only after the proxy path is gone.

Remaining proxy paths:

- `c_step` still delegates to the Python/JAX proxy. None of the standalone
  subsystem helpers are wired into the live environment yet.
- The only gameplay step subsystems still without standalone native ports are
  `update_mobs` and `spawn_mobs`. Reward/terminal bookkeeping, light-level
  updates, timestep updates, RNG threading between subsystems, and
  achievement-delta logging are also still not integrated natively.
- Rendering remains a no-op.
- `config/ocean/craftax.ini` still uses a small proxy-friendly vector size. The
  native port should raise this once step no longer calls Python.

## 2026-04-18 Standalone Crafting And Placement Step Subsystems

This phase adds native C ports for two more action subsystems, still
deliberately without integrating them into `c_step`. The live Ocean environment
continues to delegate step to the Python/JAX proxy.

- `step_crafting.h` contains standalone in-place helpers for:
  - `do_crafting`
  - `place_block`
  - `add_new_growing_plant`, used by plant placement and exposed to the test
    wrapper as a translation-unit-local helper
- `do_crafting` mirrors the JAX recipe order and sequential inventory updates
  for all twelve `MAKE_*` actions present in the current Action enum:
  pickaxes, swords, iron/diamond armour, arrows, and torches.
- `place_block` mirrors table, furnace, stone, plant, and torch placement,
  including original-block placement tests, item-map gating, mob/out-of-bounds
  rollback, first-empty growing-plant slot selection, and the padded 9x9 torch
  light update near map boundaries.
- `tests/craftax_step_crafting_test.py` builds a temporary C wrapper around the
  inline helpers and compares each subsystem against the installed JAX function
  on reset-plus-step-through states for 16 seeds. Coverage includes success,
  missing-resource/tool-cap, missing-station crafting cases; every JAX-legal
  placement target block for each placement action; illegal wall/item/mob/water
  cases where applicable; map-boundary rollback; and direct first-available-slot
  checks for growing plants.

Native-step roadmap checklist:

- [x] Native reset PRNG, noise, 9-floor world generation, and reset observation.
- [x] Standalone native simple step subsystems with JAX-parity tests.
- [x] Standalone native medium step subsystems with JAX-parity tests.
- [x] Standalone native crafting and placement subsystems with JAX-parity tests.
- [x] Standalone native `do_action` subsystem with JAX-parity tests.
- [ ] Standalone native ports for the remaining mob step subsystems:
  `update_mobs` and `spawn_mobs`.
- [ ] Native reward, terminal, timestep, light-level, RNG, and achievement-delta
  bookkeeping around the subsystem calls.
- [ ] Integrate all green subsystem ports into a native `c_step` behind one
  explicit switch, then remove the Python/JAX proxy from the normal step path.
- [ ] Restore production vector sizes in `config/ocean/craftax.ini` after native
  step is the default.
- [ ] Benchmark CPU throughput only after the proxy path is gone.

Remaining proxy paths:

- `c_step` still delegates to the Python/JAX proxy. None of the standalone
  subsystem helpers are wired into the live environment yet.
- The only gameplay step subsystems still without standalone native ports are
  `update_mobs` and `spawn_mobs`. Reward/terminal bookkeeping, light-level
  updates, timestep updates, RNG threading, and achievement-delta logging are
  also still not integrated natively.
- Rendering remains a no-op.
- `config/ocean/craftax.ini` still uses a small proxy-friendly vector size. The
  native port should raise this once step no longer calls Python.

## 2026-04-18 Standalone Medium Step Subsystems

This phase adds native C ports for five more step subsystems, again deliberately
without integrating them into `c_step`. The live Ocean environment still
delegates step to the Python/JAX proxy, so the full parity harness should remain
unchanged.

- `step_medium.h` contains standalone in-place helpers for:
  - `shoot_projectile`
  - `cast_spell`
  - `enchant`
  - `change_floor`
  - `add_items_from_chest`
- `add_items_from_chest` takes read-only `CraftaxState` context plus the
  `CraftaxInventory` being mutated because the JAX helper's special chest drops
  depend on `player_level` and `chests_opened`.
- `tests/craftax_step_medium_test.py` builds a temporary C wrapper around the
  inline helpers and compares each subsystem against the installed JAX function
  on copied reset-plus-step-through states for 16 seeds and targeted cases:
  projectile slot and resource gating, learned/unlearned spells, enchantment
  table/gem/mana/item gating, every floor transition direction, and chest potion
  and special-drop paths.
- The helpers do not allocate, do not call Python, and preserve the JAX details
  that matter for these routines, including clamped gather-style indexing,
  first-free projectile slot selection, cumulative-probability `choice` with
  `1 - uniform`, sequential Threefry split ordering, and the chest helper's
  intentionally unused wood roll.

Native-step roadmap checklist:

- [x] Native reset PRNG, noise, 9-floor world generation, and reset observation.
- [x] Standalone native simple step subsystems with JAX-parity tests.
- [x] Standalone native medium step subsystems with JAX-parity tests.
- [x] Standalone native crafting and placement subsystems with JAX-parity tests.
- [x] Standalone native `do_action` subsystem with JAX-parity tests.
- [ ] Standalone native ports for the remaining mob step subsystems:
  `update_mobs` and `spawn_mobs`.
- [ ] Native reward, terminal, timestep, light-level, RNG, and achievement-delta
  bookkeeping around the subsystem calls.
- [ ] Integrate all green subsystem ports into a native `c_step` behind one
  explicit switch, then remove the Python/JAX proxy from the normal step path.
- [ ] Restore production vector sizes in `config/ocean/craftax.ini` after native
  step is the default.
- [ ] Benchmark CPU throughput only after the proxy path is gone.

Remaining proxy paths:

- `c_step` still delegates to the Python/JAX proxy. None of the new medium
  helpers are wired into the live environment yet.
- The only gameplay step subsystems still without standalone native ports are
  `update_mobs` and `spawn_mobs`. Reward/terminal bookkeeping, light-level
  updates, timestep updates, RNG threading, and achievement-delta logging are
  also still not integrated natively.
- Rendering remains a no-op.
- `config/ocean/craftax.ini` still uses a small proxy-friendly vector size. The
  native port should raise this once step no longer calls Python.

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
- [x] Standalone native medium step subsystems with JAX-parity tests.
- [x] Standalone native crafting and placement subsystems with JAX-parity tests.
- [x] Standalone native `do_action` subsystem with JAX-parity tests.
- [ ] Standalone native ports for the remaining mob step subsystems:
  `update_mobs` and `spawn_mobs`.
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

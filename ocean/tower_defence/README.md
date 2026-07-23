# Tower Defence

Single-agent native Ocean environment with a typed placement lattice, three tower classes,
multi-path upgrades, procedural late-game waves, and a raylib demo.

## Contract

The environment uses a typed-placement ABI with native action masking:

| Surface | Layout |
| --- | --- |
| Build sites | `30 x 17` lattice (`510` sites) |
| Actions | `3572` discrete actions |
| Observations | `5686` floats |
| Native action mask | `3572` bytes |
| Agents | `1` |

Actions are site-major: `0` is noop, `1..1530` place one of three tower types,
`1531..3060` upgrade one of three paths, `3061..3570` sell, and `3571` starts the
next round. Legality is exposed once through PufferLib's native `MY_ACTION_MASK` interface for
masked sampling; it is not duplicated in the float observation. Checkpoints trained against the
older private `9258`-observation prototype are intentionally incompatible and must remain in their
matching legacy evaluator.

The simulation advances by a fixed `0.25` seconds per environment step. It starts with 200 lives
and 10,000 cash, uses fixed waves through round 20, and generates deterministic scaled waves after
that. The public challenge defaults to round 500 with a 25,000-step limit. Episode reward
parameters, target round, and limit are configurable in `config/tower_defence.ini`.

## Build and play

From the repository root:

```bash
./build.sh tower_defence --fast
./tower_defence
```

The bundled `resources/tower_defence/tower_defence_weights.bin` is the recurrent lean120M-r4 policy,
trained from scratch for 119,996,416 native transitions. The demo performs live recurrent inference
with the same checkpoint, action mask, and categorical policy semantics as evaluation. Actions are
sampled from the current game state rather than replayed from a recording. If the file is absent or
incompatible, the demo remains idle until manual input. Keep Shift held while taking control:

- `1`, `2`, `3`: select dart, sniper, or cannon
- left click: place at the hovered legal site
- `Q`, `W`, `E`: upgrade the hovered tower path
- right click or `X`: sell the hovered tower
- Space or Enter: start the next round

Rendering runs at 60 FPS while simulation remains at its fixed 4 Hz cadence. Moving sprites are
projected between simulation ticks. Projectile trails retain the preceding simulated point so a
shot visibly connects to its tower even when it travels or impacts within one 0.25-second step.
Time-stamped shot and impact events are presentation metadata only: they never enter observations,
rewards, targeting, or other rollout dynamics.

## Validation

`tests/test_tower_defence.c` covers the ABI, mask oracle, invalid numeric actions, deterministic
rollouts, split conservation across enemy-slot reuse, spawn RNG preservation, dynamically
growing projectiles, strictly improving paid fire-rate tiers, idle cooldown reset, next-tick split
movement, exact endpoint leaks, seed wraparound, terminal resets, recurrent demo state,
projectile travel segments, age-aware animation catch-up and event-ring wraparound, and randomized
stress. Defining `TD_TEST_RENDER` adds raylib render-purity and multi-environment teardown smokes.

After `./build.sh tower_defence --fast` has installed the repository-local raylib bundle, run the
headless suite with:

```bash
cc -std=c11 -O2 -Iraylib-5.5_linux_amd64/include -Isrc \
  tests/test_tower_defence.c raylib-5.5_linux_amd64/lib/libraylib.a \
  -lm -ldl -lpthread -lrt -lX11 -o /tmp/test_tower_defence
/tmp/test_tower_defence
```

On Linux with Xvfb, add `-DTD_TEST_RENDER` to the compile command and run the binary with
`xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 /tmp/test_tower_defence`.

Asset origins and the contributor's redistribution-rights confirmation are recorded in
`resources/tower_defence/ASSETS.md`.

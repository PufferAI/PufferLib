# Fight Caves

A native C Fight Caves environment for PufferLib 5.0: all 63 waves, a playable
Raylib viewer, and checkpoint replay on GPU or CPU. Training, replay and manual
play share one simulation.

## Build

Fight Caves needs only PufferLib 5.0's normal native build dependencies, which
[PufferTank 5.0](https://github.com/PufferAI/PufferTank/tree/5.0) provides; there
is no Fight Caves-specific Python package, trainer or Docker image.

```bash
./build.sh fight_caves         # native trainer with Fight Caves compiled in: ./puffer
./build.sh fight_caves --cpu   # manual play, CPU replay and checkpoint checks: ./fight_caves
```

Both builds first install the pinned [assets](#assets); add `--debug` to the CPU
build for sanitizers. Run all commands from the repository root, where the assets
and `config/fight_caves.ini` are found. If the viewer reports a `llvmpipe`
(software) renderer at startup, run `unset __GLX_VENDOR_LIBRARY_NAME` first;
PufferTank's image sets that GLX override.

## Train

```bash
./puffer train
```

Settings come from `config/fight_caves.ini` on top of `config/default.ini`, and
any of them can be overridden with `--section.key=value`. The `[env]` section
holds the starting supplies and the reward settings. Training is headless.

The environment reports episode averages for `zero_progress_ticks`,
`wave_reached`, `wrong_prayer_hits`, `reached_wave_63`, `jad_kill_rate`,
`prayer_uptime_range`, `prayer_uptime_melee`, `prayer_uptime_magic`,
`npc_healing_total`, `jad_healing_total` and `episode_length`. Puffer's `perf`
and `score` are both the Jad kill rate. Prayer uptimes are fractions of episode
ticks; healing totals count HP restored.

## Play

```bash
./fight_caves                  # add --start-wave 63 to begin at a later wave
```

Manual play needs no CUDA or checkpoint. Space starts and pauses, Right Arrow
steps one tick, R resets and Q quits. Left-click walks or attacks, right-click
opens menus, right-drag orbits the camera and the wheel zooms; 1/2/3 toggle the
protection prayers. The side panel has the combat, inventory, equipment and
prayer tabs; clicking the minimap walks there and its orb toggles running. The
console in the chat box lists targets, jumps to a wave, sets the tick rate,
toggles god mode and debug overlays, and shows player state, observations, the
action mask, rewards and the event log. The full key list is at the top of
`viewer.c`.

## Replay and check checkpoints

```bash
./puffer eval latest                  # or a checkpoint path; needs CUDA
./fight_caves eval CHECKPOINT.bin     # CPU replay in the same viewer
./fight_caves eval CHECKPOINT.bin --headless --base.eval_episodes=2
./fight_caves check CHECKPOINT.bin    # exact size and finite weights only
```

Replay disables the gameplay controls but keeps pause, step, camera and debug;
1/2/4/0 select 1x/2x/4x/10x and Up/Down change the speed. CPU replay needs an
explicit path, uses the native action masks and resets recurrent state between
episodes; its float32 inference can sample different trajectories than CUDA/BF16.
Both replay paths need the policy shape used in training: pass
`--policy.hidden_size=N --policy.num_layers=N` when it differs from the INI.

`check` confirms that a file's size matches the configured policy's float32
weights and that every weight is finite. The CUDA loader only rejects missing or
short files, so check unfamiliar files first. Neither proves where a checkpoint
came from or that it matches this game version; keep the source revision, config
and run log with your checkpoints.

## Assets

The first build downloads the pinned
[Fight Caves v3 bundles](https://github.com/jordanbailey00/fc-rl/releases/tag/fight-caves-assets-v3)
(about 6 MB) listed in `resources/fight_caves/asset_manifest.json`:

- `resources/fight_caves/runtime/`: collision, movement and line-of-sight maps.
- `resources/fight_caves/viewer/`: models, equipment parts, animations, terrain,
  textures, UI sprites and fonts, and the minimap.

Each archive's SHA-256 is checked before extraction and recorded in the installed
directory's `.archive-sha256` stamp. Later builds reuse a matching install offline
without re-reading its files, so reinstall after editing or deleting an installed
file. A failed download or checksum stops the build and keeps the existing
install. Setup needs only Python's standard library with tarfile extraction
filters (Python 3.12+, or 3.8.17/3.9.17/3.10.12/3.11.4 and later).

```bash
python3 ocean/fight_caves/tools.py setup                # what the build runs
python3 ocean/fight_caves/tools.py setup --verify-only  # check the stamps
python3 ocean/fight_caves/tools.py setup --force        # reinstall both bundles
```

Missing data stops with an error rather than falling back to an open map or
reduced graphics. `FC_COLLISION_PATH`, `FC_MOVEMENT_PATH`, `FC_LOS_PATH` and
`FC_ASSET_ROOT` (the viewer directory) override the installed locations. The
assets are exported from Old School RuneScape and are not covered by PufferLib's
software license; see `resources/fight_caves/ASSET_NOTICE.md`.

## Tests

```bash
python3 -m pytest -q tests/test_fight_caves_integration.py
FC_TEST_RENDER=1 python3 -m pytest -q tests/test_fight_caves_integration.py  # adds graphical checks
```

The suite builds a CPU executable with `./build.sh`, then checks the asset
installer, checkpoint handling and CPU replay, and runs the C simulation and
C/C++ linkage suites. Missing prerequisites fail the run; only the graphical
checks, which need `DISPLAY` or `xvfb-run`, are skipped without `FC_TEST_RENDER=1`.
`CC`/`CXX` select the compilers (default clang/clang++). No GPU training runs
in pytest.

### Clean-checkout acceptance

On Linux with CUDA and a display:

```bash
python3 -S tests/fight_caves_e2e.py clean-clone                 # committed HEAD
python3 -S tests/fight_caves_e2e.py clean-clone --working-tree  # plus uncommitted changes
```

This standard-library runner clones into a retained temporary directory, builds
both executables (downloading the assets and Raylib), runs the C suites including
NVCC linkage, opens the viewer, trains four batches and replays the checkpoints
on CPU and GPU. It also checks the errors for missing assets, a missing compiler
or display and bad checkpoints, and fails if any source file changes. Nothing is
silently skipped. Logs, screenshots, checkpoints and
`build/fight-caves-e2e/report.json` stay in the clone.

A dirty source is rejected unless `--working-tree` is given, which applies the
tracked diff and copies untracked, non-ignored files. `--source PATH_OR_URL --ref
REF` tests another commit. `--prepare-only` just prepares the clone; run
`python3 -S tests/fight_caves_e2e.py checkout` inside it, for example in
PufferTank.

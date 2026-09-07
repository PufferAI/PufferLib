# Fight Caves

Fight Caves is a single-agent native C environment for PufferLib 4.0. Its
training adapter, standalone simulator, and full Raylib viewer all compile the
same `simulation.h` implementation. Presentation remains separate from gameplay.

## Layout

The environment uses flat implementation headers, with no separate `src/`,
`include/`, or viewer source tree:

- `fight_caves.h`: Puffer lifecycle, observations, rewards and episode logging.
- `simulation.h`: game state, contracts, combat, routing, waves and loadouts.
- `binding.c`: Puffer 4.0 binding, configuration and compiled-contract export.
- `fight_caves.c`: standalone random-action simulator/benchmark.
- `viewer.c`: playable and policy-pipe entry point, input and scene lifecycle.
- `assets.h`: asset readers, models, animations, terrain and animated atlases.
- `ui.h`: OSRS interfaces, sprites, fonts, minimap and orbs.
- `render.h`: actor motion, animation selection, combat effects and debug overlays.
- `tools.py`: asset installation/verification, bundle creation, preflight,
  optional viewer build, playable launch and checkpoint replay.
- `CMakeLists.txt`: optional viewer build using Puffer's pinned Raylib 5.5.

Acceptance tests live in the repository's `tests/` directory. The full graphical
viewer is retained; the flat layout does not substitute a minimal renderer or
change the simulation, policy contract, or configuration.

## Requirements

Python 3.10 or newer and the normal PufferLib Python dependencies are required.
Activate your Python environment first: Puffer's stock `build.sh` invokes
`python` from `PATH`, which must be the same interpreter used for training.
Native builds require Clang, `ar`, and an OpenMP development runtime. The viewer
also requires CMake, OpenGL development libraries, and X11 development headers
on Linux.

On Ubuntu, the relevant system packages are:

```bash
sudo apt-get install clang libomp-dev libomp5 cmake \
  libgl1-mesa-dev libx11-dev libxrandr-dev libxi-dev \
  libxcursor-dev libxinerama-dev x11-utils xvfb
```

The environment-local preflight exits with a nonzero status and names any
missing dependency. It never substitutes a reduced simulator or viewer.
The shared `build.sh` is unchanged and does not invoke Fight Caves preflight;
run the explicit check before building a backend as shown below.

## Install assets

The runtime maps and graphical viewer data are published as versioned GitHub
release bundles. Install and verify both bundles from the repository root:

```bash
python3 ocean/fight_caves/tools.py setup --all
```

The installer verifies the archive and every installed file against
`resources/fight_caves/asset_manifest.json`. A download, checksum, extraction,
or installation error exits nonzero without replacing an existing installation.

## Build and test

Use Puffer's standard build commands for the training backend:

```bash
python ocean/fight_caves/tools.py preflight --mode cpu
./build.sh fight_caves --cpu
```

For CUDA, use `preflight --mode cuda` followed by `./build.sh fight_caves`.
For the standalone simulator, use `preflight --mode native` followed by
`./build.sh fight_caves --fast`.

Build the CPU Puffer backend and run the environment acceptance tests:

```bash
bash tests/fight_caves.sh test --puffer
```

Include the viewer build:

```bash
bash tests/fight_caves.sh test --all
```

Run the playable viewer through its asset-verifying launcher:

```bash
python3 ocean/fight_caves/tools.py build-viewer
python3 ocean/fight_caves/tools.py play
```

`build-viewer` checks dependencies and assets, reuses Puffer's Raylib 5.5
installation if present, or downloads the same official release into `build/`.
Use `--raylib-root /path/to/raylib` to supply an existing installation, including
on platforms without a matching prebuilt release. An incomplete installation
fails explicitly. Viewer building does not require the Puffer backend or CUDA.

The launcher verifies all required assets and checks the graphical display.
The viewer retains tile clicking and route previews, OSRS click indicators,
camera controls, equipment/prayer/inventory tabs, run-energy and minimap orbs,
wave/TPS/target controls, god mode, debug information, prayer-window indicators,
projectiles, impacts, health bars and hitsplats.

Useful controls include `Space` to pause/resume, `Right Arrow` to step one tick,
`O` for debug overlays, right-drag to orbit, the mouse wheel to zoom, and
`Q`/`Escape` to quit.

## Checkpoint replay

Build the Fight Caves backend (`./build.sh fight_caves --cpu`, or the normal CUDA
build), then replay a checkpoint in the same viewer:

```bash
python3 ocean/fight_caves/tools.py eval --ckpt /absolute/path/to/checkpoint.bin --episodes 1
```

Replay retains the raw CUDA and PyTorch CPU checkpoint readers, compiled-contract
and size checks, masking, pause/speed controls and episode summaries. `--ckpt latest`
selects the newest compatible checkpoint; `--random` uses random legal actions
without loading a checkpoint. The existing dedicated policy-pipe evaluator is
preserved, rather than changing inference or switching to a different renderer.

The viewer defaults to `resources/fight_caves/viewer`; arena maps default to
`resources/fight_caves/runtime`. Explicit `FC_ASSET_ROOT`, `FC_REPO_ROOT`,
`FC_COLLISION_PATH`, `FC_MOVEMENT_PATH` and `FC_LOS_PATH` overrides remain available.
Use `python3 ocean/fight_caves/tools.py COMMAND --help` for setup, bundle,
preflight, viewer-build and replay options.

## Clean-clone acceptance

Maintainers can reproduce installation, native and Puffer builds, a short
training run, viewer startup, checkpoint replay, and deliberate failure cases
from a new checkout with:

```bash
bash tests/fight_caves.sh clean-clone
```

The command clones the current origin branch into a temporary directory, creates
a new virtual environment, downloads only the published pinned asset bundles,
and runs the complete acceptance sequence. Set `FC_CLEAN_CLONE_KEEP=1` to retain
the isolated checkout after a failure for inspection.
This validates the committed branch, not uncommitted local changes. The
`checkout` subcommand runs acceptance directly in a fresh checkout without
installed assets, and `test --core` runs just the asset/contract and C checks.

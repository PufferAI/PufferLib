# Fight Caves

A native C Fight Caves environment for PufferLib 4.0, with all 63 waves,
training, a playable Raylib viewer, and checkpoint replay. The trainer and
viewer share the same simulation.

## Setup in PufferTank 4.0

Use the official [PufferTank 4.0 environment](https://github.com/PufferAI/PufferTank/tree/4.0),
following [Puffer's installation instructions](https://puffer.ai/docs.html#installation).
Fight Caves does not have its own Docker image or Python/CUDA dependency stack.
For play and replay, start PufferTank with the display forwarding described by
Puffer; a headless container can train, but cannot show an interactive window.

Inside PufferTank's interactive shell, use its already-activated Python
environment. While this PR is under review, clone its source branch:

```bash
git clone --branch fight-caves-puffertank-4.0 https://github.com/jordanbailey00/PufferLib.git PufferLib-fight-caves
cd PufferLib-fight-caves
uv pip install --no-deps -e .
```

The editable installation points the existing `puffer` command at this checkout.
`--no-deps` preserves the dependencies supplied by PufferTank. Do not create
another virtual environment or replace its PyTorch/CUDA packages for Fight Caves.
Run the following commands from this checkout's root.

## Train

```bash
./build.sh fight_caves
puffer train fight_caves
```

The build automatically downloads and verifies all required assets. There is
no separate setup or viewer-build command. `config/fight_caves.ini` supplies the
750M-step configuration. For W&B logging, run `wandb login` once and train with:

```bash
puffer train fight_caves --wandb --wandb-project fight-caves
```

Ordinary training is headless; it does not create a graphical window.

## Play manually

```bash
./build.sh fight_caves --fast
./fight_caves
```

You can play without training a policy first. Press Space to start or pause,
Right Arrow to advance one tick, O to toggle debug overlays, and Q to quit.
Right-drag rotates the camera; the mouse wheel zooms. Use `--local` instead of
`--fast` for Puffer's debug/sanitizer build.

The full viewer includes tile clicks and route previews, equipment switching,
inventory and prayer controls, right-click menus, minimap/run-energy controls,
animations, projectiles, impacts, health bars, and hitsplats. The console has
wave/target/TPS selection, god mode, observations, rewards, and an event log.

## Replay a checkpoint

With the same backend and policy architecture used for training:

```bash
puffer eval fight_caves --load-model-path latest
```

Replace `latest` with a checkpoint path to select a specific model. `latest`
selects the newest file, not the highest-scoring policy. Evaluation opens the
same full viewer, using Puffer's policy inference and environment stepping.
Camera, debug, pause, and speed controls remain available; gameplay-changing
controls are disabled. Keys 1/2/4/0 select 1x/2x/4x/10x. Q closes evaluation.

Puffer compiles one selected environment/backend into `pufferlib/_C`. Rebuild
when switching environments or between native CUDA and CPU backends. Building
the standalone viewer with `--fast` does not replace the training backend.

## Assets

The first build installs the pinned [Fight Caves v3 bundles](https://github.com/jordanbailey00/fc-rl/releases/tag/fight-caves-assets-v3):

- `resources/fight_caves/runtime/`: collision, movement, and line-of-sight maps.
- `resources/fight_caves/viewer/`: models, equipment parts, animations, terrain,
  textures, UI sprites/fonts, and the minimap raster.

Archive and individual-file sizes/SHA-256 hashes are checked before installation.
Valid assets are reused, including offline. Rerunning the build repairs missing
or corrupt bundles; download or verification failure stops the build. Runtime
loads local files only, without another repository or raw OSRS cache. Missing
required data produces an error, not an open-map or reduced-graphics fallback.

Optional one-time installation or verification:

```bash
python ocean/fight_caves/tools.py setup --all
python ocean/fight_caves/tools.py setup --all --verify-only
```

OSRS assets are distributed separately and are not covered by PufferLib's
software license; see `resources/fight_caves/ASSET_NOTICE.md`.

## Optional tools

Puffer's CPU/PyTorch path uses `./build.sh fight_caves --cpu` followed by
`puffer train fight_caves --slowly` or `puffer eval fight_caves --slowly`.
Native CUDA and PyTorch checkpoints use different formats. To replay native
CUDA weights on the CPU, or stop after one episode, the compatibility tool is
still available:

```bash
./build.sh fight_caves --cpu
./build.sh fight_caves --fast
python ocean/fight_caves/tools.py eval --ckpt /path/to/checkpoint.bin --episodes 1
```

It uses the same viewer executable. Rebuild the CUDA backend before returning
to native training. `./fight_caves --benchmark` runs the headless benchmark.

Maintainer tests are explicitly invoked, not part of normal setup:

```bash
bash tests/fight_caves.sh test --core
bash tests/fight_caves.sh test --all
```

Tests additionally need pytest; graphical regression builds use CMake and an
existing DISPLAY or Xvfb. These are test tools, not extra gameplay dependencies.
`test --all` builds the CPU backend; rebuild CUDA afterward for native training.
`clean-clone` tests a committed branch, while `checkout` tests an isolated source
copy. Neither command is required to use the environment.

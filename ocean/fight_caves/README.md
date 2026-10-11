# Fight Caves

A native C simulation of all 63 Fight Caves waves, with training, manual play,
and checkpoint replay sharing the same game rules. The policy controls movement,
attacks, and protection prayers; consumables and equipment switching are
available in manual play. The default configuration uses fixed gear and no supplies.

Use an installed PufferLib checkout following [Puffer's installation instructions](https://puffer.ai/docs.html#installation).
Run commands from the repository root. Interactive play and replay require a
display; training is headless.

## Train

```bash
./build.sh fight_caves
puffer train fight_caves
```

[fight_caves.ini](../../config/fight_caves.ini) sets the rewards, model, and
750M-step training budget. Add `--wandb --wandb-project fight-caves` for W&B logging.
Episode metrics include `env/jad_kill_rate` and `env/wave_reached`. Prayer uptime
is a fraction of episode ticks; healing totals are measured in tenths of HP.

## Play

```bash
./build.sh fight_caves --fast
./fight_caves
```

The viewer starts paused. Space starts/pauses, Right Arrow advances one tick,
O toggles diagnostics, and Q quits. Click tiles or NPCs to move or attack;
use the side panels for prayers, inventory, and equipment. Right-drag rotates
the camera and the mouse wheel zooms. Use `--local` instead of `--fast` for a
debug/sanitizer build.

## Replay

With the same backend and model architecture used for training:

```bash
puffer eval fight_caves --load-model-path latest
```

Replace `latest` with a checkpoint path to choose a specific policy. Replay
allows camera, pause, speed, and diagnostic controls; gameplay controls are
disabled. Keys 1/2/4/0 select 1x/2x/4x/10x speed.

For the CPU/PyTorch backend, build with `--cpu` and add `--slowly` to train/eval.
Native CUDA and PyTorch checkpoints use different formats. Rebuild when changing
the training backend or environment; `--fast` only builds the standalone viewer.

## Assets

The build automatically installs the bundles pinned in
[asset_manifest.json](../../resources/fight_caves/asset_manifest.json).
The runtime bundle contains collision, movement, and line-of-sight maps; the
viewer bundle contains models, animations, terrain, textures, and UI assets.
Archive and file hashes are verified, valid installations are reused offline,
and rerunning the build repairs missing or corrupt assets. Exported game assets
are separate from the software license; see [ASSET_NOTICE.md](../../resources/fight_caves/ASSET_NOTICE.md).

## Tests

With pytest installed:

```bash
./build.sh fight_caves --cpu
python -m pytest -q tests/test_fight_caves_integration.py
```

This runs asset checks, compiles/runs the C simulation tests, and checks the
Puffer backend. No display is needed. Rebuild the CUDA backend before returning
to native training.

## Slow viewer in PufferTank

If the viewer reports `llvmpipe`, it is rendering on the CPU. PufferTank's forced
GLX vendor setting can cause this; try `unset __GLX_VENDOR_LIBRARY_NAME` in the
current shell before play/replay. This restores automatic driver selection;
working GPU drivers and container display access are still required.

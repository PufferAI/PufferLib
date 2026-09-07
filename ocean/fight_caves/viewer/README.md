# Fight Caves viewer

The optional Raylib viewer uses the same C simulation sources as the PufferLib
environment. Presentation state, interpolation, UI, and policy replay remain
outside the gameplay core.

Install the runtime and graphical asset bundles from the repository root:

```bash
bash ocean/fight_caves/scripts/setup-data.sh --all
```

Build and launch the playable viewer:

```bash
./build.sh fight_caves --viewer
bash ocean/fight_caves/scripts/run-viewer.sh
```

The launcher verifies every installed bundle file before starting the binary.
Missing or corrupt required data is fatal and reports the asset setup command;
the viewer does not silently use a reduced asset set.

The build reuses PufferLib's pinned Raylib 5.5 distribution. On Linux it also
requires the standard X11 and OpenGL development libraries used by PufferLib's
other native renderers.

For policy replay, first build the Fight Caves Puffer backend and then provide
a Puffer checkpoint. The evaluator strictly supports both the flat raw format
written by the CUDA trainer and the PyTorch state dictionary written by the CPU
trainer:

```bash
./build.sh fight_caves --cpu
python3 ocean/fight_caves/viewer/eval_viewer.py \
    --ckpt /absolute/path/to/checkpoint.bin \
    --episodes 1
```

`--ckpt latest` selects the newest size- and contract-compatible checkpoint
under `checkpoints/`. `--random` exercises the same policy-pipe protocol with
random legal actions and does not load a checkpoint.

The viewer defaults to `resources/fight_caves/viewer` and the simulator maps
default to `resources/fight_caves/runtime`. `FC_ASSET_ROOT`, `FC_REPO_ROOT`,
`FC_COLLISION_PATH`, `FC_MOVEMENT_PATH`, and `FC_LOS_PATH` remain available as
explicit development/test overrides.

Useful viewer controls:

- `Space`: pause or resume.
- `Right Arrow`: advance one tick while paused.
- `O`: cycle the debug overlay.
- Right mouse drag: orbit the camera.
- Mouse wheel: zoom.
- `Q` or `Escape`: quit.

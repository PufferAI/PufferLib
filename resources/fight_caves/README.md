# Fight Caves assets

Fight Caves uses two independently versioned asset bundles:

- `core` contains the collision, movement, and line-of-sight maps required by
  training, evaluation, and the viewer.
- `viewer` contains the models, animations, terrain, textures, sprites, fonts,
  and minimap used only by the graphical viewer.
  Version 3 includes composable player equipment/body parts, their visibility
  map, the corrected Venator ring icon and the bold RuneC context-menu font.

The archives are pinned in `asset_manifest.json` by URL, byte size, and SHA-256.
Every installed file is also checked by byte size and SHA-256 before it is
accepted. Installed asset directories are intentionally excluded from Git.

From the PufferLib repository root, install only the simulator data:

```bash
python3 ocean/fight_caves/tools.py setup --core
```

Install only the graphical assets, or install both bundles:

```bash
python3 ocean/fight_caves/tools.py setup --viewer
python3 ocean/fight_caves/tools.py setup --all
```

Running the script without a bundle option is equivalent to `--all`. Verify an
existing installation without downloading or changing it with:

```bash
python3 ocean/fight_caves/tools.py setup --all --verify-only
```

The `tools.py build-viewer`, `play`, and `eval` commands verify required assets
automatically and fail with a nonzero exit status when data is absent or corrupt.
Before using Puffer's unchanged `build.sh`, run
`python3 ocean/fight_caves/tools.py preflight --mode cpu` (or `cuda`/`native` for
those builds). The simulator also refuses to start when required arena maps
cannot be loaded; it does not fall back to open maps. Viewer launch never
substitutes an incomplete graphical asset set.

The simulator retains the `FC_COLLISION_PATH`, `FC_MOVEMENT_PATH`, and
`FC_LOS_PATH` environment-variable overrides for controlled development and
testing. The integrated viewer uses `resources/fight_caves/viewer` as its
default asset root.

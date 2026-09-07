# Fight Caves assets

Fight Caves uses two independently versioned asset bundles:

- `core` contains the collision, movement, and line-of-sight maps required by
  training, evaluation, and the viewer.
- `viewer` contains the models, animations, terrain, textures, sprites, fonts,
  and minimap used only by the graphical viewer.

The archives are pinned in `asset_manifest.json` by URL, byte size, and SHA-256.
Every installed file is also checked by byte size and SHA-256 before it is
accepted. Installed asset directories are intentionally excluded from Git.

From the PufferLib repository root, install only the simulator data:

```bash
bash ocean/fight_caves/scripts/setup-data.sh --core
```

Install only the graphical assets, or install both bundles:

```bash
bash ocean/fight_caves/scripts/setup-data.sh --viewer
bash ocean/fight_caves/scripts/setup-data.sh --all
```

Running the script without a bundle option is equivalent to `--all`. Verify an
existing installation without downloading or changing it with:

```bash
bash ocean/fight_caves/scripts/setup-data.sh --all --verify-only
```

Fight Caves build, viewer-launch, and policy-replay entry points invoke this
verification automatically and fail with a nonzero exit status when required
data is absent or corrupt. They do not fall back to open arena maps or an
incomplete graphical asset set.

The simulator retains the `FC_COLLISION_PATH`, `FC_MOVEMENT_PATH`, and
`FC_LOS_PATH` environment-variable overrides for controlled development and
testing. The integrated viewer uses `resources/fight_caves/viewer` as its
default asset root.

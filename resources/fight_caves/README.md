# Fight Caves assets

Normal users do not need a separate setup command. Every standard Fight Caves
build installs/verifies the pinned bundles automatically:

```bash
./build.sh fight_caves          # Native CUDA backend, also ready for puffer eval
./build.sh fight_caves --cpu    # CPU/PyTorch backend
./build.sh fight_caves --fast   # Playable ./fight_caves executable
```

The first build downloads the archives named in `asset_manifest.json` from the
versioned GitHub release. Existing valid installations are reused without
network access. Archive and per-file sizes/SHA-256 are verified before replacing
an installed bundle; failure aborts the build with a diagnostic.

- `runtime/` contains the three collision, movement, and LOS maps (12 KiB total).
- `viewer/` contains models, equipment parts, animations, terrain, textures,
  sprites, fonts, and the minimap raster.

These installed directories are ignored by Git. Runtime loads local assets
directly from these paths; no reference repository, export tools, setup wrapper,
or runtime download is required. Headless training never initializes graphics.

For maintenance only:

```bash
python3 ocean/fight_caves/tools.py setup --all --verify-only
python3 ocean/fight_caves/tools.py setup --all --force
```

Explicit `FC_COLLISION_PATH`, `FC_MOVEMENT_PATH`, `FC_LOS_PATH`,
`FC_ASSET_ROOT`, and `FC_REPO_ROOT` overrides remain available for development
and tests. Run standard commands from the Puffer repository root.

# Fight Caves

Fight Caves is a single-agent native C environment for PufferLib 4.0. Its
training adapter, standalone simulator, and optional Raylib viewer all link the
same gameplay sources listed in `sources.txt`.

## Requirements

Python 3.10 or newer and the normal PufferLib Python dependencies are required.
Native builds require Clang, `ar`, and an OpenMP development runtime. The viewer
also requires CMake, OpenGL development libraries, and X11 development headers
on Linux.

On Ubuntu, the relevant system packages are:

```bash
sudo apt-get install clang libomp-dev libomp5 cmake \
  libgl1-mesa-dev libx11-dev libxrandr-dev libxi-dev \
  libxcursor-dev libxinerama-dev x11-utils xvfb
```

The build preflight exits with a nonzero status and names any missing
dependency. It never substitutes a reduced simulator or viewer.

## Install assets

The runtime maps and graphical viewer data are published as versioned GitHub
release bundles. Install and verify both bundles from the repository root:

```bash
bash ocean/fight_caves/scripts/setup-data.sh --all
```

The installer verifies the archive and every installed file against
`resources/fight_caves/asset_manifest.json`. A download, checksum, extraction,
or installation error exits nonzero without replacing an existing installation.

## Build and test

Build the CPU Puffer backend and run the environment acceptance tests:

```bash
bash ocean/fight_caves/scripts/test-env.sh --puffer
```

Include the viewer build:

```bash
bash ocean/fight_caves/scripts/test-env.sh --all
```

Run the playable viewer through its asset-verifying launcher:

```bash
bash ocean/fight_caves/scripts/run-viewer.sh
```

See `viewer/README.md` for checkpoint replay.

## Clean-clone acceptance

Maintainers can reproduce installation, native and Puffer builds, a short
training run, viewer startup, checkpoint replay, and deliberate failure cases
from a new checkout with:

```bash
bash ocean/fight_caves/scripts/validate-clean-clone.sh
```

The script clones the current origin branch into a temporary directory, creates
a new virtual environment, downloads only the published pinned asset bundles,
and runs the complete acceptance sequence. Set `FC_CLEAN_CLONE_KEEP=1` to retain
the isolated checkout after a failure for inspection.

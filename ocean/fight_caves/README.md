# Fight Caves

A single-agent native C Fight Caves environment for PufferLib 4.0. Training,
human play, and checkpoint evaluation share the same simulation and full Raylib
viewer. Gameplay, observations, rewards, action heads, and the default 750M-step
configuration are unchanged by the standard-workflow integration.

## Setup and training

Use a normal Puffer 4.0 development environment: Python 3.10+, Clang/OpenMP,
Raylib's system/OpenGL dependencies, and the CUDA/cuDNN/NCCL development stack
for native GPU training. These are shared Puffer prerequisites, not a separate
Fight Caves installation. An optional reproducible Docker setup is below.

From the repository root, in your activated Python environment:

```bash
python -m pip install -e .
./build.sh fight_caves
puffer train fight_caves --wandb
```

The build **automatically installs and verifies both asset bundles**. There is
no separate Fight Caves setup, preflight, or viewer-build command to remember.
Verified assets are reused on subsequent builds, including offline builds.
The first installation requires access to the pinned GitHub release.

CPU/PyTorch training follows Puffer's ordinary alternative:

```bash
./build.sh fight_caves --cpu
puffer train fight_caves --slowly
```

Puffer compiles one selected environment/backend into `pufferlib/_C`. Rebuild
when switching environments or between native CUDA and CPU backends.

## Play manually

```bash
./build.sh fight_caves --fast
./fight_caves
```

Use `--local` instead of `--fast` for Puffer's debug/sanitizer build. Human play
does not require CUDA or a compiled Python backend. A graphical desktop is
required. The game starts paused; press Space to begin.

The full viewer is retained: tile clicking and path previews, camera controls,
equipment switching and right-click menus, inventory/prayer interfaces, minimap
and run-energy orbs, wave/TPS/target controls, god mode, diagnostics, projectiles,
animations, health bars, hitsplats, and Prayer-window indicators.

Space pauses; Right Arrow single-steps; O toggles the debug overlay; right-drag
orbits; the mouse wheel zooms; Q quits. The console contains wave, target,
speed, and god-mode controls. `./fight_caves --benchmark` retains the optional
headless random-action benchmark.

## Watch a checkpoint

After building the same backend used to train the checkpoint:

```bash
puffer eval fight_caves --load-model-path latest
```

Or pass a specific checkpoint path. PyTorch checkpoints use Puffer's `--slowly`
backend. Standard evaluation uses Puffer's own policy inference, masking, and
environment stepping; the viewer only displays snapshots of the evaluated
environment. Graphics are initialized lazily by `c_render()`, never by ordinary
headless training. Terminal snapshots are retained before same-step autoreset.

The same viewer supports camera/debug/pause/speed controls during evaluation.
Gameplay-changing controls are disabled in replay. Keyboard 1/2/4/0 selects
1x/2x/4x/10x playback. Closing the window or pressing Q ends evaluation.
Puffer's standard `latest` means newest by file time, not highest-scoring.

### Optional CPU-only compatibility replay

Puffer's native CUDA and PyTorch backends use different checkpoint formats.
The optional compatibility reader is retained for replaying native CUDA weights
on the CPU, deterministic sampling, or a fixed episode limit:

```bash
./build.sh fight_caves --cpu
./build.sh fight_caves --fast
python ocean/fight_caves/tools.py eval --ckpt /path/to/checkpoint.bin --episodes 1
```

It uses the same `./fight_caves` executable and the existing contract checks.
It is not required for ordinary `puffer eval`. Rebuild the CUDA backend before
resuming native training after a CPU build.

## Assets

`build.sh` invokes the existing pinned installer automatically for Fight Caves.
Archive and individual-file SHA-256 checks precede transactional installation:

- `resources/fight_caves/runtime/`: collision, movement, and LOS maps.
- `resources/fight_caves/viewer/`: models, equipment parts, animations, terrain,
  textures, sprites, fonts, and the minimap raster.

The simulator and viewer load these local paths directly; no cache export,
reference repository, external codebase, or runtime network call is needed.
Missing or invalid required data fails rather than substituting open maps or
reduced graphics. Rerunning the build repairs missing/corrupt installed bundles;
download or validation failure stops the build.

`tools.py setup`, `bundle`, and `preflight` remain explicit maintenance tools.
The old `tools.py build-viewer` and `play` commands are compatibility aliases
for the standard standalone build and executable.

## Maintainer checks and layout

```bash
bash tests/fight_caves.sh test --core
bash tests/fight_caves.sh test --all
```

The optional graphical test target uses CMake/Xvfb. Neither is required to
build or launch the ordinary viewer. Tests cover assets/failure handling,
contracts, equipment, graphics, and rendering-versus-headless trajectory parity.

- `simulation.h`: unchanged combat, movement, waves, items, contracts and state.
- `fight_caves.h`, `binding.c`: Puffer integration and lazy renderer connection.
- `fight_caves.c`: conventional playable entry point and optional benchmark.
- `viewer.c`: shared viewer lifecycle, input, frame rendering and compatibility pipe.
- `assets.h`, `ui.h`, `render.h`: existing assets, interface and presentation.
- `tools.py`: asset maintenance and optional cross-backend checkpoint replay.
- `CMakeLists.txt`: optional graphical regression builds.
- `Dockerfile` and constraints: optional reproducible development environment.

`bash tests/fight_caves.sh clean-clone` tests a committed branch, not uncommitted
working-copy changes. Use `checkout` in an isolated source copy to validate
uncommitted work.

## Docker (Ubuntu / NVIDIA)

Ubuntu 24.04 x86-64, CUDA 13.0 and Raylib 5.5. `docker-constraints.txt` pins
PyTorch 2.9.1+cu130 and W&B 0.28.1 for this branch's `wandb.util.generate_id()` call.

The host needs an NVIDIA GPU with a driver compatible with CUDA 13.0,
and [NVIDIA Container Toolkit configured for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Assets: [fight-caves-assets-v3](https://github.com/jordanbailey00/fc-rl/releases/tag/fight-caves-assets-v3),
pinned in `resources/fight_caves/asset_manifest.json`.

Build from the repository root:

```bash
docker build --platform linux/amd64 \
  -f ocean/fight_caves/Dockerfile -t fight-caves:local .
```

The image builds the viewer and runs the core tests without a GPU. The CUDA
backend is built once inside each new container, where Puffer's default
`NVCC_ARCH=native` can detect the GPU.

Training or headless tests:

```bash
docker run -it --name fight-caves-test --gpus all --shm-size=1g \
  --mount type=volume,source=fight-caves-checkpoints,target=/workspace/PufferLib/checkpoints \
  --mount type=volume,source=fight-caves-logs,target=/workspace/PufferLib/logs \
  fight-caves:local
```

Desktop play/replay requires X11 or XWayland and `xauth`. Run from the host's
graphical session with `DISPLAY` set:

```bash
FC_XAUTH=$(mktemp /tmp/fight-caves-xauth.XXXXXX)
xauth -f "${XAUTHORITY:-$HOME/.Xauthority}" nlist "$DISPLAY" \
  | sed 's/^..../ffff/' | xauth -f "$FC_XAUTH" nmerge -

docker run -it --name fight-caves-test --gpus all --shm-size=1g \
  -e DISPLAY -e XAUTHORITY=/tmp/fight-caves.Xauthority \
  --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly \
  --mount "type=bind,source=$FC_XAUTH,target=/tmp/fight-caves.Xauthority,readonly" \
  --mount type=volume,source=fight-caves-checkpoints,target=/workspace/PufferLib/checkpoints \
  --mount type=volume,source=fight-caves-logs,target=/workspace/PufferLib/logs \
  fight-caves:local
```

Keep the temporary authorization file while that container is in use. A new
desktop login may require a fresh authorization file and container. Both launch
examples use the same container name; choose one.

Inside the container, in `/workspace/PufferLib`:

```bash
./build.sh fight_caves
bash tests/fight_caves.sh test --core
```

750M-step training with W&B:

```bash
wandb login
puffer train fight_caves --wandb --wandb-project fight-caves
```

Viewer and checkpoint replay:

```bash
./fight_caves
puffer eval fight_caves --load-model-path latest
```

Headless viewer check:

```bash
xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 \
  ./fight_caves --screenshot playable.png
```

# AGENTS.md

Guidance for autonomous agents working on the Pathfinder PufferLib env.

## Current Workspace

This machine is `g240`.

Writable working repo:

- `/home/claude/pathfinder`
- Branch: `pathfinder`
- Base: official PufferLib `5.0`
- Remote: `origin https://github.com/PufferAI/PufferLib.git`

This workspace is for a new PufferLib 5 Ocean environment named
`pathfinder`, based on Milton Bradley's 1977 Pathfinder board game.

Reference repos:

- `/home/claude/dogfight5`
  - PufferLib 5 Dogfight port workspace.
  - Use as the local reference for uv setup, CUDA 12.8 activation, native build
    behavior, and PufferLib 5 state-memory conventions.
  - Do not modify it unless the human explicitly redirects work there.
- `/home/claude/PufferLib`
  - Dirty Dogfight 4 control workspace.
  - Do not use as the implementation target for Pathfinder.

## Goal

Build a single-agent Pathfinder maze-solving environment first.

The env should generate a hidden 6x6 barricade maze, hide a pawn, start the
agent at `A1`, and train the agent to discover walls and find a path to the
hidden pawn using only four movement actions.

The current design source is:

- `/home/claude/pathfinder/PATHFINDER_SPEC.md`

Update that spec deliberately when requirements change. Do not let
implementation drift away from it silently.

## Rule Source

The board-game rules were read from:

- User-provided rules scan:
  `http://www.transformertoys.co.uk/images/instruction-scans/hasbro/Pathfinder.pdf`
- User-provided video transcript.

Important game facts:

- The board is 6x6 with rows `A-F` and columns `1-6`.
- Barriers live in slots around and between squares.
- A hidden short pawn sits on one square.
- There must be at least one route from a column-1 entrance to the hidden pawn.
- Movement is orthogonal only.
- Asking for a blocked edge reveals a wall and leaves the pawn in place.
- Asking for an open edge moves the pawn.
- Moving into the hidden pawn square wins.

For v1, simplify the board-game entry/re-entry rule:

- The agent always starts at `A1`.
- The generator must guarantee a route from `A1` to the hidden pawn.
- Action space is four moves: north, east, south, west.

## Safety Rules

- Do implementation work only in `/home/claude/pathfinder`.
- Keep Pathfinder implementation under `ocean/pathfinder/` whenever possible.
- Allowed repo-level additions for this env:
  - `config/pathfinder.ini`
  - root planning docs such as `PATHFINDER_SPEC.md` and `AGENTS.md`
  - minimal build/integration fixes already needed for this PufferLib 5 branch.
- Do not modify `/home/claude/dogfight5`, `/home/claude/PufferLib`,
  `/home/claude/dogfight3`, or `/home/claude/dogfight4`.
- Avoid destructive Git operations. Do not run `git reset --hard`, mass
  deletes, or history rewrites unless the human explicitly asks.
- Do not add CPU fallback work unless explicitly requested. Native CUDA is the
  expected path on this machine.

## Environment Setup

This repo uses `uv` and an editable install:

```bash
cd /home/claude/pathfinder
uv venv --python 3.12 .venv
uv sync
uv pip install -e .
source .venv/bin/activate
python --version
```

The current venv is verified with:

- Python `3.12.3`
- Torch `2.10.0+cu128`
- CUDA runtime `12.8`
- `CUDA_HOME=/usr/local/cuda-12.8`
- `CC=clang`

The venv activation script has the same CUDA 12.8 hook pattern as
`dogfight5`:

```bash
source .venv/bin/activate
echo "$CUDA_HOME"
which nvcc
```

Expected:

```text
/usr/local/cuda-12.8
/usr/local/cuda-12.8/bin/nvcc
```

The Codex sandbox may not expose GPU devices. That can make PyTorch report
`torch.cuda.is_available() == False` and make `nvcc -arch=native` warn that no
valid GPU is visible. Native extension builds can still compile in the sandbox.
GPU training/sweeps may need to run outside the sandbox.

## Verified Build Baseline

The fresh workspace has already been prepared with:

- `uv sync`
- `uv pip install -e .`
- local `raylib-5.5_linux_amd64` copied from `/home/claude/dogfight5`
- the `dogfight5` `build.sh` fix that links versioned cuDNN/NCCL wheel
  libraries correctly.

Verified native CUDA extension builds:

```bash
source .venv/bin/activate
./build.sh g2048
./build.sh breakout
```

Both produced:

```text
pufferlib/_C.cpython-312-x86_64-linux-gnu.so
```

Do not use `./build.sh ENV --cpu` as the default smoke path for Pathfinder.

## Implementation Target

Expected files:

```text
ocean/pathfinder/pathfinder.h
ocean/pathfinder/pathfinder.c
ocean/pathfinder/binding.c
ocean/pathfinder/tests/
config/pathfinder.ini
```

Follow PufferLib 5 state-aware env patterns from:

```text
ocean/g2048/
ocean/boxoban/
ocean/craftax/
```

`ocean/maze/` is useful for navigation ideas but should not be copied blindly;
it is not the best state-memory reference on this branch.

## Current Design Defaults

- Fixed board size: 6x6.
- Spawn: `A1`.
- Hidden pawn: random non-`A1` cell, and generated maze must connect `A1` to
  it.
- Actions: four discrete moves.
- Observation:
  - 84 wall slots as floats:
    - `-1.0` unobserved
    - `0.0` observed open
    - `1.0` observed wall
  - normalized current x
  - normalized current y
  - total `OBS_SIZE = 86`
- First implementation should keep `train.state_buffer_size = 0` until
  deterministic state roundtrip tests exist.

## Test Expectations

Use test-driven development for Pathfinder behavior:

1. Add focused tests for wall indexing, maze solvability, reset observation,
   movement/wall reveal semantics, and terminal success.
2. Make the narrow tests pass.
3. Build native:

```bash
source .venv/bin/activate
./build.sh pathfinder
```

4. Run a short training smoke only after build and behavior tests pass:

```bash
source .venv/bin/activate
python -m pufferlib.pufferl train pathfinder --train.total-timesteps 2097152
```

If tests require helper binaries, put them under `ocean/pathfinder/tests/` and
keep them scoped to Pathfinder.

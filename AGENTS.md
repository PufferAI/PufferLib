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

If `python -m pufferlib.pufferl train pathfinder --train.gpus 1` fails inside
Codex with `Assertion 'device_count > 0 && "CUDA is not available"' failed`,
do not switch to CPU, rewrite CUDA setup, or debug the venv as the first
response. This is the managed sandbox hiding GPU devices. Re-run the same
training command with escalated/outside-sandbox execution so it can see
`/dev/nvidia*`.

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
- Hidden pawn: generated at the current curriculum distance from `A1`, and the
  maze must connect `A1` to it.
- Initial curriculum: `env.start_solution_len = 4` starts targets exactly 4 moves
  from `A1`; each solve advances the next generated puzzle by 1 move, capped at
  the board maximum.
- Set `env.curriculum_enabled = 0` to disable curriculum and start every
  generated map at the board maximum difficulty immediately.
- Actions: four discrete moves.
- Observation:
  - 84 wall slots as floats:
    - `-1.0` unobserved
    - `0.0` observed open
    - `1.0` observed wall
  - normalized current x
  - normalized current y
  - total `OBS_SIZE = 86`
- Rewards:
  - first-time wall discovery has no extra penalty beyond step cost
  - repeated known-wall hits get a small penalty
  - repeated known-wall hits terminate the attempt
  - first visits get a small discovery reward
  - revisits get a small penalty
  - immediate two-cell oscillation like `A1 -> B1 -> A1 -> B1` terminates with
    a large penalty
- Failed attempts reset the agent to `A1` on the same true map but clear all
  discovered wall/open observations back to `-1.0`. Solves generate the next
  map and advance curriculum.
- First implementation should keep `train.state_buffer_size = 0` until
  deterministic state roundtrip tests exist.

## Test Expectations

Use test-driven development for Pathfinder behavior:

1. Add focused tests for wall indexing, maze solvability, reset observation,
   movement/wall reveal semantics, and terminal success.
2. Make the narrow tests pass.
3. Run the Pathfinder core tests. The script compiles its own scoped C test
   binary from the current `ocean/pathfinder/` source before executing it; it
   does not use an existing `pufferlib/_C*.so` and is not testing stale native
   extension code:

```bash
source .venv/bin/activate
bash ocean/pathfinder/tests/run_all.sh
```

4. Build the native PufferLib extension. Do this before any Python-level
   Pathfinder smoke test, training run, or eval, because those paths import the
   built `pufferlib/_C*.so` artifact:

```bash
source .venv/bin/activate
./build.sh pathfinder
```

5. Run a short training smoke only after build and behavior tests pass:

```bash
source .venv/bin/activate
python -m pufferlib.pufferl train pathfinder --train.gpus 1
```

If tests require helper binaries, put them under `ocean/pathfinder/tests/` and
keep them scoped to Pathfinder.

When the human asks to "run all tests" for Pathfinder, do not manually iterate
the repo-root `tests/test_*.py` files. Those files are stale upstream or
experimental tests for older PufferLib APIs and optional dependencies, not the
current Pathfinder acceptance suite. Known expected failures there include
missing `pufferlib.emulation`, missing `pufferl.make_parser`, missing
`pufferlib/src/models.cu`, optional packages such as `heavyball`, `pandas`, and
`pyximport`, and CUDA visibility failures inside the Codex sandbox.

For Pathfinder, the supported baseline is:

```bash
source .venv/bin/activate
bash ocean/pathfinder/tests/run_all.sh
./build.sh pathfinder
```

If running any Python-level Pathfinder check after source edits, rebuild with
`./build.sh pathfinder` first so Python does not import an old native extension.

If the human explicitly asks for the stale repo-root Python tests anyway, state
that they are not the Pathfinder baseline before running them, and do not report
their expected failures as a Pathfinder regression.

## Quick Ops Notes

- The eval command supports `--load-model-path latest` to automatically pick the
  latest available checkpoint in the workspace.
- Verified working invocation:

  ```bash
  source .venv/bin/activate && DISPLAY=:0 python -m pufferlib.pufferl eval pathfinder --load-model-path latest
  ```

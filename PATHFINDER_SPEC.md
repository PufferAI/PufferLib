# Pathfinder Environment Spec

Status: draft, ready for user review

Workspace: `/home/claude/pathfinder`

Target env name: `pathfinder`

## Intent

Build a PufferLib 5 Ocean environment inspired by Milton Bradley's 1977
Pathfinder board game. The first version is a single-agent maze-solving task:
the environment generates a hidden, traversible barricade layout and a hidden
pawn, and the agent learns to discover walls while finding a path to the pawn.

The full two-player race is intentionally out of scope for the first version.
The board-game rules remain the source model for the grid, hidden pawn,
barriers, entry from column 1, adjacent movement, wall discovery, and terminal
goal detection.

## Rule Summary

Rules source:

- Official scanned rules PDF supplied by the user:
  `http://www.transformertoys.co.uk/images/instruction-scans/hasbro/Pathfinder.pdf`
- User-supplied video transcript describing play.

Board-game facts that matter for the env:

- The board is a 6x6 grid with rows `A-F` and columns `1-6`.
- Each player hides a short pawn on any square of their home grid.
- Each player secretly places barrier chips in slots around and between grid
  squares.
- The barricades must leave at least one route from a column-1 entrance square
  to the hidden pawn.
- Some column-1 entrances may be blocked. Some open column-1 entrances may be
  dead ends.
- A searching player starts by asking whether a column-1 square can be entered.
- After entering, the player asks to move to adjacent orthogonal squares.
  Diagonal movement is not allowed.
- If the requested edge has no barrier, the tall pawn moves and the turn
  continues. If the requested edge has a barrier, the pawn stays put, the wall
  is marked on the tracking grid, and turn control would pass in the board game.
- Backtracking is legal and uses the same adjacent-square calls.
- If an unblocked requested square contains the hidden pawn, the searcher wins.

The single-agent env treats every action as one RL step. It does not model
board-game turn passing or the pre-entry choice, because there is no opponent
policy in the first version.

## Board Model

Constants:

- `PF_ROWS = 6`
- `PF_COLS = 6`
- Cells use zero-based `(row, col)` internally.
- Row names map `A-F -> 0-5`.
- Column names map `1-6 -> 0-5`.

Wall slots are represented as graph edges:

- Vertical slots: `V[row][edge_col]`, shape `6 x 7`.
  - `edge_col = 0` is the left entry/exit edge for column 1.
  - `edge_col = 1..5` are internal east/west edges.
  - `edge_col = 6` is the right board boundary.
- Horizontal slots: `H[edge_row][col]`, shape `7 x 6`.
  - `edge_row = 0` is the top board boundary.
  - `edge_row = 1..5` are internal north/south edges.
  - `edge_row = 6` is the bottom board boundary.

Total wall observation slots: `6*7 + 7*6 = 84`.

Boundary behavior:

- Left boundary edges are real Pathfinder entry/exit edges and may be open or
  blocked by maze generation.
- Top, right, and bottom boundary edges are hard board boundaries. Their truth
  value is `1` and the agent can discover them by attempting to move off-board.

## Maze Generation

The generator must produce legal, solvable layouts without hand-authored maps.

Episode generation:

1. Pick a hidden pawn cell. Default: uniform over all non-`A1` cells.
2. Initialize all wall slots to blocked.
3. Open the left entry edge for `A1`.
4. Carve at least one orthogonal path from `A1` to the hidden pawn.
5. Store `A1` as the agent spawn for the episode.
6. Add false branches and optional loops without breaking the solved-path
   invariant.
7. Optionally open additional column-1 entries. These can connect to useful
   routes or dead-end branches.
8. Validate with BFS that `A1` reaches the pawn.

Default generator style:

- Use a randomized depth-first or randomized Prim-style graph carver over the
  6x6 cell graph.
- Preserve at least one path from `A1` to the pawn.
- Add branch density as a difficulty knob rather than trying to exactly copy
  human barricade layouts.

Config knobs:

- `branch_prob`: probability of adding false branches from the main route.
- `loop_prob`: probability of opening extra internal edges after carving.
- `extra_entry_prob`: probability each non-`A1` column-1 entrance is open.
- `min_solution_len`: reject mazes with a shortest solution path below this.
- `max_solution_len`: reject mazes with a shortest solution path above this.
  Use a small value for early curriculum so the hidden pawn starts close to
  `A1`; `0` disables the upper bound.
- `max_steps`: timeout.
- `seed`: inherited from vector env config.

The first implementation should keep board size fixed at 6x6. Larger synthetic
boards can be a later variant after the baseline trains.

## Action Space

Use one categorical action head.

Action count: `4`

- `0`: move north.
- `1`: move east.
- `2`: move south.
- `3`: move west.

Action semantics:

- Reset starts the agent at `A1`.
- Cardinal moves test the corresponding wall slot.
- If the tested slot is blocked, reveal it and keep the same position.
- If the tested slot is open, reveal it and move to the destination cell.
- Moving west from column 1 through an open left edge is treated as an exit
  attempt and leaves the agent in place in v1. This keeps action space simple
  and avoids an outside-grid state. Full retreat/re-entry can be added later.
- Attempts to move through top, right, or bottom board boundaries reveal a
  blocked boundary slot and keep the agent in place.

## Observation Space

Use a float observation tensor.

Observation layout:

1. `wall_obs[84]`
   - `-1.0`: unobserved wall slot.
   - `0.0`: observed open slot.
   - `1.0`: observed wall slot.
2. `pos_x`
   - Current column normalized to `[0, 1]`.
3. `pos_y`
   - Current row normalized to `[0, 1]`.

Observation size: `86`.

The hidden pawn location is not directly observed. It is only discovered when
the agent enters or moves into that square through an open edge.

## Rewards and Termination

Default reward model:

- `+1.0` for reaching the hidden pawn.
- `-0.001` per step.
- `0.0` extra penalty for hitting a newly discovered wall; the agent paid the
  step cost but gained information.
- `-0.01` for hitting a wall that was already known.
- `-0.01` for impossible movement, such as attempting to exit through the
  left edge in v1.
- `0.0` for a successful nonterminal move, except for the step penalty.

Termination:

- Success: agent reaches the hidden pawn.
- Timeout: `tick >= max_steps`.

Logged metrics:

- `perf`: `1.0` on success, `0.0` on timeout.
- `score`: success reward adjusted by path efficiency.
- `episode_return`
- `episode_length`
- `success`
- `wall_hits`
- `known_walls`
- `known_open_edges`
- `shortest_path_len`
- `agent_path_len`
- `n`

## PufferLib Integration

Expected files:

- `ocean/pathfinder/pathfinder.h`
- `ocean/pathfinder/pathfinder.c`
- `ocean/pathfinder/binding.c`
- `ocean/pathfinder/tests/`
- `config/pathfinder.ini`

The env should follow current PufferLib 5 Ocean patterns:

- Define `Log`.
- Define `State`.
- Keep rollout-local future-affecting fields inside `State`.
- Define env struct `Pathfinder` with required pointers:
  - `float* observations`
  - `float* actions`
  - `float* rewards`
  - `float* terminals`
  - `int num_agents`
  - `Log log`
  - `State state`
  - `unsigned int rng`
- In `binding.c`, set:
  - `OBS_SIZE 86`
  - `NUM_ATNS 1`
  - `ACT_SIZES {4}`
  - `OBS_TENSOR_T FloatTensor`
  - `Env Pathfinder`
  - `puffer_state_refresh(Pathfinder* env)` to rebuild observations from
    restored state.
- Keep `state_buffer_size = 0` initially unless state-memory tests exist.

`g2048`, `boxoban`, and `craftax` are the nearest 5.0 state-aware references.
`maze` is useful for navigation ideas but is not state-memory complete on this
branch.

## Testing

Add focused tests before training:

- Wall index mapping:
  - Every cardinal move maps to the expected `V` or `H` slot.
- Observation initialization:
  - All wall observations start at `-1.0`.
  - Position is `A1` after reset.
- Maze legality:
  - Every generated maze has at least one open column-1 entrance.
  - BFS from open column-1 entrances reaches the hidden pawn.
  - `A1` is connected to the hidden pawn.
  - Generated truth walls never allow top, right, or bottom off-board movement.
- Step semantics:
  - Open edge reveals `0.0` and moves.
  - Closed edge reveals `1.0` and does not move.
  - Repeated known wall hit keeps the same position.
  - West move from column 1 through an open left edge reveals the edge but does
    not move in v1.
  - Reaching the hidden pawn sets terminal and success log.
- State roundtrip:
  - Save state after partial exploration.
  - Restore into a fresh env.
  - Verify observations and scripted future steps match.

Verification commands after implementation:

```bash
source .venv/bin/activate
./build.sh pathfinder
python -m pufferlib.pufferl train pathfinder --train.total-timesteps 2097152
```

Native GPU builds are the expected path for this workspace.

## Initial Training Config

Start conservative:

- `vec.total_agents = 8192` or `16384`
- `vec.num_buffers = 2`
- `env.max_solution_len = 4` for early closer-target curriculum
- `train.gpus = 1`
- `train.total_timesteps = 100M` for real runs, at least `2,097,152` for smoke
  tests with the initial `8192 x 128` rollout geometry.
- `train.horizon = 128`
- `train.minibatch_size = 32768`
- `train.learning_rate = 0.001`
- `train.gamma = 0.995`
- `train.use_rnn = 1`
- `train.state_buffer_size = 0`
- `train.cl_frac = 0`

Reasoning: the task is partially observable and memory-dependent. Recurrent
policy support is likely useful because the agent must remember discovered
walls and path context, even though the observation also contains discovered
wall memory.

## Later Extensions

After the single-agent solver works:

- Add curriculum over branch density, loop density, and solution length.
- Add larger synthetic boards while retaining a 6x6 rules mode.
- Add direct coordinate-call action variants for closer board-game fidelity.
- Add explicit outside-grid entry/re-entry actions for closer board-game
  fidelity.
- Add two-player self-play where each side generates or chooses a maze and
  races to find the opposing pawn.
- Add render mode showing true hidden maze, observed tracking grid, current
  pawn, and discovered path.

# Pathfinder Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first PufferLib 5 `pathfinder` Ocean environment: a generated 6x6 hidden-wall maze, fixed `A1` spawn, hidden pawn target, four movement actions, and wall-memory observations.

**Architecture:** Implement the environment as a compact C env under `ocean/pathfinder/`, following `g2048` for state ownership and `boxoban`/`maze` for navigation style. Put deterministic helpers in `pathfinder.h` so focused C tests can include them directly before the PufferLib CUDA binding is built.

**Tech Stack:** C, PufferLib 5 Ocean static vec binding, `uv` editable install, native CUDA build via `./build.sh pathfinder`.

---

### Task 1: Core Pathfinder Model and Tests

**Files:**
- Create: `ocean/pathfinder/pathfinder.h`
- Create: `ocean/pathfinder/tests/test_pathfinder_core.c`
- Create: `ocean/pathfinder/tests/run_all.sh`

- [ ] **Step 1: Write the failing core test**

Create `ocean/pathfinder/tests/test_pathfinder_core.c` with tests that include `../pathfinder.h` and check:

- `PATHFINDER_OBS_SIZE == 86`
- `PATHFINDER_NUM_WALLS == 84`
- reset starts at row `0`, col `0`
- all wall observations start at `-1.0f`
- generated mazes connect `A1` to the hidden pawn
- moving through an open edge reveals `0.0f` and moves
- moving into a blocked edge reveals `1.0f` and does not move

- [ ] **Step 2: Run the core test to verify RED**

Run:

```bash
cd /home/claude/pathfinder
bash ocean/pathfinder/tests/run_all.sh
```

Expected: FAIL because `ocean/pathfinder/pathfinder.h` does not exist yet.

- [ ] **Step 3: Implement minimal core model**

Create `ocean/pathfinder/pathfinder.h` with:

- constants for 6x6 cells, 84 wall slots, 86 obs size, 4 actions
- `Log`, `State`, and `Pathfinder` structs
- wall indexing helpers for vertical and horizontal slots
- deterministic `pathfinder_generate_maze`
- `pathfinder_update_observations`
- `pathfinder_reset`
- `pathfinder_step`
- `refresh_state`, `init`, `c_reset`, `c_step`, `c_close`

- [ ] **Step 4: Run core test to verify GREEN**

Run:

```bash
cd /home/claude/pathfinder
bash ocean/pathfinder/tests/run_all.sh
```

Expected: PASS for core model tests.

### Task 2: PufferLib Binding and Config

**Files:**
- Create: `ocean/pathfinder/binding.c`
- Create: `ocean/pathfinder/pathfinder.c`
- Create: `config/pathfinder.ini`

- [ ] **Step 1: Write the failing build expectation**

Run:

```bash
cd /home/claude/pathfinder
source .venv/bin/activate
./build.sh pathfinder
```

Expected: FAIL because `ocean/pathfinder/binding.c` does not exist.

- [ ] **Step 2: Add binding**

Create `ocean/pathfinder/binding.c` using the `g2048`/`boxoban` pattern:

- `#include "pathfinder.h"`
- `#define OBS_SIZE PATHFINDER_OBS_SIZE`
- `#define NUM_ATNS 1`
- `#define ACT_SIZES {4}`
- `#define OBS_TENSOR_T PrecisionTensor`
- `#define Env Pathfinder`
- `static inline void puffer_state_refresh(Pathfinder* env) { refresh_state(env); }`
- `#include "vecenv.h"`
- `my_init` reads `branch_prob`, `loop_prob`, `extra_entry_prob`, `min_solution_len`, and `max_steps`
- `my_log` exports `perf`, `score`, `episode_return`, `episode_length`, `success`, `wall_hits`, `known_walls`, `known_open_edges`, `shortest_path_len`, `agent_path_len`

- [ ] **Step 3: Add local demo/standalone file**

Create `ocean/pathfinder/pathfinder.c` as a small random-action local runner matching the simple env style used by `g2048.c` and `boxoban.c`.

- [ ] **Step 4: Add config**

Create `config/pathfinder.ini` with:

- `[base] env_name = pathfinder`
- `[vec] total_agents = 8192`, `num_buffers = 2`, `num_threads = 0`
- `[env] branch_prob`, `loop_prob`, `extra_entry_prob`, `min_solution_len`, `max_steps`
- `[policy] hidden_size = 512`, `num_layers = 2`, `expansion_factor = 1`
- `[train] gpus = 1`, `total_timesteps = 100000000`, `horizon = 128`, `minibatch_size = 32768`, `learning_rate = 0.001`, `gamma = 0.995`, `use_rnn = 1`, `state_buffer_size = 0`, `cl_frac = 0`

- [ ] **Step 5: Run native build to verify GREEN**

Run:

```bash
cd /home/claude/pathfinder
source .venv/bin/activate
./build.sh pathfinder
```

Expected: PASS and build `pufferlib/_C.cpython-312-x86_64-linux-gnu.so`.

### Task 3: Training Smoke and Documentation Check

**Files:**
- Modify: `PATHFINDER_SPEC.md` only if implementation exposes a deliberate spec correction.
- Modify: `AGENTS.md` only if commands or defaults changed.

- [ ] **Step 1: Run core tests**

Run:

```bash
cd /home/claude/pathfinder
bash ocean/pathfinder/tests/run_all.sh
```

Expected: PASS.

- [ ] **Step 2: Run native build**

Run:

```bash
cd /home/claude/pathfinder
source .venv/bin/activate
./build.sh pathfinder
```

Expected: PASS.

- [ ] **Step 3: Run short train smoke if GPU access permits**

Run:

```bash
cd /home/claude/pathfinder
source .venv/bin/activate
python -m pufferlib.pufferl train pathfinder --train.total-timesteps 2097152
```

Expected: The run starts, validates the compiled env name, and completes at least one rollout/update. If sandbox GPU access blocks training, record the exact failure and do not treat it as an env logic failure.

- [ ] **Step 4: Review docs and status**

Run:

```bash
cd /home/claude/pathfinder
rg -n "Action count|Observation size|OBS_SIZE|ACT_SIZES|A1|state_buffer_size" PATHFINDER_SPEC.md AGENTS.md
git status --short --branch
```

Expected: docs still say 4 actions, `OBS_SIZE = 86`, fixed `A1` spawn, and initial state memory off.

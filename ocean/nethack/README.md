# NetHack environment for PufferLib

C-level NLE binding for the PufferLib RL framework. Each env owns a
per-env `nle_ctx_t` holding all of NetHack's mutable game state, with a
private 64 MB memory arena. `libnethack.so` is linked directly (no
dlopen). Auto-dismiss for prompts, compile-time observation selection,
reward shaping, multi-threaded OMP stepping.

---

## Full setup (from scratch on any HPC)

### 1. Clone PufferLib + vendored NLE

```bash
git clone https://github.com/PufferAI/PufferLib.git && cd PufferLib
git checkout 4.0

# Clone the modified NLE into vendor/nle
git clone https://github.com/liujonathan24/NetHack.git vendor/nle
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[nethack]"    # installs pufferlib + nethack deps (torch, numpy, wandb, etc.)
```

If `[nethack]` extras aren't defined yet, use:
```bash
pip install -e .
pip install bz2file           # if system libbz2 headers missing
```

### 3. System modules (cluster-specific)

You need a C compiler (clang preferred, gcc works), CUDA toolkit, and
an OpenMP runtime. On Princeton Della:

```bash
module load cudatoolkit/12.8 intel-oneapi/2024.2
```

On other clusters, find equivalents for:
- **CUDA**: `nvcc` for GPU training backend
- **OpenMP**: `libiomp5.so` (Intel) or `libgomp.so` (GCC) — needed at link and runtime
- **clang** (optional but preferred): `build.sh` uses clang flags by default. Set `CC=gcc` if clang isn't available, but note `-ferror-limit` must be removed from `build.sh`.

### 4. Build libnethack.so

```bash
# First-time only: configure cmake
cd vendor/nle/src
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cd ../../../..

# Build (always from repo root)
make -C vendor/nle/src/build nethack -j$(nproc)
```

Produces `vendor/nle/src/build/libnethack.so` and data files in
`vendor/nle/src/build/dat/` (including `nhdat`).

If cmake fails with missing `bz2`, install libbz2-dev
(`apt install libbz2-dev` or `yum install bzip2-devel`).

### 5. Build PufferLib C extension (_C.so)

```bash
bash build.sh nethack          # auto-detects CUDA; falls back to CPU
```

Produces `pufferlib/_C.cpython-*.so`. If you get linker errors about
`-liomp5`, make sure the Intel OpenMP module is loaded.

### 6. Set NETHACKDIR

Point to the directory containing `nhdat`:

```bash
export NETHACKDIR=$(pwd)/vendor/nle/src/build/dat
```

### 7. Verify

```bash
# Quick training test (runs on login node if GPU available)
puffer train nethack \
    --vec.total-agents 64 --vec.num-buffers 1 --vec.num-threads 4 \
    --train.gpus 1 --train.total-timesteps 1000000 --train.minibatch-size 64

# Interactive play
./live_view -i

# Random agent viewer
./live_view --random --steps 200
```

### 8. SLURM training

```bash
sbatch sweep_nethack.slurm     # hyperparameter sweep
# or single run:
puffer train nethack --wandb \
    --vec.total-agents 4096 --vec.num-buffers 1 --vec.num-threads 16 \
    --train.gpus 1 --train.total-timesteps 1000000000
```

### Rebuild after code changes

```bash
make -C vendor/nle/src/build nethack -j$(nproc)   # if vendor/nle changed
bash build.sh nethack                               # always (relinks _C.so)
```

### Pulling NLE updates

The modified NLE lives in a separate repo. To pull latest changes:

```bash
cd vendor/nle
git pull origin main
cd ../..
make -C vendor/nle/src/build nethack -j$(nproc)
bash build.sh nethack
```

To push NLE changes (after editing files under `vendor/nle/`):

```bash
cd vendor/nle
git add -A && git commit -m "description of changes"
git push origin main
cd ../..
```

Note: `vendor/nle/` has its own `.git` — it is NOT tracked by the
PufferLib repo. Add `vendor/nle` to PufferLib's `.gitignore` if it
isn't already.

### Troubleshooting

| Error | Fix |
|---|---|
| `clang: command not found` | `module load llvm` or set `CC=gcc` |
| `cannot find -liomp5` | `module load intel-oneapi/2024.2` (or equivalent) |
| `libiomp5.so: cannot open shared object file` | Same module, also at runtime |
| `cannot allocate memory in static TLS block` | Too many `__thread` vars — rebuild libnethack |
| `NETHACKDIR is misconfigured` | `export NETHACKDIR=$(pwd)/vendor/nle/src/build/dat` |
| `ZeroDivisionError` in train | Need `--train.gpus 1` (even on CPU-only, the CUDA build requires it) |
| Core dump / segfault at T>1 | Check that libnethack.so was rebuilt after latest source changes |

---

## Observation (compile-time selectable, default = chars only)

Each enabled field reserves a slice of a single flat `ByteTensor`
observation buffer. Override defaults with `-DNETHACK_USE_<FIELD>=1` in
build.sh's `EXTRA_CFLAGS`.

| Field        | Default | Bytes/element | Total (one env)              |
|--------------|--------:|--------------:|------------------------------|
| `chars`      |    1    | 1             | 1,659                        |
| `colors`     |    0    | 1             | 1,659                        |
| `specials`   |    0    | 1             | 1,659                        |
| `glyphs`     |    0    | 2 (le i16)    | 3,318                        |
| `blstats`    |    0    | 4 (i32, trunc)| 108                          |
| `message`    |    0    | 1             | 256                          |
| `inv`        |    0    | 1 (letters+oclasses) | 110                   |

`OBS_SIZE` is the sum of enabled fields.

## Action space (18 actions)

```
 0  N            8  N_RUN         16  >  (down)
 1  S            9  S_RUN         17  <  (up)
 2  W           10  W_RUN
 3  E           11  E_RUN
 4  NW          12  NW_RUN
 5  NE          13  NE_RUN
 6  SW          14  SW_RUN
 7  SE          15  SE_RUN
```

The 8 cardinal/intercardinal moves use vi-keys (kjhl ynbu). Long
"run" versions are uppercase (KJHL YNBU).

## Reward shaping (config-tunable via nethack.ini)

```
reward = score_coef    * (score - prev_score)          # game score delta
       + descent_coef  * max(0, depth - prev_depth)    # new max depth bonus
       + scout_coef    * (new_tile_this_level)          # exploration bonus
       + illegal_penalty * (illegal_action)             # sub-prompt penalty
```

All coefficients are set in `config/nethack.ini` under `[env]`.

## Auto-dismiss hook

After each agent action, the harness inspects `misc[]`
(`in_yn_function`, `in_getlin`, `xwaitingforspace`) plus a heuristic
message-ends-in-`?` check:

- `yn_function` prompts: auto-answered `y` (commit to the action)
- `getlin` prompts: auto-dismissed with ESC
- `--More--` / `xwaitforspace`: auto-dismissed with space/ESC

Capped at `NETHACK_AUTODISMISS_MAX=64` iterations. Episodes also
auto-reset after `NETHACK_MAX_EPISODE_STEPS=10000` steps.

## Per-episode log entries

| Key                | Meaning                                                |
|--------------------|--------------------------------------------------------|
| `perf` / `score`   | Final NetHack score                                    |
| `depth`            | Final dungeon level                                    |
| `episode_return`   | Sum of shaped reward                                   |
| `episode_length`   | Number of c_steps                                      |
| `valid_moves`      | c_steps where NetHack's turn counter advanced          |
| `illegal_actions`  | c_steps where the agent triggered a sub-prompt         |
| `new_tiles`        | Unique tiles entered this episode                      |

## Standalone tools

```bash
# Live viewer (interactive play)
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./live_view -i

# Live viewer (random agent)
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./live_view --random --steps 500

# Live viewer (replay a recorded trajectory)
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./live_view --replay path/to/recording.bin
```

Build live_view from source:
```bash
clang -O2 -I vendor/nle/src/include -I vendor/nle/src/build/include \
    -I vendor/nle/src/third_party/deboost.context/include \
    -DDEFAULT_WINDOW_SYS=\"rl\" -DDLB -DNLE_ALLOW_SEEDING \
    -DNLE_PER_ENV_FILES=1 -DNLE_PER_ENV_FLAGS=1 -DNLE_USE_ARENA_FREE=1 \
    -DNLE_USE_TILES -DNOCLIPPING -DNOCWD_ASSUMPTIONS -DNOMAIL -DNOTPARMDECL \
    -DNETHACK_USE_BLSTATS=1 \
    ocean/nethack/live_view.c -o live_view \
    -L./vendor/nle/src/build -lnethack -lm -lbz2 -lpthread \
    -Wl,-rpath=$(pwd)/vendor/nle/src/build
```

## Performance

| Configuration | SPS | Notes |
|---|---|---|
| N=512 T=4 (4M model) | 40K | GPU-bound (train=68%) |
| N=4096 T=4 (4M model) | 128K | Balanced (env=66%, train=30%) |
| N=4096 T=16 (4M model) | ~400K | More threads = more env throughput |
| N=8192 T=64 B=2 (sweep) | ~1M | Full utilization with double-buffering |

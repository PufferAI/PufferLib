# PufferLib Ocean Environment Guide

Quick reference for implementing C-based RL environments in PufferLib.

## File Structure

```
pufferlib/ocean/{env_name}/
├── {env_name}.h      # C implementation (header-only)
├── binding.c         # Python-C glue (~20 lines)
└── {env_name}.py     # Python wrapper

pufferlib/config/ocean/{env_name}.ini  # Training config
```

Build: `python setup.py build_ext --inplace --force`

## 1. C Header (`{env_name}.h`)

### Required Structs

```c
// Log: ONLY floats, last field must be `n`
typedef struct Log {
    float episode_return;
    float episode_length;
    float score;
    float perf;           // 0-1 normalized metric
    // ... custom metrics ...
    float n;              // REQUIRED last: episode count
} Log;

// Main env struct
typedef struct EnvName {
    float* observations;      // or char* for discrete obs
    float* actions;           // ALWAYS float* (even discrete)
    float* rewards;
    unsigned char* terminals;
    Log log;
    Client* client;           // raylib, NULL until render
    // ... env state ...
} EnvName;
```

### Required Functions

| Function | Purpose |
|----------|---------|
| `init(Env*)` | Allocate internal buffers |
| `c_reset(Env*)` | Reset episode state |
| `c_step(Env*)` | Advance simulation |
| `c_render(Env*)` | Raylib rendering |
| `c_close(Env*)` | Free memory |
| `compute_observations(Env*)` | Fill obs buffer |
| `add_log(Env*, ...)` | Accumulate stats |

### Step Pattern

```c
void c_step(Env* env) {
    env->tick++;
    env->rewards[0] = 0;
    env->terminals[0] = 0;

    // ... physics/game logic ...

    if (terminal_condition) {
        env->terminals[0] = 1;
        add_log(env, ...);
        c_reset(env);
        return;
    }
    compute_observations(env);
}
```

### Logging Pattern

```c
void add_log(Env* env) {
    env->log.episode_return += env->episodic_return;
    env->log.episode_length += env->tick;
    env->log.score += env->score;
    env->log.n += 1.0f;  // increment episode count
}
```

## 2. Binding (`binding.c`)

```c
#include "{env_name}.h"

#define Env EnvName
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->param1 = unpack(kwargs, "param1");
    env->param2 = unpack(kwargs, "param2");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    return 0;
}
```

## 3. Python Wrapper (`{env_name}.py`)

```python
import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.{env_name} import binding

class EnvName(pufferlib.PufferEnv):
    def __init__(self, num_envs=16, render_mode=None, buf=None,
                 param1=100, param2=0.5, **kwargs):

        self.single_observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(OBS_SIZE,), dtype=np.float32
        )
        # Continuous: Box    Discrete: Discrete(n)
        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(ACT_SIZE,), dtype=np.float32
        )

        self.num_agents = num_envs
        self.render_mode = render_mode
        super().__init__(buf)

        # CRITICAL for continuous actions:
        self.actions = self.actions.astype(np.float32)

        c_envs = []
        for i in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[i:i+1],
                self.actions[i:i+1],
                self.rewards[i:i+1],
                self.terminals[i:i+1],
                self.truncations[i:i+1],
                i,  # seed
                param1=param1,
                param2=param2,
            ))
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed or 0)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        log = binding.vec_log(self.c_envs)
        if log:
            info.append(log)
        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)
```

## 4. Config (`pufferlib/config/ocean/{env_name}.ini`)

```ini
[base]
package = ocean
env_name = puffer_{env_name}

[vec]
num_envs = 8

[env]
num_envs = 1024
param1 = 100
param2 = 0.5

[train]
total_timesteps = 100_000_000
learning_rate = 0.0003
gamma = 0.99
# ... PPO hyperparams ...
```

## Reference Environments

| Env | Use For |
|-----|---------|
| `drone_race/` | Continuous actions, quaternions, RK4 physics |
| `drone_swarm/` | Multi-agent continuous |
| `snake/` | Multi-agent discrete, grid world |
| `target/` | Simple tutorial, well-commented |
| `impulse_wars/` | Box2D physics integration |

## Common Patterns

### Vector/Quaternion Math (from dronelib.h)

```c
typedef struct { float x, y, z; } Vec3;
typedef struct { float w, x, y, z; } Quat;

Vec3 add3(Vec3 a, Vec3 b);
Vec3 sub3(Vec3 a, Vec3 b);
Vec3 scalmul3(Vec3 a, float s);
float dot3(Vec3 a, Vec3 b);
float norm3(Vec3 a);

Quat quat_mul(Quat a, Quat b);
void quat_normalize(Quat* q);
Vec3 quat_rotate(Quat q, Vec3 v);
Quat quat_inverse(Quat q);
```

### Observation Normalization

```c
// Normalize to roughly [-1, 1]
env->observations[0] = position.x / MAX_X;
env->observations[1] = velocity.x / MAX_VEL;
env->observations[2] = quat.w;  // already [-1, 1]
```

### Action Handling

```c
// Continuous: actions already in [-1, 1]
float throttle = (env->actions[0] + 1.0f) * 0.5f;  // remap to [0, 1]
float elevator = env->actions[1];  // keep [-1, 1]

// Discrete trigger
bool fire = env->actions[4] > 0.5f;
```

### Raylib Rendering

```c
void c_render(Env* env) {
    if (env->client == NULL) {
        InitWindow(WIDTH, HEIGHT, "Env Name");
        SetTargetFPS(60);
        env->client = calloc(1, sizeof(Client));
    }

    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    // ... draw stuff ...
    EndDrawing();
}
```

## Performance Tips

1. **No allocations after init** - malloc only in `init()`
2. **Pass structs by pointer** - avoid copies
3. **Inline small functions** - `static inline`
4. **Batch operations** - process all agents in tight loops
5. **Avoid divisions** - precompute `1/x` where possible

## Checklist for New Env

- [ ] Create folder `pufferlib/ocean/{name}/`
- [ ] Implement `{name}.h` with all required functions
- [ ] Create `binding.c` with `my_init()` and `my_log()`
- [ ] Create `{name}.py` Python wrapper
- [ ] Create `pufferlib/config/ocean/{name}.ini`
- [ ] Build: `python setup.py build_ext --inplace --force`
- [ ] Test: `from pufferlib.ocean.{name} import EnvName`

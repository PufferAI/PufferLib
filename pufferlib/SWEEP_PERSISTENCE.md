# Sweep Persistence & Override

This document describes the crash recovery and override injection features for Protein sweeps.

## Overview

Protein sweeps can run for days across hundreds of training runs. Two problems arise:

1. **Crashes lose progress** - A crash at run 50 of 100 means starting over
2. **No way to inject knowledge** - Users can't guide the sweep based on observations

This implementation solves both:

- **Persistence**: Sweep state is saved to JSON after each run, enabling crash recovery
- **Override**: Users can inject specific hyperparameters mid-sweep via a JSON file

## Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROTEIN SWEEP FLOWCHART                             │
│                     ★ = PERSISTENCE FEATURES                                │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Protein()     │
                              │   __init__      │
                              └────────┬────────┘
                                       │
                         ★ ┌───────────▼───────────┐
                           │ Clean orphaned .tmp   │
                           │ files from last crash │
                           └───────────┬───────────┘
                                       │
                         ★ ┌───────────▼───────────┐
                           │ State file exists?    │
                           └───────────┬───────────┘
                                       │
                          ┌────────────┴────────────┐
                          │ YES                     │ NO
                    ★ ┌───▼────────────────┐        │
                      │ Load state:        │        │
                      │ • suggestion_idx   │        │
                      │ • observations     │        │
                      │ • score bounds     │        │
                      │ Print "Resumed..." │        │
                      └───┬────────────────┘        │
                          │                         │
                          └────────────┬────────────┘
                                       │
                              ┌────────▼────────┐
                              │  Ready for      │
                              │  sweep loop     │
                              └────────┬────────┘
                                       │
═══════════════════════════════════════╪═══════════════════════════════════════
                              SWEEP LOOP (per run)
═══════════════════════════════════════╪═══════════════════════════════════════
                                       │
        ┌─────────────────────────────►│
        │                     ┌────────▼────────┐
        │                     │    suggest()    │
        │                     │ suggestion_idx++│ ← in memory only
        │                     └────────┬────────┘
        │                              │
        │                ★ ┌───────────▼───────────┐
        │                  │ Override file exists? │
        │                  └───────────┬───────────┘
        │                              │
        │               ┌──────────────┴──────────────┐
        │               │ YES                         │ NO
        │         ★ ┌───▼────────────────────┐        │
        │           │ Read override.json     │        │
        │           │ Pop first suggestion   │        │
        │           │ Atomic write remaining │        │
        │           │ Apply params to fill   │        │
        │           │ Print "OVERRIDE:..."   │        │
        │           └───┬────────────────────┘        │
        │               │                             │
        │               │                ┌────────────▼────────────┐
        │               │                │ Normal suggestion:      │
        │               │                │ • Sobol (early runs)    │
        │               │                │ • GP optimization       │
        │               │                └────────────┬────────────┘
        │               │                             │
        │               └──────────────┬──────────────┘
        │                              │
        │                     ┌────────▼────────┐
        │                     │ Return hypers   │
        │                     │ to pufferl.py   │
        │                     └────────┬────────┘
        │                              │
        │                     ┌────────▼────────┐
        │                     │  TRAINING RUN   │
        │                     └────────┬────────┘
        │                              │
        │                 ┌────────────┴────────────┐
        │                 │                         │
        │           ┌─────▼─────┐            ┌──────▼──────┐
        │           │  CRASH    │            │  COMPLETES  │
        │           └─────┬─────┘            └──────┬──────┘
        │                 │                         │
        │                 │                ┌────────▼────────┐
        │                 │                │    observe()    │
        │                 │                └────────┬────────┘
        │                 │                         │
        │                 │           ┌─────────────┴─────────────┐
        │                 │           │                           │
        │                 │     ┌─────▼─────┐              ┌──────▼──────┐
        │                 │     │ NaN/Fail  │              │   Success   │
        │                 │     └─────┬─────┘              └──────┬──────┘
        │                 │           │                           │
        │                 │     ★ ┌───▼───────────────┐    ★ ┌────▼────────────┐
        │                 │       │ Add to            │      │ Add to          │
        │                 │       │ failure_obs       │      │ success_obs     │
        │                 │       │ _save_state()     │      │ _save_state()   │
        │                 │       └─────────┬─────────┘      └────────┬────────┘
        │                 │                 │                         │
        │                 │                 └────────────┬────────────┘
        │                 │                              │
        │                 │                              ▼
        │                 │               ★ ┌─────────────────┐
        │                 │                 │ {project}_sweep │
        │                 │                 │ .json updated   │
        │                 │                 └────────┬────────┘
        │                 │                          │
        │                 ▼                          │
        │           ┌─────────────┐                  │
        │           │ Wait for    │                  │
        │           │ restart     │                  │
        │           └─────────────┘                  │
        │                                            │
        └────────────────────────────────────────────┘
```

## File Formats

### State File: `{project}_sweep.json`

Created automatically. Named after your wandb/neptune project.

```json
{
  "suggestion_idx": 42,
  "success_observations": [
    {
      "input": [0.123, -0.456, ...],
      "output": 0.85,
      "cost": 100000000
    }
  ],
  "failure_observations": [
    {
      "input": [0.789, -0.012, ...],
      "output": NaN,
      "cost": 50000000,
      "is_failure": true
    }
  ],
  "min_score": 0.12,
  "max_score": 0.85,
  "log_c_min": 17.42,
  "log_c_max": 18.42
}
```

### Override File: `{project}_override.json`

Create this file manually to inject hyperparameters into the next run(s).

```json
{
  "suggestions": [
    {
      "params": {
        "train/learning_rate": 0.0005,
        "train/ent_coef": 0.01,
        "env/reward_scale": 0.5
      },
      "reason": "Testing higher entropy for exploration"
    },
    {
      "params": {
        "train/learning_rate": 0.0001
      },
      "reason": "Testing lower LR after seeing instability"
    }
  ]
}
```

**Behavior:**
- Each `suggest()` call pops the first entry from the list
- Partial params are merged with defaults (only specify what you want to override)
- File is deleted when all suggestions are consumed
- Invalid parameter paths are skipped with a warning

## Usage

### Basic Sweep (persistence automatic)

```bash
python -m pufferlib.pufferl sweep puffer_dogfight --wandb --wandb-project df9
```

State saves to `df9_sweep.json` after each completed run.

### Crash Recovery

Just restart the same command:

```bash
# Crashed at run 47...
# Just run again:
python -m pufferlib.pufferl sweep puffer_dogfight --wandb --wandb-project df9
# Output: [Protein] Resumed from df9_sweep.json: 46 obs, idx=47
```

### Injecting Overrides

While a sweep is running (or before starting):

```bash
# Create override file
cat > df9_override.json << 'EOF'
{
  "suggestions": [
    {
      "params": {"train/learning_rate": 0.0003, "train/ent_coef": 0.015},
      "reason": "Promising region from wandb analysis"
    }
  ]
}
EOF
```

Next `suggest()` call will use these params instead of GP suggestion:
```
[Protein] OVERRIDE: Promising region from wandb analysis
```

## Crash Recovery Behavior

| Scenario | What's Preserved | What's Lost |
|----------|------------------|-------------|
| Crash during training run | All completed runs | Current run only |
| Crash during `_save_state()` | All previous state | Nothing (atomic write) |
| Corrupted state file | Nothing | Starts fresh with warning |
| Corrupted override file | All state | Override deleted with warning |

### Atomic Write Pattern

All file writes use atomic replacement to prevent corruption:

```
1. Write to {file}.tmp
2. os.replace(tmp, file)  ← atomic on POSIX
3. On failure: delete .tmp, original intact
```

## Error Handling

| Error | Behavior |
|-------|----------|
| State file corrupted/truncated | Warning printed, starts fresh |
| State file deleted mid-load | Warning printed, starts fresh |
| Override file corrupted | Warning printed, file deleted, normal suggestion |
| Override path doesn't exist | Warning printed, path skipped, other params applied |
| Disk full during save | Warning printed, .tmp cleaned up, training continues |

## API Reference

### New Protein Methods

```python
@staticmethod
def _json_default(obj):
    """JSON serializer for numpy types."""

def _save_state(self):
    """Save sweep state to JSON. Called after each observe()."""

def _load_state_if_exists(self):
    """Load state on init. Cleans orphaned .tmp files."""

def _check_override(self):
    """Check for and consume override file. Returns params dict or None."""
```

### Config Keys

Added to sweep config (injected by `pufferl.py`):

```python
sweep_config['state_file'] = f'{project}_sweep.json'
sweep_config['override_file'] = f'{project}_override.json'
```

## Testing

```bash
python tests/test_sweep_persistence_and_override.py
```

Tests cover:
- Save/load round-trip
- Crash recovery with state preservation
- Override consumption (single and multiple)
- Partial override (merge with defaults)
- Corrupted/empty/missing file handling
- Race conditions (file deleted between check and open)
- Atomic write failure recovery
- Orphaned .tmp cleanup
- Analysis helper functions (read_sweep_results, create_override)

**Note:** `tests/test_sweep_hyper.py` is a pre-existing file (not part of this PR) that seems to be a research script for tuning GP hyperparameters. It requires a manually-generated `sweep_observations.pkl` file and will fail if run standalone.

## Implementation Notes

1. **State saved in `observe()`, not `suggest()`**: If training crashes, `suggestion_idx` isn't persisted. On restart, may re-suggest similar params. This is intentional - we don't save until we have results.

2. **Override params are not validated**: Values outside sweep ranges are allowed. This lets users intentionally explore outside the configured space.

3. **Failure observations are tracked**: NaN scores go to `failure_observations`. The GP doesn't train on these, but they're preserved for debugging.

4. **Near-duplicate filtering**: Success observations within `EPSILON` distance are deduplicated, keeping the most recent.

## Analyzing Sweep Results

The state file stores **normalized** hyperparameter values in [-1, 1] range, which is unreadable without the sweep config. Two helper functions make analysis easy:

### read_sweep_results()

Denormalizes state file to human-readable dicts:

```python
from pufferlib.sweep import read_sweep_results
from pufferlib.pufferl import load_config_file

# Load config
config = load_config_file('pufferlib/config/ocean/dogfight.ini')

# Read and denormalize sweep state
results = read_sweep_results('df9_sweep.json', config['sweep'])

# Results are sorted by score (best first)
best = results[0]
print(f"Best score: {best['score']:.3f}")
print(f"Learning rate: {best['params']['train/learning_rate']}")
```

### create_override()

Programmatically inject hyperparameters into the next sweep run(s):

```python
from pufferlib.sweep import create_override

# Single override (note: params and reason must be lists)
create_override('df9_override.json', [{
    'train/learning_rate': 0.00069,
    'train/ent_coef': 0.0069,
}], reason=['Testing hypothesis from correlation analysis'])

# Multiple overrides
create_override('df9_override.json', [
    {'train/learning_rate': 0.001},
    {'train/learning_rate': 0.002},
    {'train/learning_rate': 0.003},
], reason=['low LR', 'medium LR', 'high LR'])
```

### Jupyter Notebook Workflow

```python
from pufferlib.sweep import read_sweep_results, create_override
from pufferlib.pufferl import load_config_file
import pandas as pd

# Load config and read sweep state
config = load_config_file('pufferlib/config/ocean/dogfight.ini')
results = read_sweep_results('df9_sweep.json', config['sweep'])

# Convert to DataFrame for analysis
df = pd.DataFrame([
    {**r['params'], 'score': r['score'], 'cost': r['cost']}
    for r in results
])

# Find best runs
df.sort_values('score', ascending=False).head(10)

# Correlation analysis - which params matter most?
df.corr()['score'].sort_values()

# Based on analysis, inject promising params
create_override('df9_override.json', [{
    'train/learning_rate': 0.00069,
    'train/ent_coef': 0.0069,
}], reason=['Testing correlation hypothesis'])
```

### Testing Analysis Functions

```bash
python tests/test_sweep_persistence_and_override.py
```

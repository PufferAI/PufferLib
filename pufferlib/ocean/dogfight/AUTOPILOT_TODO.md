# Autopilot System TODO

Technical debt and future improvements for the target aircraft autopilot system.

---

## Critical Issues

### 1. Python/C Enum Sync Problem
**Risk: Silent breakage if enums diverge**

```c
// autopilot.h
typedef enum { AP_STRAIGHT = 0, AP_LEVEL, ... } AutopilotMode;
```
```python
# dogfight.py
class AutopilotMode:
    STRAIGHT = 0  # Must manually match C
```

**Fix options:**
- [ ] Generate Python constants from C header at build time
- [ ] Add runtime validation that checks enum values match
- [ ] Add static_assert in C for enum count

### 2. No Vectorized set_autopilot
```python
def set_autopilot(self, env_idx=0, ...):  # Must call N times for N envs
```

**Fix:**
- [ ] Add `set_autopilot_all()` method
- [ ] Or accept `env_idx=None` to mean "all environments"
- [ ] Add C binding `vec_set_autopilot()` for efficiency

### 3. force_state() Doesn't Reset PID State
When teleporting plane via `force_state()`, autopilot PID state (`prev_vz`, `prev_bank_error`) retains stale values causing derivative spikes.

**Fix:**
- [ ] Reset autopilot PID state in `force_state()` C function
- [ ] Or add `reset_pid` parameter to force_state

### 4. No Mode Bounds Check
Invalid mode values (e.g., `mode=99`) silently become AP_STRAIGHT.

**Fix:**
- [ ] Add bounds check in `autopilot_set_mode()`
- [ ] Return error or clamp to valid range

---

## Curriculum Learning Gaps

### Mode Weights for Non-Uniform Selection
Currently AP_RANDOM picks uniformly. Need weighted selection for curriculum.

```c
// Needed in AutopilotState:
float mode_weights[AP_COUNT];
```

**Tasks:**
- [ ] Add `mode_weights` array to AutopilotState
- [ ] Implement weighted random selection in `autopilot_randomize()`
- [ ] Add Python API: `set_autopilot(mode_weights={...})`
- [ ] Default weights = uniform

### Per-Episode Parameter Variance
Bank angle and climb rate are fixed at `set_autopilot()` time.

**Need:**
- [ ] `bank_deg_min`, `bank_deg_max` fields - randomize within range each reset
- [ ] `climb_rate_min`, `climb_rate_max` fields
- [ ] `throttle_min`, `throttle_max` fields

### Difficulty Abstraction
Single 0.0-1.0 difficulty scale that controls multiple parameters.

```python
# Desired API:
env.set_difficulty(0.3)  # Maps to mode weights, bank angles, etc.
```

**Tasks:**
- [ ] Design difficulty mapping (what parameters at what difficulty)
- [ ] Implement `set_difficulty()` in Python
- [ ] Document difficulty levels

### Per-Environment Difficulty
In vectorized envs, all opponents currently share settings.

**Need:**
- [ ] Allow different autopilot settings per env index
- [ ] Or difficulty gradient across env indices

---

## Test Integration Gaps

### Player Autopilot for test_flight.py
Currently autopilot only controls opponent. test_flight.py tests player with Python PID.

**Tasks:**
- [ ] Add `AutopilotState player_ap` to Dogfight struct
- [ ] Add `player_autopilot_enabled` flag
- [ ] Add Python API `set_player_autopilot()`
- [ ] Migrate test_flight.py PID tests to use C autopilot

### Query Autopilot State
No way to verify autopilot mode from Python.

**Tasks:**
- [ ] Add `get_autopilot_mode()` C binding
- [ ] Return current mode, bank, climb_rate, etc.
- [ ] Add to Python wrapper

---

## Missing Maneuvers

### Basic Extensions
- [ ] `AP_WEAVE` - S-turns with configurable period
- [ ] `AP_CLIMBING_TURN` - Combined climb + bank
- [ ] `AP_DESCENDING_TURN` - Combined descent + bank

### Evasive Maneuvers (Hard difficulty)
- [ ] `AP_JINK` - Random direction changes at intervals
- [ ] `AP_BREAK` - Hard turn away from threat
- [ ] `AP_BARREL_ROLL` - Defensive roll

### Pursuit Behaviors
- [ ] `AP_PURSUIT` - Turn toward player
- [ ] `AP_LEAD_PURSUIT` - Aim ahead of player
- [ ] `AP_LAG_PURSUIT` - Trail behind player

### Opponent Combat
- [ ] Enable opponent firing (`actions[4]` currently hardcoded to -1)
- [ ] Accuracy scaling based on difficulty
- [ ] Reaction time/delay modeling

---

## Code Quality

### Explicit Random Mode List
Current implicit range assumption is fragile:
```c
int mode = 1 + (rand() % (AP_COUNT - 2));  // Assumes modes 1..5 are valid
```

**Fix:**
- [ ] Replace with explicit array of randomizable modes
```c
static const AutopilotMode RANDOM_MODES[] = {AP_LEVEL, AP_TURN_LEFT, ...};
```

### PID Derivative Smoothing
First step after reset may have derivative spike.

**Fix:**
- [ ] Initialize `prev_vz` to current `vz` on first step
- [ ] Or use filtered derivative

---

## Priority Order

1. **High:** Vectorized set_autopilot (blocking for multi-env curriculum)
2. **High:** Mode weights (core curriculum feature)
3. **Medium:** Per-episode parameter variance
4. **Medium:** Player autopilot for tests
5. **Low:** Additional maneuvers
6. **Low:** Opponent combat

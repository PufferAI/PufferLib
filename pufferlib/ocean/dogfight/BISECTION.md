# Training Regression Bisection

## Problem

Agent used to train to ~1.0 kills easily. After adding features (curriculum, stages), even the simplest scenario (TAIL_CHASE) only achieves ~0.5 kills.

**Goal**: Find the commit where training regressed.

---

## Instructions

### Git rules:

- **OK**: `git stash`, `git checkout`, `git diff`, `git log`
- **NOT OK**: `git add`, `git commit`, `git push` - DO NOT modify git history

### For each commit you test:

```bash
# 1. Stash any local changes if needed
git stash

# 2. Checkout the commit
git checkout HASH

# 3. Build
python setup.py build_ext --inplace --force

# 4. Run training 3 times, each to a separate log
python -m pufferlib.pufferl train puffer_dogfight 2>&1 | tee pufferlib/ocean/dogfight/baselines/bisect_HASH_r1.log
python -m pufferlib.pufferl train puffer_dogfight 2>&1 | tee pufferlib/ocean/dogfight/baselines/bisect_HASH_r2.log
python -m pufferlib.pufferl train puffer_dogfight 2>&1 | tee pufferlib/ocean/dogfight/baselines/bisect_HASH_r3.log

# 5. Return to working branch and restore stash
git checkout dogfight
git stash pop  # if you stashed earlier
```

### How to fill in the table:

1. **Kills**: Average the final `kills` metric from all 3 runs
2. **Return**: Average the final `episode_return` from all 3 runs
3. **Notes**:
   - If one run is a major outlier (e.g., 0.3, 0.9, 0.35), note it: "outlier: 0.9"
   - Note anything unusual: "no curriculum in this version", "different reward scale", etc.

### Choosing which commit to test next:

**DO NOT blindly bisect.** Use the commit messages to make informed choices.

1. Start with **a4172661** (baseline) to confirm training once worked
2. Use binary search as a GUIDE, but prioritize commits with meaningful names:
   - "Fix Reward and Score" - likely affects kills metric
   - "Simplify Penalties and Rewards" - definitely relevant
   - "Rewards Fixed - Sweepable" - relevant
   - "Debug Prints" - probably NOT relevant, skip unless adjacent to regression
3. If bisect lands you on "Debug Prints" but "Fix Reward and Score" is one commit away, test the meaningful one instead
4. The goal is to UNDERSTAND what broke, not just find a hash

**Think like a debugger**: What change COULD have broken training? Test those commits first.

---

## Results

Kills = 3-run average. Return = episode return average. Perf = older metric (same as kills in later commits).

| Hash | Message | Kills | Perf | Return | Notes |
|------|---------|-------|------|--------|-------|
| 4e640ee0 | Trying to Fix Curriculum - Agent Trains Poorly | ~0.5 | | | Current state, KNOWN BAD |
| f6c821d7 | Still Trying to Fix Blowups | | | | |
| 2c3073f5 | Debug Prints | | | | |
| b68d1b22 | Simplify Penalties and Rewards | | | | 6 obs schemes, default=0 |
| b68_obs0 | ^ scheme 0 (size 12) | 0.35 | 0.35 | -9 | BAD, eps_per_stage=120 |
| b68_obs1 | ^ scheme 1 (size 17) | | | | |
| b68_obs2 | ^ scheme 2 (size 10) | | | | |
| b68_obs3 | ^ scheme 3 (size 10) | | | | |
| b68_obs4 | ^ scheme 4 (size 13) | | | | |
| b68_obs5 | ^ scheme 5 (size 15) | | | | |
| 9dca5c67 | Reduce Prints | | | | |
| ab222bfc | Increase Batch Size for Speed | | | | eps_per_stage=15000 |
| ab2_obs5 | ^ scheme 5 (size 15) | 1.0 | 1.0 | -0.8 | **GOOD** last good commit |
| 7fd88f1c | Next Sweep Improvements - Likes to Aileron Roll too Much | | | | 6 obs, eps_per_stage=60 |
| 7fd_obs5 | ^ scheme 5 (size 15) | 0.02 | 0.02 | -82 | **TERRIBLE** eps_per_stage=60 |
| 7fd_obs0 | ^ scheme 0, eps=100k | 0.62 | 0.62 | -23 | All 3 runs consistent |
| 7fd_obs1 | ^ scheme 1, eps=100k | 0.71 | 0.71 | -18 | outlier r3=0.84 |
| 7fd_obs2 | ^ scheme 2, eps=100k | 0.81 | 0.81 | -26 | outlier r2=0.99! |
| 7fd_obs3 | ^ scheme 3, eps=100k | 0.55 | 0.55 | -45 | WORST, outlier r3=0.30 |
| 7fd_obs4 | ^ scheme 4, eps=100k | 0.62 | 0.62 | -35 | All 3 runs consistent |
| 7fd_obs5 | ^ scheme 5, eps=100k | 0.78 | 0.78 | -14 | outlier r3=0.96 |
| 30fa9fed | Fix Obs 5 Schema and Adjust Penalties | 0.94 | 0.94 | -34 | GOOD, outlier r3=0.81 |
| 652ab7a6 | Fix Elevator Problems | | | | |
| fe7e26a2 | Roll Penalty - Elevator Might Be Inversed | | | | |
| bc728368 | New Obs Schemas - New Sweep Prep | | | | 6 obs schemes (0-5) |
| bc7_obs0 | ^ scheme 0 (size 12) | | | | |
| bc7_obs1 | ^ scheme 1 (size 17) | | | | |
| bc7_obs2 | ^ scheme 2 (size 10) | | | | |
| bc7_obs3 | ^ scheme 3 (size 10) | | | | |
| bc7_obs4 | ^ scheme 4 (size 13) | | | | |
| bc7_obs5 | ^ scheme 5 (size 15) | 1.0 | 1.0 | 0.1 | **GOOD** ini default, very short eps |
| 2606e20e | Apply Sweep df1 84 u5i33hej | | | | 6 obs schemes (0-5) |
| 260_obs0 | ^ scheme 0 (size 19) | | | | |
| 260_obs1 | ^ scheme 1 (size 21) | | | | |
| 260_obs2 | ^ scheme 2 (size 12) | | | | |
| 260_obs3 | ^ scheme 3 (size 17) | | | | |
| 260_obs4 | ^ scheme 4 (size 10) | | | | |
| 260_obs5 | ^ scheme 5 (size 43) | | | | |
| 17f18c19 | Fix Reward and Score | | | | kills tracking added here |
| 3cc5b588 | More Sweep Prep | | | | |
| a31d1dc7 | Fix Terminals and Loggin | | | | |
| 26709b93 | Preparing for Sweeps | | | | |
| 04dd0167 | Rewards Fixed - Sweepable | | | | 6 obs schemes, default=0 |
| 04d_obs0 | ^ scheme 0 (size 19) | 3.4 | 0.93 | 41 | **GOOD BASELINE** |
| 04d_obs1 | ^ scheme 1 (size 21) | | | | |
| 04d_obs2 | ^ scheme 2 (size 12) | | | | |
| 04d_obs3 | ^ scheme 3 (size 17) | | | | |
| 04d_obs4 | ^ scheme 4 (size 10) | | | | |
| 04d_obs5 | ^ scheme 5 (size 43) | | | | |
| 63a7aaed | Observation Schemas Swept | | | | |
| 0a1c2e6d | Weighted Random Actions | | | | |
| 80bcf31e | Vectorized Autopilot | | | | |
| 85980679 | Autopilot Seperate File | | | | |
| 374871df | Small Perf - Move cosf Out of Loop | | | | |
| 1131e836 | Simple Optimizations | | | | |
| 1c30c546 | Coordinated Turn Tests | | | | |
| 3582d2d4 | Physics in Own File - Test Flights | | | | |
| 95eb2efd | Moved Physics to File | | | | |
| b29bf5ac | Renamed md Files | | | | |
| 0116b97c | Physics model: incidence, comments, test suite | | | | |
| 332a9ae0 | Good Claude - Wireframe Planes | | | | |
| daaf9024 | Rendered with spheres or something | | | | |
| 49af2d49 | Reward Changes | | | | |
| a4172661 | Trains and Evals | 0 | N/A | -43 | no kills tracking yet |

---

## Commit Summaries

Run `git diff HASH~1 HASH` to see what changed. Summarize each commit as you test it. Keep in chronological order (newest first).

### 4e640ee0
(summarize after reviewing diff)

### f6c821d7
(summarize after reviewing diff)

### 2c3073f5
(summarize after reviewing diff)

### b68d1b22
(summarize after reviewing diff)

### 9dca5c67
(summarize after reviewing diff)

### 7fd88f1c
(summarize after reviewing diff)

### 30fa9fed
(summarize after reviewing diff)

### 652ab7a6
(summarize after reviewing diff)

### fe7e26a2
(summarize after reviewing diff)

### bc728368
(summarize after reviewing diff)

### 2606e20e
(summarize after reviewing diff)

### 17f18c19
(summarize after reviewing diff)

### 3cc5b588
(summarize after reviewing diff)

### a31d1dc7
(summarize after reviewing diff)

### 26709b93
(summarize after reviewing diff)

### 04dd0167
Rewards Fixed - Sweepable. Proper kills/perf tracking. perf=0.93 (93% episodes get ≥1 kill), kills=3.4 avg, return=41. **GOOD BASELINE - training works here.**

### 63a7aaed
(summarize after reviewing diff)

### 0a1c2e6d
(summarize after reviewing diff)

### 80bcf31e
(summarize after reviewing diff)

### 85980679
(summarize after reviewing diff)

### 374871df
(summarize after reviewing diff)

### 1131e836
(summarize after reviewing diff)

### 1c30c546
(summarize after reviewing diff)

### 3582d2d4
(summarize after reviewing diff)

### 95eb2efd
(summarize after reviewing diff)

### b29bf5ac
(summarize after reviewing diff)

### 0116b97c
(summarize after reviewing diff)

### 332a9ae0
(summarize after reviewing diff)

### daaf9024
(summarize after reviewing diff)

### 49af2d49
(summarize after reviewing diff)

### a4172661
Initial dogfight commit. Creates entire environment from scratch. No kills/perf tracking - only logs episode_return, episode_length, n. Returns: -43.9, -109.6, -57.4 (avg -70). Cannot use as baseline for kills metric.

---

## Findings

### Possible Cause: `episodes_per_stage` Config Change

| Commit | episodes_per_stage | perf |
|--------|-------------------|------|
| 30fa9fed (GOOD) | 15000 | 0.94 |
| 7fd88f1c (BAD) | 60 | 0.02 |
| 7fd_edit (PARTIAL) | 100000 | 0.76 |
| b68d1b22 (BAD) | 120 | 0.35 |

**Finding**: `episodes_per_stage` is a MAJOR factor (0.02 → 0.76 when increased), but code changes in 7fd88f1c also matter (0.76 vs 1.0 in last good commit).

### First Bad Commit: 7fd88f1c

**Config changes** (ab222bfc → 7fd88f1c):
- `episodes_per_stage`: 15000 → 60 (250x fewer!)
- `penalty_roll`: 0.0015 → 0.003 (2x)
- NEW: `penalty_aileron = 0.1`
- NEW: `penalty_bias = 0.01`
- NEW: `reward_approach = 0.005`
- NEW: `reward_level = 0.02`
- Swapped reward_firing_solution ↔ reward_tracking

**Code changes**: 454 insertions, 101 deletions in dogfight.h, autopilot.h, etc.

### All Obs Schemes at 7fd88f1c (eps_per_stage=100000)

| Scheme | Size | r1 | r2 | r3 | Avg Perf | Notes |
|--------|------|-----|-----|-----|----------|-------|
| 0 | 12 | 0.62 | 0.63 | 0.62 | 0.62 | consistent |
| 1 | 17 | 0.64 | 0.63 | 0.84 | 0.71 | high variance |
| 2 | 10 | 0.80 | 0.99 | 0.64 | 0.81 | r2 nearly perfect! |
| 3 | 10 | 0.70 | 0.65 | 0.30 | 0.55 | WORST scheme |
| 4 | 13 | 0.66 | 0.58 | 0.62 | 0.62 | consistent |
| 5 | 15 | 0.70 | 0.67 | 0.96 | 0.78 | high variance |

**Key Finding**: ALL obs schemes underperform vs last good commit (1.0). This is NOT just one bad scheme - it's a deeper code problem in 7fd88f1c.

Observations:
- High variance across runs for all schemes
- Scheme 2 had one run hit 0.99, scheme 5 hit 0.96 - so 1.0 is achievable but inconsistent
- Scheme 3 is worst (0.55 avg)
- Problem affects all observation schemes equally

### To Investigate

1. ~~Test 7fd88f1c with `episodes_per_stage=100000` to isolate config vs code~~ **DONE**: perf varies by scheme, 0.55-0.81
2. ~~Test all obs schemes at 7fd88f1c~~ **DONE**: ALL schemes underperform, not one bad scheme
3. Diff dogfight.h code between ab222bfc and 7fd88f1c to find the breaking change
4. Focus on: new penalties (aileron=0.1, bias=0.01), new rewards (approach, level), or code logic bugs

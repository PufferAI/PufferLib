# Bat next sweep research notes

Purpose: concise decision notes for future agents before changing Bat physics.

Status: planning note only.

## Current baseline to preserve first

Before adding new physics, commit and sweep the current Bat state that already includes:

- Timer observation normalized `0..1`.
- Timeout terminal value as `-1.0`.
- Chirp usage normalized `0..1` with death/termination if exceeding the allowed budget.
- Reward/log cleanup.
- Recording code moved out of `bat.h`.
- Audio helpers moved out of `bat.h`.

Reason: ear directivity and wing micro-Doppler are real behavior changes. They should not be mixed into the baseline sweep used to judge timer/log/audio cleanup.

## Candidate A: static ear directivity

Add a cheap polar response for each ear.

Expected benefit:

- Stronger left/right spatial cue.
- Facing direction matters more.
- Rear echoes become weaker.

Main risk:

- Exploration may become harder if rear/side gain is too low.
- Observation scale may drift if gains are not normalized.

Recommended first form:

```c
float front = clamp(dot(source_dir, forward), 0, 1);
float left = clamp(dot(source_dir, left_ear_dir), 0, 1);
float right = clamp(dot(source_dir, right_ear_dir), 0, 1);
left_gain = rear_floor + front_gain*front*front + side_gain*left*left;
right_gain = rear_floor + front_gain*front*front + side_gain*right*right;
```

Suggested constants:

```c
#define BAT_EAR_REAR_GAIN 0.15f
#define BAT_EAR_FRONT_GAIN 0.55f
#define BAT_EAR_SIDE_GAIN 0.45f
#define BAT_EAR_GAIN_NORM (1.0f / (BAT_EAR_REAR_GAIN + BAT_EAR_FRONT_GAIN + BAT_EAR_SIDE_GAIN))
```

Research doc:

- `ocean/bat/BAT_EAR_DIRECTIVITY_RESEARCH.md`

## Candidate B: bug wing echo sideband

Add prey-specific wing flutter echo structure.

Expected benefit:

- Bug echoes become distinguishable from obstacle echoes.
- Adds a moving-prey cue without raw audio simulation.

Main risk:

- More echo events can saturate event capacity or observation bins.
- If energy is too low it adds no learnable signal; if too high it changes task scale.

Recommended first form:

- Keep body echo unchanged.
- Add one extra bug-only wing echo.
- Use triangle phase, no `sinf`.

Suggested constants:

```c
#define BAT_BUG_WING_ECHO_GAIN 0.20f
#define BAT_BUG_WING_FREQ_OFFSET 0.06f
#define BAT_BUG_WING_PHASE_STEP 0.11f
```

Research doc:

- `ocean/bat/BAT_WING_ECHO_RESEARCH.md`

## Sweep ordering

1. Baseline current Bat.
2. Ear directivity only.
3. Wing sideband only.
4. Combined directivity + wing sideband only if individual variants are viable.

Do not add both new physics changes before an ablation. It will make results ambiguous.

## Success metrics to compare

Use the same training/eval flow as the recent Bat work:

- Build passes.
- Bat C tests pass.
- Training completes on current ini without timestep override.
- Compare `perf`, `base_perf`, `SPS`, `timeout`, and qualitative eval behavior.
- Level 5 eval should still look reasonable.

Known recent baseline from audio-helper move:

- `perf` around `0.375`.
- `base_perf` around `0.942`.
- `SPS` around `1.5M`.
- `timeout` around `0.001`.

Do not overinterpret one training run. Use it as a regression/sanity check, then sweep.

## Source anchors

Ear directivity:

- https://doi.org/10.1038/nature11664
- https://doi.org/10.3389/fphys.2013.00089
- https://doi.org/10.1073/pnas.1006630107
- https://doi.org/10.1121/1.418271
- https://doi.org/10.1121/1.3488304
- https://doi.org/10.1242/jeb.210252
- https://doi.org/10.3389/fphys.2013.00191

Wing echo / micro-Doppler:

- https://doi.org/10.1007/BF00612592
- https://doi.org/10.1098/rspb.2003.2487
- https://doi.org/10.1098/rspb.2012.2830
- https://doi.org/10.1037/bne0000315

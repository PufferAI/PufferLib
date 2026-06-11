# Bat ear directivity research notes

Purpose: preserve research and implementation guidance for a possible low-cost directional hearing model in `ocean/bat/`.

Status: research/design note only. No behavior change is implied by this document.

## Short answer

Yes, the model should not treat each ear as an omnidirectional scalar receiver. Bat echolocation uses directional emission, directional reception, and binaural differences. The useful terms are:

- `HRTF`: head-related transfer function, the direction-dependent filtering from a sound source to each ear.
- `HRIR`: time-domain head-related impulse response.
- `ILD`: interaural level difference, the loudness/intensity difference between left and right ears.
- `ITD`: interaural time difference, the arrival-time difference between ears.
- `Pinna directivity`: direction-dependent gain/filtering caused by the external ear shape.
- `Beam pattern` or `polar response`: gain as a function of angle.
- `Acoustic field of view`: the spatial volume that is ensonified or heard well enough for detection.

For Bat env purposes, the best first implementation is a cheap per-ear gain curve in `bat_schedule_echo`, based on relative angle to target/obstacle/echo source. It should use dot products and multiplications, not `atan2f`, not tables, and not per-frequency filters.

## What the literature says

### Bats have directional sonar emission and dynamic beam width

Jakobsen, Ratcliffe, and Surlykke found that multiple vespertilionid species converge on similar sonar fields of view. The Nature abstract reports a directivity index around `11 +/- 1 dB`, half-amplitude angle about `37 degrees`, and on-axis source level around `108 +/- 4 dB SPL re 20 uPa rms at 10 cm` under their tested condition.

Source:

- Jakobsen, L.; Ratcliffe, J. M.; Surlykke, A. `Convergent acoustic field of view in echolocating bats`. Nature 493, 93-96, 2013. DOI: https://doi.org/10.1038/nature11664
- Nature page: https://www.nature.com/articles/nature11664

Implementation relevance:

- The environment already has directional structure via left/right echo channels, but the hearing side can plausibly become more directional.
- A simple polar response is justified: forward is strong, rear is weak, lateral differs by ear.
- A 2D game does not need full 3D HRTF. The important behavioral signal is `left/right relative energy`, not spectral notches.

### Directionality and intensity jointly define what the bat can detect

Jakobsen, Brinklov, and Surlykke reviewed bat echolocation intensity and directionality. Key implementation-relevant points:

- Bat calls are directional; more energy is focused forward than to the sides.
- An object detectable directly in front at a given range may not be detectable at the same range off-axis.
- Directionality reduces clutter because less energy is emitted to the sides/back.
- Beam shape acts as a spatial filter before echoes return.
- Bats dynamically control intensity, duration, frequency, and directionality.
- Nose emitters can have beam shape affected by nostril separation and noseleaf geometry.
- Mouth emitters can affect directionality via gape size.

Source:

- Jakobsen, L.; Brinklov, S.; Surlykke, A. `Intensity and directionality of bat echolocation signals`. Frontiers in Physiology 4:89, 2013. DOI: https://doi.org/10.3389/fphys.2013.00089
- Open full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC3635024/

Implementation relevance:

- If we add hearing directivity, it should be part of the echo energy calculation, not an observation post-process.
- It should affect both bug and obstacle echoes consistently.
- We should keep it cheap enough to run per echo/event/source.

### Bats can broaden beams in terminal pursuit

Jakobsen and Surlykke showed that `Myotis daubentonii` and `Eptesicus serotinus` broaden their biosonar beam during prey pursuit. Crossref metadata includes the useful quantitative anchor: `M. daubentonii` increased half-amplitude angle from about `40 degrees` to about `90 degrees` horizontally and from about `45 degrees` to more than `90 degrees` vertically, mostly by dropping call frequency by about one octave from `55 kHz` to `27.5 kHz`.

Source:

- Jakobsen, L.; Surlykke, A. `Vespertilionid bats control the width of their biosonar sound beam dynamically during prey pursuit`. PNAS 107(31), 13930-13935, 2010. DOI: https://doi.org/10.1073/pnas.1006630107
- PNAS page: https://www.pnas.org/doi/10.1073/pnas.1006630107

Implementation relevance:

- This is more about emission than reception, but it argues against a single static omnidirectional model.
- We do not need to implement dynamic beam width yet. It would be a meaningful physics change and should be isolated in a sweep.
- If implemented later, chirp duration/frequency choices could alter beam width. That would make action consequences richer, but it is not the minimum ear-directivity change.

### Reception-side filtering matters too

Wotton, Jenison, and Hartley modeled/combined emission and external-ear reception in the big brown bat. Their abstract says localization cues become clearer when emission spectra and external-ear spectra are convolved; spectral peaks sharpen and peak/notch contrast increases. It also notes cues restricted to a cone of about `+/-30 degrees`.

Source:

- Wotton, J. M.; Jenison, R. L.; Hartley, D. J. `The combination of echolocation emission and ear reception enhances directional spectral cues of the big brown bat, Eptesicus fuscus`. JASA 101(3), 1723-1733, 1997. DOI: https://doi.org/10.1121/1.418271
- AIP/JASA page: https://pubs.aip.org/asa/jasa/article/101/3/1723/559358/The-combination-of-echolocation-emission-and-ear

Implementation relevance:

- Full spectral filtering is overkill for current Bat. The obs are low-dimensional echo features, not raw waveforms.
- A cheap gain curve per ear captures the important part for policy learning: direction-dependent intensity.
- Avoid adding FFTs, filters, or per-frequency HRTF tables unless the environment changes to raw audio observations.

### Noseleaf and pinnae can cooperate dynamically

Kuc proposed a model where noseleaf and pinnae cooperate through direct and delayed acoustic paths. The abstract says the delayed pinna component can increase on-axis emission strength, narrow beam width, and sculpt frequency-dependent beam patterns.

Source:

- Kuc, R. `Morphology suggests noseleaf and pinnae cooperate to enhance bat echolocation`. JASA 128(5), 3190-3199, 2010. DOI: https://doi.org/10.1121/1.3488304
- AIP/JASA page: https://pubs.aip.org/asa/jasa/article/128/5/3190/917806/Morphology-suggests-noseleaf-and-pinnae-cooperate

Zhang et al. studied great roundleaf bats and found coordinated noseleaf and pinna movements during echolocation.

Source:

- Zhang, S.; et al. `Dynamic relationship between noseleaf and pinnae in echolocating hipposiderid bats`. Journal of Experimental Biology, 2019. DOI: https://doi.org/10.1242/jeb.210252
- JEB page: https://journals.biologists.com/jeb/article/222/20/jeb210252/224403/Dynamic-relationship-between-noseleaf-and-pinnae-in

Vanderelst et al. found that the noseleaf of `Rhinolophus formosae` focuses the FM component of calls.

Source:

- Vanderelst, D.; Lee, Y.-F.; Geipel, I.; Kalko, E. K. V.; Kuo, Y.-M.; Peremans, H. `The noseleaf of Rhinolophus formosae focuses the Frequency Modulated (FM) component of the calls`. Frontiers in Physiology 4:191, 2013. DOI: https://doi.org/10.3389/fphys.2013.00191
- Frontiers page: https://www.frontiersin.org/articles/10.3389/fphys.2013.00191/full

Implementation relevance:

- These papers support a directional receive model, but they also warn that exact geometry is species-specific and complex.
- For Bat env, do not model moving pinnae/noseleaf first. That would create extra state and new parameters without proving learning benefit.
- Keep the first model static and symmetric, then sweep it.

## Current likely Bat code location

The directivity should probably be applied in or near the echo scheduling/energy path, around the existing left/right echo gain logic. Earlier review found a mild directional term like this in `bat_schedule_echo`:

```c
float left_gain = 0.75f + 0.25f * something;
float right_gain = 0.75f + 0.25f * something;
```

That is a weak directional receiver. A stronger, biologically motivated model would replace that with a front-and-side polar response.

Do not add this in render/audio code. The training observation echo energy must change, not only playback.

## Recommended cheap implementation

Use only normalized source direction and bat forward/side vectors. No angle, no trig.

Definitions:

- `ux, uy`: unit vector from bat to echo source.
- `fx, fy`: bat forward unit vector.
- `lx, ly`: bat left-ear preferred lateral unit vector, usually left of forward.
- `rx, ry`: bat right-ear preferred lateral unit vector, usually right of forward.
- `front`: nonnegative forward alignment.
- `left_side`: nonnegative left-ear side alignment.
- `right_side`: nonnegative right-ear side alignment.
- `rear_floor`: minimum rear sensitivity so rear echoes are not impossible.

Sketch:

```c
float front = bat_clampf(ux*fx + uy*fy, 0.0f, 1.0f);
float left_side = bat_clampf(ux*lx + uy*ly, 0.0f, 1.0f);
float right_side = bat_clampf(ux*rx + uy*ry, 0.0f, 1.0f);

float front2 = front * front;
float left2 = left_side * left_side;
float right2 = right_side * right_side;

float left_gain = rear_floor + front_gain*front2 + side_gain*left2;
float right_gain = rear_floor + front_gain*front2 + side_gain*right2;
```

Potential initial constants:

```c
#define BAT_EAR_REAR_GAIN 0.15f
#define BAT_EAR_FRONT_GAIN 0.55f
#define BAT_EAR_SIDE_GAIN 0.45f
```

Normalize if needed:

```c
#define BAT_EAR_GAIN_NORM (1.0f / (BAT_EAR_REAR_GAIN + BAT_EAR_FRONT_GAIN + BAT_EAR_SIDE_GAIN))
left_gain *= BAT_EAR_GAIN_NORM;
right_gain *= BAT_EAR_GAIN_NORM;
```

This keeps max gain near `1.0`, gives front-left stronger left signal, front-right stronger right signal, and keeps behind weak but nonzero.

## Variant: ear axes angled forward

Pure side vectors can make lateral echoes too strong compared with forward echoes. A better biological-ish 2D approximation is ears pointed outward but forward-biased.

Given forward `f` and left normal `n`:

```c
float ear_forward = 0.75f;
float ear_side = 0.66f;
float left_ear_x = ear_forward*fx + ear_side*nx;
float left_ear_y = ear_forward*fy + ear_side*ny;
float right_ear_x = ear_forward*fx - ear_side*nx;
float right_ear_y = ear_forward*fy - ear_side*ny;
```

If `ear_forward^2 + ear_side^2` is approximately `1`, no normalization needed. `0.75/0.66` is close enough for a cheap model.

Then:

```c
float left_lobe = bat_clampf(ux*left_ear_x + uy*left_ear_y, 0.0f, 1.0f);
float right_lobe = bat_clampf(ux*right_ear_x + uy*right_ear_y, 0.0f, 1.0f);
left_gain = rear_floor + main_gain * left_lobe * left_lobe;
right_gain = rear_floor + main_gain * right_lobe * right_lobe;
```

This is even simpler and likely enough.

## Performance considerations

Good:

- Dot products.
- Multiplication for squaring.
- `bat_clampf` or inline clamp.
- Constants as `#define`.

Avoid:

- `atan2f` per echo.
- `cosf`/`sinf` per echo if forward/side vectors already exist.
- Per-frequency HRTF tables.
- New heap allocations.
- Raw audio convolution.

The model should cost only a few multiplies per scheduled echo.

## Expected behavior change

Likely effects:

- Better left/right spatial signal when target is off-center.
- Rear obstacles/bugs become less audible.
- Policy may learn to turn/scan because facing matters more.
- Existing trained checkpoint performance may change because observations change.

Potential risk:

- If rear/side gain is too low, exploration may get harder.
- If gains are not normalized, reward/observation scale may drift.
- If directivity is applied on top of an already strong directional term, left/right energy may saturate.

## Sweep recommendation

Do not combine this with wing micro-Doppler in the same first sweep. Use a clean ablation:

- Baseline: current Bat after timer/log/audio cleanup.
- Variant A: static ear directivity only.
- Variant B: wing sidebands only.
- Variant C: both, only if A and B individually help or at least do not hurt.

Suggested parameters for first sweep:

```ini
[env]
ear_directivity_enabled = 1
ear_rear_gain = 0.15
ear_front_gain = 0.55
ear_side_gain = 0.45
```

If avoiding config bloat, hard-code the first constants behind defines and sweep by branch/commit instead.

## Implementation checklist

- Apply directivity before writing echo energy into observations.
- Apply to bug and obstacle echoes unless there is a specific reason not to.
- Keep left/right symmetry exact.
- Keep max gain normalized near current max so observation scale does not drift hard.
- Add one focused C test for left/right asymmetry if tests are desired.
- Run build/tests/train/eval before comparing performance.

## Source list

- https://doi.org/10.1038/nature11664
- https://www.nature.com/articles/nature11664
- https://doi.org/10.3389/fphys.2013.00089
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3635024/
- https://doi.org/10.1073/pnas.1006630107
- https://www.pnas.org/doi/10.1073/pnas.1006630107
- https://doi.org/10.1121/1.418271
- https://pubs.aip.org/asa/jasa/article/101/3/1723/559358/The-combination-of-echolocation-emission-and-ear
- https://doi.org/10.1121/1.3488304
- https://pubs.aip.org/asa/jasa/article/128/5/3190/917806/Morphology-suggests-noseleaf-and-pinnae-cooperate
- https://doi.org/10.1242/jeb.210252
- https://journals.biologists.com/jeb/article/222/20/jeb210252/224403/Dynamic-relationship-between-noseleaf-and-pinnae-in
- https://doi.org/10.3389/fphys.2013.00191
- https://www.frontiersin.org/articles/10.3389/fphys.2013.00191/full

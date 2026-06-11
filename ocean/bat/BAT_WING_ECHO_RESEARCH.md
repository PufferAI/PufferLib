# Bat insect-wing echo and micro-Doppler research notes

Purpose: preserve research and implementation guidance for possible low-cost insect wing flutter / micro-Doppler echoes in `ocean/bat/`.

Status: research/design note only. No behavior change is implied by this document.

## Short answer

Yes, insect prey should plausibly produce more than a single body echo. Flying insect wings can create echo fluctuations, amplitude modulation, and Doppler/micro-Doppler-like frequency structure. The simplest useful Bat env approximation is:

- Keep the existing normal body echo.
- For bug echoes only, add one or two weaker wing echoes near the body echo.
- Make wing echoes vary over time with a cheap phase oscillator.
- Keep obstacle echoes unchanged.

This should add a moving-prey signature without turning the environment into an expensive acoustic simulator.

## Useful terminology

- `Doppler shift`: frequency shift caused by relative motion between bat and target.
- `Micro-Doppler`: additional Doppler components from moving parts of a target, such as flapping wings, legs, rotors, or vibrating surfaces.
- `Flutter detection`: detecting oscillating target movements, especially insect wing motion, in echoes.
- `Amplitude modulation`: echo strength fluctuates as wing orientation and scattering cross-section change.
- `Spectral glints`: brief bright echo components from reflective target parts at favorable orientations.
- `Sidebands`: frequency components above and below a carrier/body frequency caused by modulation.

## What the literature says

### CF/CF-FM bats can use Doppler and flutter cues

The classic result is Schnitzler and Flieger on greater horseshoe bats detecting oscillating target movement. Crossref metadata confirms the paper:

- Schnitzler, H.-U.; Flieger, E. `Detection of oscillating target movements by echolocation in the Greater Horseshoe bat`. Journal of Comparative Physiology 153, 385-391, 1983. DOI: https://doi.org/10.1007/BF00612592

Secondary summaries and reviews describe the key idea: CF bats are especially suited to detecting target velocity and wing flutter as Doppler-shifted frequencies. Oscillating wings also create amplitude shifts that help distinguish flying prey from stationary targets.

Implementation relevance:

- Bat env currently uses chirps/echoes as compact observations. It does not need raw CF sonar.
- A cheap wing signature is still justified because it gives the policy a prey-specific temporal/frequency cue.
- Apply it only to `BAT_ECHO_BUG`, not walls/obstacles.

### Echolocation range and wingbeat timing are behaviorally linked

Holderied and von Helversen studied aerial-hawking bats and found a relationship between echolocation range and wingbeat period.

Source:

- Holderied, M. W.; von Helversen, O. `Echolocation range and wingbeat period match in aerial-hawking bats`. Proceedings of the Royal Society B 270, 2293-2299, 2003. DOI: https://doi.org/10.1098/rspb.2003.2487
- Royal Society page: https://royalsocietypublishing.org/doi/10.1098/rspb.2003.2487

Implementation relevance:

- Wingbeat dynamics are not just visual animation; they are related to sensing and prey pursuit timing.
- If Bat already has a tick-based model, wing phase can update once per tick using a fixed increment.
- No per-chirp expensive computation is needed.

### Bats can classify prey shape/material from echo structure

Geipel, Jung, and Kalko showed that `Micronycteris microtis` can detect, classify, and localize silent, motionless prey in clutter using echolocation alone. Their abstract says bats used short, multi-harmonic broadband calls and appeared to perceive a detailed acoustic image based on shape, surface structure, and material.

Source:

- Geipel, I.; Jung, K.; Kalko, E. K. V. `Perception of silent and motionless prey on vegetation by echolocation in the gleaning bat Micronycteris microtis`. Proceedings of the Royal Society B 280:20122830, 2013. DOI: https://doi.org/10.1098/rspb.2012.2830
- Royal Society page: https://royalsocietypublishing.org/doi/10.1098/rspb.2012.2830

Implementation relevance:

- Even without active wing motion, bugs are not acoustically equivalent to points.
- If adding wing sidebands, keep them as prey-specific echo complexity, not as general noise.
- This supports making bug echoes richer than obstacle echoes.

### Micro-spectral ripple research supports compact target-specific echo features

Shriram and Simmons studied bats perceiving natural-size targets as a unitary class using micro-spectral ripples in echoes.

Source:

- Shriram, U.; Simmons, J. A. `Echolocating bats perceive natural-size targets as a unitary class using micro-spectral ripples in echoes`. Behavioral Neuroscience 133(3), 297-304, 2019. DOI: https://doi.org/10.1037/bne0000315
- APA page: https://doi.apa.org/doi/10.1037/bne0000315

Implementation relevance:

- Richer echo spectra can matter, but Bat should not model detailed spectra first.
- A few deterministic sidebands are a cheap stand-in for target-specific microstructure.
- This is closer to a useful observation feature than raw acoustic realism.

## Recommended cheap model

### Core idea

When scheduling a bug echo, add:

- `body echo`: existing echo path, unchanged except for any directivity/range logic already present.
- `wing upper echo`: smaller energy, slightly higher normalized frequency.
- `wing lower echo`: smaller energy, slightly lower normalized frequency.

The upper/lower echoes represent wing motion toward/away from the bat and modulation around the body return.

Sketch:

```c
float wing_phase = env->bug_wing_phase;
float wing = 0.5f + 0.5f * sinf(wing_phase);
float wing_offset = BAT_BUG_WING_FREQ_OFFSET * (0.5f + 0.5f * wing);
float wing_energy = body_energy * BAT_BUG_WING_ECHO_GAIN;

bat_add_echo_event(env, echo_time, body_freq, body_energy, left_gain, right_gain, BAT_ECHO_BUG);
bat_add_echo_event(env, echo_time, body_freq + wing_offset, wing_energy, left_gain, right_gain, BAT_ECHO_BUG);
bat_add_echo_event(env, echo_time, body_freq - wing_offset, wing_energy, left_gain, right_gain, BAT_ECHO_BUG);
```

If avoiding `sinf`, use a triangle oscillator:

```c
float phase = env->bug_wing_phase;
float tri = phase < 0.5f ? phase * 2.0f : (1.0f - phase) * 2.0f;
float wing_offset = BAT_BUG_WING_FREQ_OFFSET * tri;
```

Then update phase once per env step:

```c
env->bug_wing_phase += BAT_BUG_WING_PHASE_STEP;
if (env->bug_wing_phase >= 1.0f) env->bug_wing_phase -= 1.0f;
```

Use a constant phase step instead of division per tick. If it needs to depend on tick rate, define the reciprocal as a constant.

### Initial constants

The actual values should be tuned by sweep, but a reasonable first pass:

```c
#define BAT_BUG_WING_ECHO_GAIN 0.20f
#define BAT_BUG_WING_FREQ_OFFSET 0.06f
#define BAT_BUG_WING_PHASE_STEP 0.11f
```

Interpretation:

- `BAT_BUG_WING_ECHO_GAIN`: each sideband gets 20% of body energy.
- `BAT_BUG_WING_FREQ_OFFSET`: normalized frequency offset, not real kHz.
- `BAT_BUG_WING_PHASE_STEP`: wing animation/sensing phase increment per env tick.

If the two sidebands make total bug energy too high, compensate:

```c
float body_energy = base_energy * 0.75f;
float wing_energy = base_energy * 0.125f;
```

This preserves total energy while adding structure. If the goal is to make bugs easier to identify, do not preserve total energy exactly; but then treat it as a real behavior change.

## Cheaper one-sideband variant

If three echo events per bug chirp is too much, use one extra echo whose sign flips with wing phase:

```c
float tri = env->bug_wing_phase < 0.5f ? env->bug_wing_phase * 2.0f : (1.0f - env->bug_wing_phase) * 2.0f;
float sign = env->bug_wing_phase < 0.5f ? 1.0f : -1.0f;
float wing_freq = body_freq + sign * BAT_BUG_WING_FREQ_OFFSET * tri;
float wing_energy = body_energy * BAT_BUG_WING_ECHO_GAIN;

bat_add_echo_event(env, echo_time, body_freq, body_energy, left_gain, right_gain, BAT_ECHO_BUG);
bat_add_echo_event(env, echo_time, wing_freq, wing_energy, left_gain, right_gain, BAT_ECHO_BUG);
```

This is half the extra event count. It gives time-varying high/low pings, but not simultaneous symmetric sidebands.

## Even cheaper amplitude-only variant

If we want no extra echo events, modulate bug echo energy:

```c
float tri = env->bug_wing_phase < 0.5f ? env->bug_wing_phase * 2.0f : (1.0f - env->bug_wing_phase) * 2.0f;
float flutter_gain = 1.0f + BAT_BUG_WING_AMP_MOD * (tri - 0.5f);
body_energy *= flutter_gain;
```

Potential constant:

```c
#define BAT_BUG_WING_AMP_MOD 0.30f
```

This is cheapest but probably less useful because the policy may see it as noise unless it can integrate over time.

## Recommended first implementation choice

Use the two-sideband model only if echo event capacity is safely high and current observations can represent multiple arrivals without saturation.

Use the one-sideband model if event pressure is a concern.

Use amplitude-only only as a fallback.

My recommendation for first sweep:

- One extra wing echo per bug echo.
- Triangle oscillator, no `sinf`.
- `BAT_BUG_WING_ECHO_GAIN = 0.20f`.
- `BAT_BUG_WING_FREQ_OFFSET = 0.06f`.
- Preserve body echo unchanged for the first test so the new signal is additive and easy to ablate.

## Where it should live in Bat

Likely location:

- Bug echo scheduling path, near `bat_schedule_echo` or wherever `BAT_ECHO_BUG` events are created.

Rules:

- Do not add this to obstacle echoes.
- Do not add this to render-only or audio-only code.
- Add/advance wing phase in the core env tick/reset state if deterministic observations depend on it.
- If state serialization exists or is added later, include wing phase.
- If randomizing initial wing phase, seed it deterministically with env RNG.

## Performance considerations

Good:

- Phase update once per step.
- Triangle wave instead of `sinf`.
- Constants as `#define`.
- Add at most one extra event first.
- Clamp normalized frequency with existing clamp logic.

Avoid:

- Per-echo trigonometry if not needed.
- FFT or convolution.
- Large target meshes.
- Per-wing geometry.
- More than one or two additional events without checking event capacity and observation saturation.

## Expected behavior change

Likely effects:

- Bug echoes become more identifiable than obstacle echoes.
- The policy may learn that moving/oscillating echo structure indicates prey.
- Depending on reward and observation clipping, it may improve pursuit or just add noise.
- If event buffers saturate, it can silently hurt by dropping echoes.

Important risk:

- PufferLib reward clipping already caused signal issues earlier. Echo observation scaling can have a similar failure mode if new wing echoes saturate observation bins. Keep energy modest and inspect normalization before committing to a sweep.

## Interaction with ear directivity

Ear directivity and wing sidebands should be tested separately first.

Reason:

- Ear directivity changes spatial gain.
- Wing sidebands change prey identity/frequency/time structure.
- Combining them at once makes it hard to know what helped or broke.

Order recommendation:

1. Sweep current baseline after timer/log/audio cleanup.
2. Add static ear directivity only.
3. Add bug wing sideband only.
4. Combine only if both individual variants look viable.

## Implementation checklist

- Add `bug_wing_phase` to env state only if needed by deterministic core observations.
- Reset/init `bug_wing_phase` deterministically.
- Advance phase with multiplication/addition, not division.
- Add wing echo only for `BAT_ECHO_BUG`.
- Clamp wing frequency after offset.
- Ensure event capacity cannot drop important echoes.
- Keep observation normalization stable.
- Build/test/train/eval before comparing to baseline.

## Source list

- https://doi.org/10.1007/BF00612592
- https://doi.org/10.1098/rspb.2003.2487
- https://royalsocietypublishing.org/doi/10.1098/rspb.2003.2487
- https://doi.org/10.1098/rspb.2012.2830
- https://royalsocietypublishing.org/doi/10.1098/rspb.2012.2830
- https://doi.org/10.1037/bne0000315
- https://doi.apa.org/doi/10.1037/bne0000315
- https://en.wikipedia.org/wiki/Animal_echolocation
- https://en.wikipedia.org/wiki/Doppler_shift_compensation

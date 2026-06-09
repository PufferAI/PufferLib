# Bat Sonar Observation Notes

Status: design note for current and future Bat agents

Workspace: `/home/claude/pathfinder`

Related spec: `BAT_SPEC.md`

## Purpose

This note records the intended next observation and echo model for the Bat
environment. The current implementation was deliberately simplified to get a
trainable baseline. The next rung should make active echolocation real: the bat
should hear frequency energy only when echoes from its own chirps return.

## Retired Scaffold Implementation

The first Bat observation was a fast synthetic feature extractor, not a true
chirp-return audio model. It has been retired, but the notes are kept here so
future agents understand why the env moved away from it.

Current layout:

- `left_range_energy[16]`
- `left_doppler_energy[16]`
- `right_range_energy[16]`
- `right_doppler_energy[16]`
- `chirp_age_norm`
- `last_chirp_start_freq_norm`
- `last_chirp_end_freq_norm`
- `last_chirp_duration_norm`
- `forward_speed_norm`
- `turn_rate_norm`

Total size: `70`.

Each frame, the env recomputes current echo features from the current bat,
bug, wall, and obstacle positions. The bug is one strong moving reflector.
Walls and obstacle edges are sampled into static point reflectors. For each
reflector, the env computes approximate left-ear and right-ear path lengths,
attenuation, left/right gain, and a normalized Doppler value. It then deposits
energy into range-indexed observation slots.

This is useful for a first baseline, but it is too informative:

- The bat gets fresh echo-like information every frame, even if it did not
  chirp.
- Chirp start frequency, end frequency, and duration do not materially affect
  the acoustic observation.
- The Doppler channels are scalar range-indexed values, not FFT bins.
- Range is exposed as direct binned path length instead of being inferred from
  echo return timing.

## Current Target Model

The observation should be per-tick binaural frequency energy:

- `left_freq_bins[N]`
- `right_freq_bins[N]`
- chirp metadata
- cooldown/age metadata
- self-motion metadata

No explicit delay/range bins are needed in the observation. Distance should be
implicit in time. The policy should infer range from when frequency energy
returns after a chirp.

Current layout:

- `left_freq_bins[16]`
- `right_freq_bins[16]`
- `chirp_age_norm`
- `chirp_cooldown_norm`
- `last_chirp_start_freq_norm`
- `last_chirp_end_freq_norm`
- `last_chirp_duration_norm`
- `forward_speed_norm`
- `turn_rate_norm`

Total size: `39`.

If 16 bins is too coarse after implementation, use 24 bins per ear for a total
size of `55`.

## Event-Driven Echo Model

Do not synthesize raw audio and do not run an FFT per environment step. Use an
analytic event model that directly deposits echo energy into frequency bins at
the tick when the echo reaches each ear.

When a chirp is emitted:

1. Break the chirp into a small number of time slices.
2. For each slice, compute the emitted frequency from chirp start frequency,
   end frequency, and duration.
3. For each reflector, compute when that slice reaches the reflector.
4. Compute when the reflected sound reaches the left ear and right ear.
5. Compute returned amplitude, ear gain, and Doppler-shifted frequency.
6. Enqueue an echo event for each ear.

Each echo event should store:

- receive time in continuous ticks or seconds
- target ear
- returned normalized frequency
- intensity
- source chirp identifier or chirp birth tick, if useful for debugging

On each env tick:

1. Clear left/right frequency bins.
2. Process all echo events whose receive time falls in the current tick window.
3. Deposit event intensity into the relevant frequency bin, with optional
   fractional spill into neighboring bins.
4. Add a small configurable noise floor.
5. Apply bounded compression, such as `log1p(k * energy) / log1p(k)`.
6. Append chirp and self-motion metadata.

This produces the desired behavior:

- No chirp means no new echo energy, aside from noise or any intentionally
  modeled lingering sensor state.
- A low-to-high chirp creates a time-coded return pattern.
- Multiple reflectors can overlap naturally in the same tick and frequency
  bin.
- Range must be inferred from echo timing, not from a direct range channel.

## Example: Two-Frequency Chirp and Two Targets

Assume two frequency bins: low and high.

The bat emits a two-slice chirp:

- slice 0: high frequency
- slice 1: low frequency

There are two static targets, one near and one far. With zero Doppler, the
per-tick ear spectrum could look like:

```text
[0, 0]  sound still traveling
[0, 0]  sound still traveling
[0, 1]  near target returns high slice
[1, 1]  near target returns low slice, far target returns high slice
[1, 0]  far target returns low slice
[0, 0]  no active returns
```

This is the intended observation style. It is not a delay-bin representation.
The temporal sequence itself contains the delay/range information.

## Timing and Physics Notes

Echo timing is two-way:

```text
emit position -> reflector -> ear
```

For static reflectors, the approximate return time is:

```text
t_receive = t_emit
          + distance(chirp_origin, reflector) / sound_speed
          + distance(reflector, ear_at_receive) / sound_speed
```

For moving reflectors, such as the bug, the hit time should use predicted
reflector position at the time of impact. A linear-motion approximation is good
enough for the next implementation.

Doppler should be based on the rate of change of the acoustic path length:

```text
doppler_shift ~= -path_length_rate / sound_speed
```

Static walls and obstacles can still have Doppler from bat self-motion. The
moving bug additionally contributes target radial velocity.

Use fractional receive times internally. The env control tick can stay at
`1/60` second while echo events are scheduled at sub-tick times and deposited
into the nearest tick or split across adjacent ticks.

## Chirp Overlap and Memory

Without explicit delay bins, the policy needs temporal memory to infer range.
The observation at a single tick only says what frequency energy is arriving
now. It does not directly say how long ago that sound was emitted unless the
policy remembers the chirp sequence or the env provides reliable chirp-age
metadata.

For the next rung, use one active chirp at a time:

- `chirp_cooldown_ticks >= max_echo_return_ticks`
- include `chirp_age_norm`
- include last chirp start frequency, end frequency, and duration

This keeps return timing interpretable before adding overlapping chirps. Later
curriculum stages can reduce cooldown and allow ambiguity from multiple active
chirps.

## Performance Constraints

The target is high SPS. Avoid raw waveform buffers, convolution, and per-step
FFT.

Use:

- a fixed upper bound on active chirps
- a fixed upper bound on echo events
- static reflector precomputation after reset
- direct frequency-bin deposition
- simple geometric attenuation and ear gain
- first-order reflections only

The expected work per tick should stay near:

```text
active_chirps * chirp_slices * reflectors * ears
```

With small constants, this remains cheap C code and should preserve the spirit
of the current native PufferLib env.

## Implementation Direction

The next implementation should replace current range/Doppler observation
generation with an event queue.

Suggested data structures:

- `ChirpEvent`: emitted chirp metadata, birth time, origin, frequency sweep
- `Reflector`: position, velocity, strength, normal or type
- `EchoEvent`: receive time, ear, frequency, intensity

Suggested tests:

- no chirp produces no echo energy beyond noise
- single static reflector returns at expected two-way travel time
- left and right ears receive slightly different timings/intensities off-axis
- two chirp slices and two reflectors produce the expected overlapping bin
  pattern
- moving bug shifts frequency in the expected Doppler direction
- cooldown prevents ambiguous overlapping chirps in the initial curriculum
- bug echo progress reward only fires when the echo-derived bug path is shorter
  than the previous bug echo path
- static echoes never receive bug echo progress reward

## Non-Goals for the Next Rung

Do not add raw audio synthesis yet.

Do not add an actual FFT dependency yet.

Do not add full wave acoustics.

Do not add multi-bounce reverberation yet.

Do not expose direct range bins if the goal is to force temporal echolocation.

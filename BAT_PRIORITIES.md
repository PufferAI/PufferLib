# Bat Priorities

Current near-term priorities for the Bat PufferLib environment.

## 0. Video capture with audio

- RayLib can render and play audio, but it does not natively encode MP4.
- Preferred path: keep RayLib as the renderer/audio source, capture frames/audio
  during eval, and use `ffmpeg` to mux an MP4.
- A future helper should make this feel like one command, but avoid embedding an
  MP4 encoder in the env.
- Existing GIF capture remains useful for quick silent demos.
- Later render polish: play audible reflection blips in addition to emitted
  chirps. Keep this eval-only. Bug reflections and static wall/obstacle
  reflections should likely use distinguishable volume, timbre, panning, or
  marker sounds so the debug audio stays interpretable.

## 1. Bug-reflection chirp timing penalty

- Replace broad "chirp before all echoes clear" pressure with bug-specific
  timing pressure.
- Penalize a valid chirp if it is emitted before the previous chirp's expected
  bug reflection has returned.
- Scale the penalty by remaining wait fraction, so chirping immediately after a
  prior chirp is worse than chirping shortly before the bug echo arrives.
- Keep the coefficient sweepable through `chirp_overlap_penalty`.
- Do not penalize based on all static wall/obstacle reflections; clutter may
  legitimately require reacquisition chirps.

## 2. Resume performance work

- Use level 7 and level 10 evals as visual sanity checks.
- Focus on harder-level failures where the bat spends chirps before acquiring
  the bug.
- Keep reward shaping minimal and prefer terminal/curriculum/perf pressure where
  possible.

## 3. Prepare the next sweep

- Make sure the next sweep includes any new timing penalty coefficient ranges.
- Sweep `chirp_cooldown_ticks` in a bounded range. Current range is `6..18`.
- Keep `max_chirps_per_episode` fixed at `15` for this sweep so budget does
  not confound timing penalty and cooldown effects.
- Keep sweep ranges bounded so runs cannot become extremely slow from oversized
  policies or excessive env settings.
- Watch `perf`, `base_perf`, `curriculum_perf`, `chirps_emitted`,
  `chirp_overlap_fraction`, `chirp_tempo_ratio`, `collision`, and SPS.

## Priority judgment

The current ordering is sound: the video/audio capture work is useful for demos,
but the bug-reflection timing penalty is more likely to improve level 7/10
performance before the next sweep.

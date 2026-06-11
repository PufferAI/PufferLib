# Bat9 Sweep Analysis

Date: 2026-06-11

This note summarizes the local W&B `bat9` sweep after the timer observation,
sweepable ear directivity, and bug wing sideband changes. It is modeled after
`BAT8_SWEEP_ANALYSIS.md` and should be used before copying a Bat9 candidate into
`config/bat.ini`.

## Future Agent Workflow

Use this workflow when trying a Bat9 candidate.

1. Pick by behavior objective, not only by `env/perf`.
   - Start with `ifbn2epd` for the first balanced retrain/video candidate.
   - Keep `ewgh6l5l` as the best composite scalar anchor.
   - Use `qkwrqhzy` when SPS matters, `sfmk59n1` or `w938us46` for high
     curriculum checks, and `cpx4gj2f` for a 128x5 balanced comparison.
   - Treat `1a2s8uvf` as a low-chirp experiment only; its timeout rate is high.

2. Pull exact hyperparameters from the local W&B config.
   - Use `wandb/run-*-<run_id>/files/config.yaml` or `logs/bat/<run_id>.json`.
   - Copy only concrete values from `vec`, `policy`, `env`, and `train`.
   - Do not copy sweep search-space sections.
   - Keep the run's configured `train.total_timesteps`.
   - Local JSON/YAML files do not always store W&B display names. When the name
     matters, query W&B with the run hash, for example:
     `wandb.Api().run("kinvert-k/bat9/<run_id>").name`.

3. Before each candidate train/eval cycle, run:

   ```bash
   source .venv/bin/activate && ./build.sh bat && bash ocean/bat/tests/run_all.sh
   ```

4. Train with the selected config and no timestep override:

   ```bash
   source .venv/bin/activate && python -m pufferlib.pufferl train bat --train.gpus 1
   ```

   If CUDA is hidden inside Codex, rerun the same command outside the sandbox or
   escalated. Do not switch Bat to CPU.

5. Run fixed-level visual evals before adopting defaults:

   ```bash
   timeout 45s bash -lc 'source .venv/bin/activate && DISPLAY=:0 python -m pufferlib.pufferl eval bat --load-model-path latest --env.curriculum-initial-level 5 --env.curriculum-successes-per-level 1000000'
   timeout 45s bash -lc 'source .venv/bin/activate && DISPLAY=:0 python -m pufferlib.pufferl eval bat --load-model-path latest --env.curriculum-initial-level 10 --env.curriculum-successes-per-level 1000000'
   ```

6. Record the first postable MP4 only after a retrained checkpoint looks clean:

   ```bash
   timeout 45s bash -lc 'source .venv/bin/activate && DISPLAY=:0 python -m pufferlib.pufferl eval bat --load-model-path latest --env.curriculum-initial-level 5 --env.curriculum-successes-per-level 1000000 --env.record-video 1 --env.record-video-fps 30 --env.record-video-seconds 30 --env.record-video-audio 1'
   ```

   Expected output is `recordings/bat_recording.mp4`. Do not commit recordings,
   gifs, local W&B folders, logs, or checkpoint artifacts unless asked.

## Source And Filter

Source data is the local `wandb/` tree in `/home/claude/pathfinder`, filtered to
runs where the W&B metadata/config has `--wandb-project bat9` or
`wandb_project = bat9`.

- Sweep invocation in run metadata:
  `python -m pufferlib.pufferl ... --sweep.gpus 1 --train.gpus 1 --sweep.use-gpu "" --sweep.max-runs 1000 --wandb --wandb-project bat9`
- Git commit in run metadata: `ac61d3bfebb5c24c6a0703c3998940904df0a140`
- Hardware in run metadata: `G240`, `NVIDIA GeForce RTX 5060`
- Bat9 rows with `env/perf` and usable W&B config: `789`
- Rows with `env/perf >= 0.25`: `507`
- Pareto front rows over the selected objectives: `123`

The previous handoff mentioned `773` complete runs. The local tree had grown by
the time this snapshot was frozen; the top-six ordering remained unchanged.

`env/perf` is the composite sweep objective:

```text
perf = base_perf * curriculum_difficulty * chirp_perf
```

This is still not the same as "best visible behavior." High scalar scores can
come from low chirp counts, curriculum progress, or catch rate in different
proportions.

## Bat9 Code Changes

Bat9 differs from Bat8 in three behavior-relevant ways:

- Timer observation: `BAT_OBS_SIZE` is now `41`, and observation slot `40`
  receives normalized elapsed episode time. This should give the policy urgency
  information that Bat8 lacked.
- Ear directivity: `ear_rear_gain`, `ear_front_gain`, and `ear_side_gain` are
  sweepable, and the echo scheduler mixes rear baseline, forward response, and
  left/right side response into per-ear intensity.
- Bug wing sidebands: bug echoes add adjacent frequency-bin sideband energy
  scaled by `bug_wing_sideband_gain`.

Bat9 also logs `env/mean_chirp_bandwidth`, which helps distinguish detection
benefits from policies that merely chirp broadly or noisily.

## Overall Distribution

Across all 789 complete Bat9 rows:

| `env/perf` quantile | value |
| ---: | ---: |
| min | `0.0000` |
| 25% | `0.2023` |
| 50% | `0.2996` |
| 75% | `0.3628` |
| 90% | `0.4014` |
| 95% | `0.4274` |
| 99% | `0.4861` |
| max | `0.5565` |

Filtered high-perf rows (`env/perf >= 0.25`) look like this:

| metric | mean | median | q25 | q75 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `env/perf` | `0.3477` | `0.3459` | `0.3058` | `0.3843` | `0.2505` | `0.5565` |
| `env/base_perf` | `0.9173` | `0.9237` | `0.9028` | `0.9427` | `0.5768` | `0.9741` |
| `env/curriculum_perf` | `0.8004` | `0.8161` | `0.7841` | `0.8350` | `0.4398` | `0.9327` |
| `env/chirp_perf` | `0.4452` | `0.4417` | `0.3936` | `0.4877` | `0.3011` | `0.6888` |
| `env/curriculum_level` | `11.94` | `11.99` | `11.12` | `12.83` | `8.05` | `15.62` |
| `env/chirps_emitted` | `8.33` | `8.38` | `7.69` | `9.11` | `4.67` | `10.50` |
| `env/chirp_overlap_fraction` | `0.1249` | `0.1069` | `0.0510` | `0.1742` | `0.0001` | `0.5243` |
| `env/mean_chirp_bandwidth` | `0.4071` | `0.3750` | `0.3356` | `0.4556` | `0.0010` | `0.9753` |
| `env/timeout` | `0.0023` | `0.0011` | `0.0004` | `0.0025` | `0.0000` | `0.0521` |
| `env/collision` | `0.0804` | `0.0737` | `0.0549` | `0.0962` | `0.0182` | `0.4232` |
| `bad_terminal` | `0.0827` | `0.0763` | `0.0573` | `0.0972` | `0.0259` | `0.4232` |
| `env/episode_length` | `251.18` | `249.34` | `224.93` | `272.49` | `153.72` | `420.04` |
| `SPS` | `1.09M` | `0.99M` | `0.44M` | `1.63M` | `0.40M` | `2.45M` |

## Bat8 To Bat9 Read

This is a qualitative before/after, not a matched statistical test. The Bat8
numbers are from `BAT8_SWEEP_ANALYSIS.md` plus the same local W&B scan for
episode length.

| high-perf metric | Bat8 mean | Bat9 mean | read |
| --- | ---: | ---: | --- |
| `env/perf` | `0.3250` | `0.3477` | Bat9 shifted the upper half upward, though Bat8's single best scalar was slightly higher (`0.5695` vs `0.5565`). |
| `env/base_perf` | `0.8592` | `0.9173` | Clear catch-rate improvement. |
| `env/curriculum_perf` | `0.7583` | `0.8004` | Bat9 reaches harder behavior more consistently. |
| `env/chirp_perf` | `0.4591` | `0.4452` | Slightly worse chirp efficiency; Bat9 spends a bit more chirp budget. |
| `env/curriculum_level` | `11.10` | `11.94` | Curriculum level improved by about `0.85`. |
| `env/chirps_emitted` | `8.12` | `8.33` | Small increase in chirp usage. |
| `env/chirp_overlap_fraction` | `0.1580` | `0.1249` | Overlap improved. |
| `env/timeout` | `0.0020` | `0.0023` | No aggregate timeout win from the timer observation. |
| `env/collision` | `0.1388` | `0.0804` | Large collision reduction. |
| `env/episode_length` | `214.93` | `251.18` | Episodes got longer, so the timer did not simply make policies rush. |
| `SPS` | `1.63M` | `1.09M` | Bat9 is slower, mostly because many good runs are wider models. |

Timer read: the timer observation did not remove timeout/circling risk by
itself. In the top six, `1a2s8uvf` still times out at `0.0512`, `qkwrqhzy` at
`0.0216`, and `ewgh6l5l`/`gli5dke9` have long episodes. Fixed-level visual eval
is still required before declaring a default.

## Top Composite Runs

| run id | role | perf | base | curriculum | chirp perf | level | chirps | timeout | collision | episode len | SPS | model |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rdjj5r21` | silent inbound exploit | `0.8438` | `0.9638` | `0.8450` | `0.9985` | `10.97` | `0.02` | `0.0282` | `0.0079` | `468.48` | `2.62M` | 64x5 |
| `ewgh6l5l` | best composite | `0.5565` | `0.9497` | `0.8392` | `0.6544` | `11.30` | `5.19` | `0.0088` | `0.0414` | `378.95` | `1.13M` | 256x5 |
| `ifbn2epd` | first retrain/video pick | `0.5398` | `0.9639` | `0.8524` | `0.6372` | `11.11` | `5.44` | `0.0051` | `0.0310` | `303.99` | `2.15M` | 64x5 |
| `gli5dke9` | high scalar, slow wide model | `0.5371` | `0.9459` | `0.8363` | `0.6448` | `11.54` | `5.33` | `0.0139` | `0.0402` | `420.04` | `0.46M` | 512x5 |
| `sfmk59n1` | high curriculum top-six | `0.5314` | `0.9285` | `0.9213` | `0.5840` | `13.51` | `6.24` | `0.0014` | `0.0701` | `263.42` | `0.98M` | 256x5 |
| `1a2s8uvf` | lowest chirp top-six, timeout risk | `0.5243` | `0.9048` | `0.7648` | `0.6888` | `10.47` | `4.67` | `0.0512` | `0.0441` | `362.04` | `1.07M` | 256x5 |
| `qkwrqhzy` | fastest top-six | `0.5145` | `0.9227` | `0.8075` | `0.6389` | `11.05` | `5.42` | `0.0216` | `0.0557` | `330.63` | `2.45M` | 64x4 |
| `63dl6lpc` | low-chirp comparison | `0.4975` | `0.9138` | `0.7219` | `0.6864` | `10.03` | `4.70` | `0.0447` | `0.0415` | `379.87` | `1.32M` | 256x4 |
| `cpx4gj2f` | balanced 128x5 candidate | `0.4968` | `0.9522` | `0.8269` | `0.6070` | `10.96` | `5.89` | `0.0119` | `0.0360` | `309.02` | `1.72M` | 128x5 |

## Pursuit-Biased Short-Episode Candidates

This pass looks for high `env/perf` with lower `env/episode_length`, under the
working hypothesis that shorter successful episodes are more likely to be active
pursuit than waiting/intercept behavior. This is only a proxy: very short
episodes can also mean fast collisions, so the best candidates below keep
`base_perf` high and avoid large timeout/collision rates.

Across the current local Bat9 logs, `env/perf >= 0.40` has median episode length
`248.23` and q25 `223.10`. The rows below are the most promising pursuit-biased
visual candidates, with W&B names resolved from `kinvert-k/bat9` on 2026-06-11.

| hash | W&B name | why inspect | perf | episode len | base | level | chirps | timeout | collision | SPS | model |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `o7yrj371` | `vivid-breeze-268` | Best first pursuit candidate: high perf, short episode, low timeout/collision, and fast 64x5 model. | `0.4749` | `221.67` | `0.9323` | `13.21` | `6.29` | `0.0008` | `0.0670` | `1.99M` | 64x5 |
| `wxyb10fq` | `happy-shadow-619` | Highest short-episode candidate under ~230 steps; high level and good SPS. | `0.4846` | `225.07` | `0.9237` | `13.99` | `6.12` | `0.0003` | `0.0760` | `1.70M` | 128x5 |
| `rm3a29ie` | `generous-violet-224` | Highest `perf / episode_length` among the near-0.485 perf group; moderate collision risk. | `0.4847` | `226.97` | `0.9144` | `12.91` | `5.90` | `0.0014` | `0.0842` | `1.00M` | 256x5 |
| `x1ayhg3j` | `clean-pyramid-454` | Strong pursuit-ratio candidate: ~199-step episodes with good base and low timeout. | `0.4458` | `199.11` | `0.9234` | `12.90` | `6.72` | `0.0005` | `0.0761` | `0.99M` | 256x5 |
| `zfxopb9j` | `vocal-snowflake-675` | Similar to `x1ayhg3j`, slightly higher perf and slightly longer episode; collision is higher but not extreme. | `0.4535` | `203.92` | `0.9091` | `13.39` | `6.48` | `0.0013` | `0.0896` | `0.92M` | 256x5 |
| `e4ut00v8` | `gentle-wind-710` | Cleaner terminal profile: zero timeout, low collision, high base, short-ish episode. | `0.4365` | `218.73` | `0.9472` | `13.78` | `7.14` | `0.0000` | `0.0528` | `1.66M` | 128x5 |
| `op9q6evk` | `sparkling-plasma-859` | Low collision and high base; slower 512x5 model, but a useful clean-pursuit comparison. | `0.4301` | `215.24` | `0.9408` | `13.60` | `7.21` | `0.0005` | `0.0586` | `0.44M` | 512x5 |
| `vt8s8kok` | `golden-moon-129` | Good 128x5 speed/behavior balance with sub-200 episode length and low timeout. | `0.4274` | `198.50` | `0.9297` | `13.58` | `7.14` | `0.0002` | `0.0701` | `1.79M` | 128x5 |
| `tuz0bo8d` | `youthful-lake-438` | Short, low-timeout 128x5 backup; lower perf than the rows above. | `0.4211` | `198.97` | `0.9330` | `13.59` | `7.33` | `0.0001` | `0.0669` | `1.46M` | 128x5 |

Risky short-episode rows to treat with caution:

| hash | W&B name | caution | perf | episode len | base | level | chirps | timeout | collision | model |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `d46xhryw` | `hopeful-universe-874` | Very short and high level, but lower base and high collision; may be fast contact/failure rather than clean pursuit. | `0.4608` | `178.85` | `0.8634` | `15.30` | `6.66` | `0.0000` | `0.1366` | 512x5 |
| `l2sg0cpf` | `scarlet-butterfly-228` | Shortest high-perf episode length, but collision is too high for first visual pass. | `0.4573` | `153.72` | `0.8273` | `15.08` | `6.14` | `0.0001` | `0.1726` | 256x5 |

## Distance-Tempo Chirp Candidates

Bat logs `env/far_chirp_rate`, `env/near_chirp_rate`, and
`env/chirp_tempo_ratio`. These are distance-region metrics, not strict
episode-time buckets: "far" means the bat-bug distance is greater than `0.66`
of the start distance, and "near" means less than `0.33`. The tempo ratio is
`near_chirp_rate / far_chirp_rate`, capped at `10`.

This shortlist looks for runs that may chirp sparsely while far from the bug
and chirp faster once close. The strict filter was `perf >= 0.40`,
`base_perf >= 0.90`, `timeout <= 0.01`, and `collision <= 0.10`; rows were then
ranked by high tempo ratio, low far rate, enough near rate, later mean chirp
time, lower chirp count, and scalar perf. This pass used `logs/bat/*.json`,
which currently has `907` Bat9 rows with tempo metrics. Display names were
resolved from W&B (`kinvert-k/bat9`) on 2026-06-11.

| hash | W&B name | why inspect | perf | episode len | chirps | far rate | near rate | tempo ratio | mean chirp tick | timeout | collision | SPS | model |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `x1ayhg3j` | `clean-pyramid-454` | Best first practical eval: short episodes, good perf/base, high near rate, and strong `1.74x` close/far tempo. Level 10 visual review showed a one/few-chirp blind-intercept tactic, not simply continuous close-range chirping. | `0.4458` | `199.11` | `6.72` | `0.0291` | `0.0480` | `1.74` | `0.210` | `0.0005` | `0.0761` | `0.99M` | 256x5 |
| `agdoug04` | `unique-dawn-764` | Best composite sparse-far/fast-near row; far chirp rate is very low, but episodes are long. | `0.4565` | `352.20` | `6.00` | `0.0213` | `0.0298` | `1.80` | `0.362` | `0.0035` | `0.0813` | `2.16M` | 128x5 |
| `ks16xv58` | `ancient-pyramid-609` | Extreme tempo-ratio study: almost no far chirping and very late mean chirp time; slower 512x5 and lower perf. | `0.4036` | `354.60` | `7.29` | `0.0088` | `0.0387` | `4.92` | `0.497` | `0.0060` | `0.0663` | `0.44M` | 512x5 |
| `899idvcg` | `giddy-eon-856` | Balanced candidate with short-ish episodes, low overlap, and clean terminal stats. | `0.4258` | `248.10` | `7.08` | `0.0262` | `0.0405` | `1.57` | `0.260` | `0.0042` | `0.0708` | `1.46M` | 64x5 |
| `m6vsxc7b` | `balmy-glitter-608` | Clean terminal profile and high base; useful if the shorter `x1ayhg3j` behavior is too noisy. | `0.4161` | `251.40` | `7.51` | `0.0266` | `0.0416` | `1.60` | `0.262` | `0.0001` | `0.0520` | `0.91M` | 256x5 |
| `z6mh0t3b` | `zesty-oath-412` | Highest scalar among the cleaner tempo candidates, but the tempo ratio is milder. | `0.4757` | `247.50` | `6.46` | `0.0254` | `0.0313` | `1.28` | `0.250` | `0.0015` | `0.0443` | `1.06M` | 256x5 |
| `63dl6lpc` | `jolly-night-696` | Risky extreme: `9.31x` tempo ratio and only 4.70 chirps, but timeout is high and episodes are long. | `0.4975` | `379.87` | `4.70` | `0.0008` | `0.0375` | `9.31` | `0.629` | `0.0447` | `0.0415` | `1.32M` | 256x4 |
| `gli5dke9` | `efficient-totem-502` | High perf with high tempo ratio, but very long/slow 512x5 and timeout is above the clean filter. | `0.5371` | `420.04` | `5.33` | `0.0126` | `0.0276` | `3.04` | `0.492` | `0.0139` | `0.0402` | `0.46M` | 512x5 |

## Silent Outlier

`rdjj5r21` / `atomic-dragon-816` stands far outside the rest of Bat9. In the
current local logs, it ranks 1st of 930 Bat9 rows by `env/perf` (`0.8438`) while
averaging only `0.022` chirps per episode. The next-lowest-chirp high-perf rows
are around `4.7` chirps. It also has the 3rd-lowest collision rate (`0.0079`)
and top-quartile base success (`0.9638`), but it is extremely slow: episode
length `468.48`, 4th-longest in the local Bat9 set, with timeout `0.0282`.

Metric interpretation: this is not a close-range tempo-chirp policy. Its
`first_chirp_tick_norm` and `mean_chirp_tick_norm` are both about `0.996`, which
mostly means "no chirp happened" under the current logging convention. The high
score comes from combining high success and high curriculum difficulty with
near-perfect `chirp_perf`.

Visual eval after retraining from the exact `rdjj5r21` hyperparameters confirmed
the exploit. The run loaded `checkpoints/bat/1781207491807/0000000034340864.bin`
via `--load-model-path latest`. Per human review, the bat mostly circles, almost
never chirps, and appears to wait for the inbound bug to hit it accidentally.
This explains the very high scalar score and very long episodes: it is not a
usable pursuit policy.

Physics implication: the policy is exploiting the level 8+ inbound bug
curriculum and timer/motion priors rather than echolocation. At inbound levels,
the bug is re-aimed toward the bat every tick with noise and optional lateral
maneuver. A near-silent policy can therefore learn a wait/patrol strategy that
avoids collisions and catches the bug late. The observation does not include
direct bug position: it contains echo bins, chirp state/cooldown, speed, turn
rate, and timer. This run should remain documented as a useful failure case, not
as a default or video candidate.

## Physics Knob Analysis

Spearman correlations across all 789 complete rows:

| pair | rho |
| --- | ---: |
| `perf` vs `chirps_emitted` | `-0.777` |
| `perf` vs `chirp_perf` | `0.776` |
| `perf` vs `far_chirp_rate` | `-0.678` |
| `perf` vs `chirp_overlap_fraction` | `-0.589` |
| `perf` vs `curriculum_perf` | `0.543` |
| `perf` vs `curriculum_level` | `0.508` |
| `perf` vs `bad_terminal` | `-0.443` |
| `perf` vs `base_perf` | `0.443` |
| `perf` vs `collision` | `-0.377` |
| `perf` vs `timeout` | `-0.300` |

Wing sideband read:

- `bug_wing_sideband_gain` has a weak positive relationship with `perf`
  (`rho = 0.158`), stronger with `curriculum_perf` (`rho = 0.220`) and
  `curriculum_level` (`rho = 0.237`).
- Top-quartile sideband gain hit `env/perf >= 0.25` in `69.2%` of rows versus
  `51.0%` in the bottom quartile.
- It does not look like sidebands merely encourage broad noisy chirps:
  correlation with `mean_chirp_bandwidth` is slightly negative (`rho = -0.078`),
  and correlation with `chirps_emitted` is also slightly negative (`rho = -0.064`).

Ear directivity read:

- Raw ear gain effects are weak. `ear_rear_gain` is mildly positive for `perf`
  (`rho = 0.071`), while `ear_front_gain` and `ear_side_gain` are near zero.
- Lower `front_to_rear` ratios look better: bottom-quartile hit rate is `70.7%`
  versus `60.1%` for the top quartile.
- High `ear_side_gain` correlates with worse collision (`rho = 0.175`) and lower
  `base_perf` (`rho = -0.170`), but the hit-rate split is flat. Treat this as a
  weak caution, not a rule.
- `ear_separation_scale` has weak positive `perf` signal and high-perf IQR
  around `1.73..1.99`.

Other sweep reads:

- `reflector_strength` is now beneficial at the higher end. Top quartile hit
  rate is `81.3%` versus `53.6%` in the low quartile, unlike the Bat8 low-strength
  preference.
- `horizon = 64` remains the only reliable setting. The few `128`/`256` rows are
  mostly failures.
- `num_layers = 5` is still the default region; `num_layers = 4` can work for
  speed (`qkwrqhzy`), while shallower models are under-sampled or weak.
- `hidden_size = 128` and `256` have the best hit rates, but the top Pareto run
  is a 64x5 model. Use model size as a speed/behavior tradeoff, not a hard rule.

ExtraTrees feature-importance sanity check ranked these as the top predictors of
`env/perf`: `train.beta1`, `train.clip_coef`, `train.ent_coef`,
`train.vf_clip_coef`, `env.curriculum_successes_per_level`, `env.bat_max_speed`,
`train.prio_beta0`, `env.chirp_cooldown_ticks`, `env.reflector_strength`, and
`policy.hidden_size`. Treat this as nonlinear importance, not causal proof.

## Candidate Shortlist

Display names below were resolved from W&B (`kinvert-k/bat9`) on 2026-06-11.

| hash | W&B name | use when | human / visual notes | analysis notes |
| --- | --- | --- | --- | --- |
| `rdjj5r21` | `atomic-dragon-816` | failure-mode study for silent inbound exploit | Per human visual review after retrain: it mostly circles and almost never chirps, apparently waiting for the inbound bug to accidentally hit the bat. Interesting as a scalar exploit, but not a usable pursuit policy. | Rank 1 local Bat9 scalar outlier: `0.8438` perf, `0.9638` base, `0.8450` curriculum perf, `0.9985` chirp perf, `0.022` chirps, `468.48` episode length, `0.0282` timeout, and `0.0079` collision. Fresh checkpoint `1781207491807` reproduced the scalar profile and loaded `checkpoints/bat/1781207491807/0000000034340864.bin` via `--load-model-path latest`. This points to the inbound bug retargeting policy as a curriculum exploit source. |
| `ifbn2epd` | `super-wind-258` | first retrain and video attempt; behavior-strategy study | Per human visual review: performed poorly overall, but learned a very interesting speed-gated chirp strategy. It flies around most of the time presumably at minimum speed, accelerates presumably to max speed just before chirping, then slows right back down after the chirp. | Chosen first because it had the best Pareto score, `0.5398` perf, `0.9639` base, low collision, low overlap, and 2.15M SPS. The speed-before-chirp pattern is worth preserving as a discovered tactic even if this run is not the final default. |
| `ewgh6l5l` | `distinctive-surf-293` | current default candidate | Per human visual review: normal/low-level eval looked erratic, often spun in place, and every watched run appeared to time out. Retesting fixed level 10 showed the desired harder-level tactic: chirp to infer where the bug is going, move to that future path, then circle/wait there until the bug reaches the bat. A fresh 2026-06-11 retrain and level 10 eval confirmed this is the behavior we want as the current default. | Top non-silent Bat9 scalar profile: `0.5565` perf, `0.9497` base, `0.8392` curriculum perf, `0.6544` chirp perf, level `11.30`, `5.19` chirps, episode length `378.95`, timeout `0.0088`, and collision `0.0414`. Fresh checkpoint `1781208977022` reproduced the scalar profile and loaded `checkpoints/bat/1781208977022/0000000033554432.bin` via `--load-model-path latest`. |
| `qkwrqhzy` | `earnest-galaxy-621` | behavior-strategy study; not default yet | Per human visual review: weak overall, about 25% wins in watched eval, and poor at levels 0-1. Surprisingly more interesting on harder eval: it often chirps enough to infer where the bug is going, moves ahead of the bug, then circles until the bug reaches the bat. | 64x4 speed candidate with `0.5145` sweep perf and strong SPS. Fresh checkpoint `1781200339036` reproduced the scalar profile, but visual robustness was too low. Preserve the intercept-and-wait tactic as a discovered behavior, but do not use as default/video pick without fixing low-level competence. |
| `sfmk59n1` | `cool-snowflake-484` | sparse-chirp memory/navigation behavior study | Per human visual review at fixed level 10: very interesting deliberate sparse-chirp strategy. It can fly for long periods without chirping, apparently remembering what it saw from an earlier chirp, navigating around, looking, then chirping again later. It does not look like the previous aimless circling/intercept pattern. Visual perf did not look obviously high, but the behavior is important. | Chosen next because it is the strongest not-yet-watched high-curriculum candidate: `0.5314` perf, `0.9213` curriculum perf, level `13.51`, only `0.0014` timeout, `6.24` chirps, and very low overlap (`0.0070`). It is not another low-chirp/intercept candidate; it trades a moderate collision rate (`0.0701`) and 256x5 speed cost (`0.98M` SPS) for cleaner curriculum progress. Physics read: max sideband gain (`0.25`), high directivity gains (`rear 0.30`, `front 0.678`, `side 0.520`; front/rear `2.26`), slow max speed (`12.91`) with high turn rate (`9.10`). |
| `o7yrj371` | `vivid-breeze-268` | pursuit-biased/default candidate | Per human visual review at fixed level 10: performs well and actively pursues the bugs. This is the first watched high-perf, short-episode candidate that visually supports the pursuit hypothesis rather than the previously observed intercept-and-wait behavior. | Chosen from the high-perf, short-episode screen: `0.4749` perf, `221.67` episode length, `0.9323` base, level `13.21`, near-zero timeout (`0.0008`), moderate collision (`0.0670`), and fast 64x5 throughput (`1.99M` SPS). Fresh checkpoint `1781204865675` was trained from the exact `o7yrj371` hyperparameters and eval loaded `checkpoints/bat/1781204865675/0000000032768000.bin` via `--load-model-path latest`. |
| `x1ayhg3j` | `clean-pyramid-454` | blind-map behavior evidence | Per human visual review after fresh retrain: it does initial chirps, builds an apparent mental map of where the bug is going, then deliberately flies where it believes is right, often continuing blindly after the map is made. Interesting behavior, but not the selected default because the desired current default is the stronger wait-at-predicted-path tactic from `ewgh6l5l`. | Fresh checkpoint `1781208510323` reproduced the original scalar profile: `0.446` perf, `0.923` base, `0.821` curriculum perf, `199.11` episode length, `6.72` chirps, `1.735` tempo ratio, near-zero timeout, and collision `0.076`. Preserve as behavior evidence and comparison point. |
| `cpx4gj2f` | `sleek-smoke-681` | balanced 128x5 comparison | Needs visual review; useful if 64-wide `super-wind-258` looks brittle. | `0.9522` base, low bad terminal `0.0478`, 1.72M SPS; not Pareto-front by the selected objective mix. |
| `1a2s8uvf` | `good-valley-684` | low-chirp experiment | Needs visual review specifically for timeout behavior. | Best top-six chirp perf (`0.6888`) and 4.67 chirps, but timeout is high at `0.0512`; not a default without strong visual evidence. |
| `w938us46` | `fanciful-shape-202` | high-level stress check | Needs visual review; use for stress behavior rather than first video. | Highest level (`15.62`) while still `0.4501` perf; collision `0.1285`. |

## Eval Notes And Video Pick

Human visual note for `ifbn2epd` / `super-wind-258`: it performed poorly overall,
but learned a notable speed-gated chirp tactic. It appears to cruise at minimum
speed, accelerate sharply just before a chirp, then slow back down after chirping.
This is important behavior evidence and should be preserved even if the run is
not adopted as a default.

Human visual note for `ewgh6l5l` / `distinctive-surf-293`: after retraining from
the sweep config into checkpoint `1781199673401` and running normal/low-level
eval, the policy looked erratic, often spun in place, and every watched run
appeared to time out. A later fixed level 10 retest using `--load-model-path
latest` showed the desired harder-level behavior: it chirps to infer where the
bug is going, moves onto that future path, then circles/waits there until the bug
reaches the bat. This is the same broad intercept-and-wait tactic later observed
in `qkwrqhzy`, but `ewgh6l5l` has the stronger scalar profile.

Fresh default retrain on 2026-06-11 set `config/bat.ini` to the exact concrete
`ewgh6l5l` hyperparameters and produced checkpoint
`checkpoints/bat/1781208977022/0000000033554432.bin`. The scalar profile again
matched the sweep: `0.556` perf, `0.950` base, `0.839` curriculum perf, level
`11.298`, `5.191` chirps, `378.948` episode length, timeout `0.009`, collision
`0.041`, and chirp tempo ratio `0.134`. Human level-10 visual review confirmed
this is the intended current default: it chooses a place on the predicted bug
path and waits/intercepts there.

Human visual note for `o7yrj371` / `vivid-breeze-268`: after training from the
exact sweep hyperparameters into checkpoint `1781204865675`, fixed level 10 eval
loaded with `--load-model-path latest` showed good performance and active bug
pursuit. This is the cleanest visual support so far for the short-episode screen:
it looked like it chased the bugs rather than mainly waiting on a predicted
intercept point.

Human visual note for `x1ayhg3j` / `clean-pyramid-454`: after training from the
exact sweep hyperparameters into checkpoint `1781206232504`, normal eval and
fixed level 10 eval loaded `checkpoints/bat/1781206232504/0000000035651584.bin`
with `--load-model-path latest`. Scalar profile reproduced the sweep pattern:
about `0.446` perf, `199` episode length, `6.72` chirps, far chirp rate `0.029`,
near chirp rate `0.048`, and tempo ratio about `1.74`. Level 10 visual review
showed that this should not be interpreted as simply chirping more continuously
when close. It sometimes chirps, builds an apparent internal estimate of the bug
trajectory/map, then flies an intercept course with no more chirps. It works
roughly half the time in watched attempts and looks like a confident blind
intercept strategy.

Human visual note for `qkwrqhzy` / `earnest-galaxy-621`: after retraining from
the sweep config into checkpoint `1781200339036`, normal eval and fixed level 10
eval showed weak overall win rate, roughly 25% in the watched sample, and poor
behavior at levels 0-1. The run nevertheless learned a notable harder-level
strategy: chirp enough to infer the bug trajectory, position itself ahead of the
bug, then circle/intercept until the bug reaches the bat. This is valuable
behavior evidence but not a default-quality policy.

Human visual note for `sfmk59n1` / `cool-snowflake-484`: after setting
`config/bat.ini` to the exact concrete `sfmk59n1` hyperparameters, the policy was
trained into checkpoint `checkpoints/bat/1781203704964/0000000036175872.bin`.
The fresh run reproduced the sweep-scale scalar profile: final `env/perf` about
`0.531`, `base_perf` about `0.929`, `curriculum_perf` about `0.921`, curriculum
level `13.507`, `6.24` chirps, timeout about `0.001`, and collision about
`0.070`. Fixed level 10 visual review showed an important sparse-chirp
memory/navigation tactic: it flies deliberately for long periods without
chirping, apparently using information remembered from an earlier chirp,
navigates around while looking, then chirps again later. It did not look like the
previous aimless circling/intercept behavior. Visual performance did not look
obviously high, but this behavior should be preserved as evidence of memory-like
navigation under limited chirping.

Recorded MP4 artifact: `recordings/bat_recording.mp4`. This is a 30.0 second,
640x640, 30 fps H.264 MP4 with AAC audio, recorded from `ewgh6l5l` /
`distinctive-surf-293` at fixed level 10 using `--load-model-path latest`. To
make `latest` resolve to this candidate after later runs had newer checkpoints,
the `ewgh6l5l` checkpoint was copied non-destructively to
`checkpoints/bat/ewgh6l5l-latest-eval/0000000033554432.bin`. The video is a
behavior-evidence artifact for the trajectory-prediction/intercept tactic, not a
claim that this run is default-quality.

Current default candidate is `ewgh6l5l` / `distinctive-surf-293`. It is not the
most robust low-level visual policy, but it is the best current match for the
desired behavior: chirp enough to infer the future bug path, move to that path,
then wait/intercept there. `qkwrqhzy` preserves a similar tactic but looked weaker
overall. `x1ayhg3j` remains useful behavior evidence for blind-map navigation,
and `sfmk59n1` remains important behavior evidence for sparse-chirp
memory/navigation.

`config/bat.ini` is now intentionally set to `ewgh6l5l` for the current default
candidate. This decision is based on fresh retrain plus level 10 visual eval, not
scalar rank alone.

## Recommended Next Defaults

Current default source:

| parameter | recommendation |
| --- | --- |
| candidate source | `ewgh6l5l` / `distinctive-surf-293` |
| `policy.hidden_size` | `256` |
| `policy.num_layers` | `5` |
| `train.horizon` | `64` |
| `vec.num_buffers` | `4` |
| `env.bug_wing_sideband_gain` | `0.19056934455600955` |
| `env.ear_rear_gain` | `0.22038613968607276` |
| `env.ear_front_gain` | `0.6419214149115183` |
| `env.ear_side_gain` | `0.28043867572747055` |
| `env.ear_separation_scale` | `2.0` |
| `env.reflector_strength` | `0.6` |
| `env.chirp_cooldown_ticks` | `11` |
| `env.curriculum_successes_per_level` | `4` |
| `env.curriculum_bug_distance_step` | `2.0` |

Keep `--sweep.use-gpu ""` for future Bat9 sweep continuation so Protein stays
off GPU while training uses `--train.gpus 1`.

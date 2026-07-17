# Guerrilla Checkers models

The standalone client and tournament preserve the original Puffer NN and two
native 5c checkpoints. The recommended training seed is the genuine two-sided
model.

| Checkpoint | Purpose | SHA-256 |
| --- | --- | --- |
| `guerrillacheckers_weights.bin` | Original Puffer NN | `f58615069b9a0e50105dd54b4729d366d602fe14b957a0042e4dc88089120398` |
| `guerrillacheckers_5c_selfplay.bin` | Earlier one-sided 5c model | `863c473942c4681e187e86fb21769f4d6bc8cd89450b2ef8547ac561d39b0000` |
| `guerrillacheckers_5c_twosided.bin` | Recommended two-sided 5c model | `9092cb9169c7ab702f31e8f4d9227e8d51d1dc3fd690de4c7ef0498893c4c60b` |

The original checkpoint is 587,264 bytes. Native 5c checkpoints are 587,808
bytes because they include the value-head row and native alignment padding.
The standalone evaluator detects both layouts directly and never rewrites the
original asset.

## Train

The default config uses genuine alternating self-play with randomized sides:
slot 0 is the trainable policy, slot 1 is the current or historical opponent,
and an all-zero action mask identifies whichever slot is waiting for its turn.
Twenty percent of environments use the historical bank. Opponent swaps happen
only after all tagged games reach an episode boundary.

```sh
./puffer train guerrillacheckers \
  base.load_model_path=resources/guerrillacheckers/guerrillacheckers_5c_twosided.bin
```

`policy.use_bias` preserves the original checkpoint architecture.
`policy.turn_based` freezes recurrent state and excludes waiting timesteps from
GAE and PPO. Both switches default off for other environments.

## Evaluate

Build the standalone client and compare a native candidate against the original
Puffer NN from both sides:

```sh
./build.sh guerrillacheckers --fast
./guerrillacheckers --compare-candidate 1000 \
  resources/guerrillacheckers/guerrillacheckers_5c_twosided.bin
```

The two-sided model's 1,000-game reciprocal results against the original were
930-70 as Guerrilla and 918-82 as COIN. Against the earlier one-sided model,
it scored 934-66 as Guerrilla and 937-63 as COIN.

The latest six-bot field reuses the recorded old-vs-old cells and adds the 11
cells involving the two-sided model. Each cell contains 100 games and is shown
as Guerrilla wins - COIN wins.

| Guerrilla / COIN | Random | Greedy | Puffer NN | MCTS 2K | MCTS 10K | Puffer 5c |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 8-92 | 0-100 | 0-100 | 0-100 | 0-100 | 0-100 |
| Greedy | 98-2 | 33-67 | 1-99 | 0-100 | 0-100 | 0-100 |
| Puffer NN | 96-4 | 99-1 | 15-85 | 12-88 | 9-91 | 4-96 |
| MCTS 2K | 99-1 | 100-0 | 50-50 | 30-70 | 11-89 | 47-53 |
| MCTS 10K | 100-0 | 100-0 | 84-16 | 81-19 | 65-35 | 82-18 |
| Puffer 5c | 96-4 | 100-0 | 92-8 | 13-87 | 2-98 | 30-70 |

Across both roles, the two-sided model scores 770-430 in this field: 333-267
as Guerrilla and 437-163 as COIN. The side-specific Bradley-Terry fit places it
at Elo 1768 as Guerrilla and 1878 as COIN.

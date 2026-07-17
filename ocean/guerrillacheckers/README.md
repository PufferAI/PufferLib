# Guerrilla Checkers

The standalone client and tournament preserve the original Puffer 40 as the
baseline for new standard 5c self-play models.

| Checkpoint | Purpose | SHA-256 |
| --- | --- | --- |
| `guerrillacheckers_weights.bin` | Original Puffer 40 | `f58615069b9a0e50105dd54b4729d366d602fe14b957a0042e4dc88089120398` |

Puffer 40 uses the legacy 587,264-byte biased layout. Newly trained standard
5c checkpoints use the 586,240-byte bias-free layout with a value-head row.
The standalone evaluator detects both formats directly.

## Train

The canonical `config/guerrillacheckers.ini` uses genuine alternating self-play
with randomized sides: slot 0 is the trainable policy, slot 1 is the current or
historical opponent, and the inactive slot receives one deterministic pass
action. Both recurrent states observe every game step, following the existing
Ocean Chess self-play convention. Twenty percent of environments use the
historical bank. Opponent swaps happen only after all tagged games reach an
episode boundary.

```sh
./puffer train guerrillacheckers
```

## Evaluate

Build the standalone client and compare a native candidate against the original
Puffer 40 from both sides:

```sh
./build.sh guerrillacheckers --fast
./guerrillacheckers --compare-candidate 1000 \
  .runtime/checkpoints/guerrillacheckers/<run>/<checkpoint>.bin
```

The recorded baseline field contains 100 games per cell, shown as Guerrilla
wins - COIN wins.

| Guerrilla / COIN | Random | Greedy | Puffer 40 | MCTS 2K | MCTS 10K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 8-92 | 0-100 | 0-100 | 0-100 | 0-100 |
| Greedy | 98-2 | 33-67 | 1-99 | 0-100 | 0-100 |
| Puffer 40 | 96-4 | 99-1 | 15-85 | 12-88 | 9-91 |
| MCTS 2K | 99-1 | 100-0 | 50-50 | 30-70 | 11-89 |
| MCTS 10K | 100-0 | 100-0 | 84-16 | 81-19 | 65-35 |

The side-specific Bradley-Terry fit is anchored at Elo 1500 across all ten
side-specific entries:

| Bot | Guerrilla Elo | COIN Elo |
| --- | ---: | ---: |
| Random | 338 | 727 |
| Greedy | 1120 | 1189 |
| Puffer 40 | 1636 | 1879 |
| MCTS 2K | 1834 | 1969 |
| MCTS 10K | 2200 | 2108 |

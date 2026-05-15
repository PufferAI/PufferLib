# Backgammon

A backgammon environment for PufferLib. The agent plays as White against a configurable opponent (Black).

## Rules

Standard backgammon rules: 15 checkers per player, roll two dice, move checkers toward your home board, bear them off to win. You can hit opponent blots (single checkers) to send them to the bar. First to bear off all 15 checkers wins.

White moves from point 24 → 1 (home board is 1-6).
Black moves from point 1 → 24 (home board is 19-24).

## Observations (35 floats)

- Points 1-24: checker counts normalized by 15 (positive = white, negative = black)
- Bar counts for white and black
- Borne-off counts for white and black  
- 4 dice values (0 if used)
- Current player
- Can-bear-off flags for both players

## Actions (104 discrete)

`action = source * 4 + die_index`

- `source`: 0 = entering from bar, 1-24 = board points, 25 = bear off
- `die_index`: which die to use (0-3, since doubles give 4 moves)

Invalid moves get a small penalty (-0.1) and are skipped.

## Rewards

- Win: +1.0
- Lose: -1.0
- Bear off a checker: +0.05
- Hit opponent: +0.02
- Invalid move: -0.1

## Opponent

The opponent difficulty is controlled by `OPPONENT_RANDOM_PROB` in `backgammon.h`:
- 1.0 = fully random (easiest)
- 0.0 = greedy heuristic (harder)

For training, start with a weak opponent and gradually decrease randomness as the agent improves.

## Training

```bash
python -m pufferlib.pufferl train puffer_backgammon   --vec.num-envs 64   --env.num-envs 256   --train.batch-size 1048576   --train.bptt-horizon 64   --train.total-timesteps 500_000_000 --train.learning-rate 0.001
```

Against a random opponent, expect >50% win rate after a few minutes of training.


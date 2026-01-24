# BossFight (PufferLib Ocean)

BossFight is a simple 2D boss-fight reinforcement learning environment.

The boss currently has **one attack**: a circular **AOE burst** and cycles between 4 states.
Player (agent) has to defeat the boss by attacking and avoiding AoE attacks by dodging (has i-frames).
All hitboxes are circles (collision = circles overlap).

## Game rules

- **Arena:** square `[-ARENA_HALF_SIZE, ARENA_HALF_SIZE]^2` (default `5.0`)
- **Boss:** stationary at `(0, 0)`
- **Episode ends on:**
  - win: boss HP reaches 0
  - loss: player HP reaches 0
  - timeout: `EPISODE_LENGTH` steps

### Boss attack cycle

The boss cycles through:

`IDLE (BOSS_IDLE_TICKS) -> WINDUP (BOSS_WINDUP_TICKS) -> ACTIVE (BOSS_ACTIVE_TICKS) -> RECOVERY (BOSS_RECOVERY_TICKS) -> ...`

During **ACTIVE**, the boss deals damage if the player overlaps the AOE circle.

### Player mechanics

- **Move** (only while idling): 4 directional movement at `PLAYER_SPEED_PER_TICK`
- **Attack**: melee hit if within `PLAYER_ATTACK_RADIUS` (locks the player for `PLAYER_ATTACK_TICKS`)
- **Dodge**:
  - lasts `PLAYER_DODGE_TICKS` and automatically moves the player directly **away from the boss** at `PLAYER_DODGE_SPEED_PER_TICK`
  - the first `PLAYER_IFRAME_TICKS` are i-frames
  - the boss AOE lasts longer than the i-frame window, so “dodge in place” isn’t sufficient -- you must **exit the AOE**
  - after dodge ends, `PLAYER_DODGE_COOLDOWN` ticks must pass before dodging again

## Action space

`Discrete(7)`:

|  id | action     |
| --: | ---------- |
|   0 | idle       |
|   1 | move up    |
|   2 | move down  |
|   3 | move left  |
|   4 | move right |
|   5 | dodge      |
|   6 | attack     |

## Observation space

`Box(shape=(13,), dtype=float32)` (see `update_observations` in `boss_fight.h`):

| idx | meaning                                                |
| --: | ------------------------------------------------------ |
|   0 | `boss_x - player_x`                                    |
|   1 | `boss_y - player_y`                                    |
|   2 | `player_x`                                             |
|   3 | `player_y`                                             |
|   4 | `boss_x`                                               |
|   5 | `boss_y`                                               |
|   6 | `player_hp`                                            |
|   7 | `boss_hp`                                              |
|   8 | `player_state` (`0=idle, 1=dodge, 2=attack`)           |
|   9 | `player_dodge_cooldown`                                |
|  10 | `player_state_ticks` (remaining)                       |
|  11 | `boss_state` (`0=idle, 1=windup, 2=attack, 3=recover`) |
|  12 | `boss_phase_ticks` (remaining)                         |

## Rewards (defaults)

All reward constants are in `boss_fight.h`:

- **Per-step:** `REWARD_TICK`
- **Shaping:** `REWARD_APPROACH * (prev_distance - distance)`
- **Events:**
  - `REWARD_PLAYER_HIT_BOSS`
  - `REWARD_BOSS_HIT_PLAYER`
  - `REWARD_DODGE_SUCCESS`
  - `REWARD_HIT_WALL`
- **Terminal:** `REWARD_KILL_BOSS`, `REWARD_PLAYER_DIED`, `REWARD_TIMEOUT`

**Dodge success reward** is only paid when:

1. you **start** a dodge while inside the AOE during the boss danger window (**WINDUP** or **ACTIVE**), and
2. you **exit** the AOE before the danger window ends.

## Rendering / manual play

- Rendering uses **Raylib**. `BossFight.render()` opens a window and draws the player/boss circles + hit radii.
- A tiny standalone debug harness lives in `boss_fight.c`:
  - Hold `Left Shift` for manual controls: `WASD` move, `Space` dodge, `J` attack
  - Without `Left Shift` it takes random actions

## Files

- `boss_fight.h`: core environment logic (`c_reset`, `c_step`, `c_render`)
- `binding.c`: CPython extension glue (uses `pufferlib/ocean/env_binding.h`)
- `boss_fight.py`: PufferLib wrapper (`PufferEnv`) + vectorized stepping
- `pufferlib/config/boss_fight.ini`: default training config for `puffer train puffer_boss_fight`

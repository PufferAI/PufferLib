# Light Thief Environment

A 2D Reinforcement Learning environment for PufferLib.

## Concept
The agent must collect loot in a dark room illuminated by moving searchlights.
**Twist**: Loot is only revealed when lit, but can ONLY be collected when the agent is in total darkness.

## Structure
- `light_thief.c`: C backend handling physics and visibility logic.
- `light_thief_env.py`: Gymnasium/PufferLib wrapper.


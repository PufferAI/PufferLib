# Light Thief Environment

A 2D Reinforcement Learning environment for PufferLib.

## Concept
The agent must collect loot in a dark room illuminated by moving searchlights.
**Twist**: Loot is only revealed when lit, but can ONLY be collected when the agent is in total darkness.

## Structure
- `light_thief.h`: C backend (env logic + rendering).
- `light_thief.c`: Standalone human-playable demo.
- `light_thief.py`: Gymnasium/PufferLib wrapper.


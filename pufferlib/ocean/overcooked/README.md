# Overcooked Environment

A multi-agent cooking coordination environment where agents cooperate to prepare and serve onion soup. Here we describe the rewards system and observation space.

## File Structure

```
overcooked/
├── overcooked.h           # Main entry point (init, reset, step, close)
├── overcooked_types.h     # Constants, enums, and struct definitions
├── overcooked_items.h     # Item and cooking pot management
├── overcooked_obs.h       # Observation computation
├── overcooked_logic.h     # Game logic (interaction, movement, cooking)
├── overcooked_render.h    # Rendering and texture management
├── binding.c              # Python bindings
└── overcooked.py          # Python environment wrapper
```

## Observation Space

**39-dimensional vector per agent** — *see [compute_observations](overcooked_obs.h#L68)*

### Player Features (34 dims)
- **Orientation** (4): One-hot encoding of facing direction — [overcooked_obs.h:92](overcooked_obs.h#L92)
- **Held Object** (4): One-hot encoding (onion, soup, dish, empty) — [overcooked_obs.h:96-106](overcooked_obs.h#L96-L106)
- **Proximity Features** (12): Normalized (dx, dy) to nearest — [overcooked_obs.h:108-139](overcooked_obs.h#L108-L139):
  - Onion source (ingredient box)
  - Dish source (plate box)
  - Plated soup on counter
  - Serving area
  - Empty counter
  - Pot (stove)
- **Nearest Soup Ingredients** (2): Onion/tomato counts in nearest plated soup or held soup (normalized) — [overcooked_obs.h:141-167](overcooked_obs.h#L141-L167)
- **Pot Soup Ingredients** (2): Onion/tomato counts in nearest pot (normalized) — [overcooked_obs.h:169-192](overcooked_obs.h#L169-L192)
- **Pot Existence** (1): Binary flag for reachable pot — [overcooked_obs.h:195](overcooked_obs.h#L195)
- **Pot State** (4): Binary flags (empty, full, cooking, ready) — [overcooked_obs.h:198-205](overcooked_obs.h#L198-L205)
- **Cooking Time** (1): Remaining cook time (normalized) — [overcooked_obs.h:208-213](overcooked_obs.h#L208-L213)
- **Wall Detection** (4): Binary flags for walls/obstacles (up, down, left, right) — [overcooked_obs.h:215-225](overcooked_obs.h#L215-L225)

### Spatial Features (4 dims)
- **Teammate Relative Position** (2): Normalized (dx, dy) to other agent — [overcooked_obs.h:228-238](overcooked_obs.h#L228-L238)
- **Absolute Position** (2): Normalized (x, y) coordinates — [overcooked_obs.h:241-242](overcooked_obs.h#L241-L242)

### Context (1 dim)
- **Reward** (1): Current step reward — [overcooked_obs.h:245](overcooked_obs.h#L245)

## Action Space

**6 discrete actions** — *see [c_step](overcooked.h#L66)*
- 0: No-op — [ACTION_NOOP](overcooked_types.h#L38)
- 1: Move up — [ACTION_UP](overcooked_types.h#L39)
- 2: Move down — [ACTION_DOWN](overcooked_types.h#L40)
- 3: Move left — [ACTION_LEFT](overcooked_types.h#L41)
- 4: Move right — [ACTION_RIGHT](overcooked_types.h#L42)
- 5: Interact (pick up/place items, use equipment) — [ACTION_INTERACT](overcooked_types.h#L43)

## Reward System

*See [evaluate_dish_served](overcooked_logic.h#L171) and [handle_interaction](overcooked_logic.h#L48)*

### Main Rewards
- **Correct dish served** (3 onions): +20.0 (shared), +5.0 (server bonus) — [overcooked_logic.h:181-184](overcooked_logic.h#L181-L184)
- **Wrong dish served** (incorrect recipe): +0.1 (shared) — [overcooked_logic.h:195-198](overcooked_logic.h#L195-L198)
- **Step penalty**: Configurable (default: 0.0) — [overcooked.h:69](overcooked.h#L69)

### Intermediate Rewards
- **Add onion to pot**: +0.1 — [overcooked_logic.h:75](overcooked_logic.h#L75)
- **Start cooking** (3 onions in pot): +0.1 — [overcooked_logic.h:88](overcooked_logic.h#L88)
- **Plate cooked soup**: +0.1 — [overcooked_logic.h:101](overcooked_logic.h#L101)

## Recipe

The correct recipe requires **exactly 3 onions** in the soup. Agents must:
1. Pick up onions from ingredient boxes
2. Add 3 onions to a pot
3. Start cooking (interact with pot when empty-handed)
4. Wait for soup to cook (20 steps)
5. Pick up a plate from plate box
6. Plate the cooked soup (interact with pot while holding plate)
7. Deliver plated soup to serving area

## Game Constants

- **Cooking time**: 20 steps — [COOKING_TIME](overcooked_types.h#L32)
- **Max ingredients per pot**: 3 — [MAX_INGREDIENTS](overcooked_types.h#L33)
- **Grid size**: 5x5 (default) — [CRAMPED_ROOM](overcooked_types.h#L187)
- **Max episode steps**: 400 (default) — [overcooked.py:12](overcooked.py#L12)

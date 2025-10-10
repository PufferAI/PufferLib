# Overcooked Environment

A multi-agent cooking coordination environment where agents cooperate to prepare and serve onion soup. Here we describe the rewards system and observation space.

## Observation Space

**39-dimensional vector per agent** — *see [compute_observations](overcooked.h#L281)*

### Player Features (34 dims)
- **Orientation** (4): One-hot encoding of facing direction — [overcooked.h:305](overcooked.h#L305)
- **Held Object** (4): One-hot encoding (onion, soup, dish, empty) — [overcooked.h:309-319](overcooked.h#L309-L319)
- **Proximity Features** (12): Normalized (dx, dy) to nearest — [overcooked.h:322-352](overcooked.h#L322-L352):
  - Onion source (ingredient box)
  - Dish source (plate box)
  - Plated soup on counter
  - Serving area
  - Empty counter
  - Pot (stove)
- **Nearest Soup Ingredients** (2): Onion/tomato counts in nearest plated soup or held soup (normalized) — [overcooked.h:356-380](overcooked.h#L356-L380)
- **Pot Soup Ingredients** (2): Onion/tomato counts in nearest pot (normalized) — [overcooked.h:382-405](overcooked.h#L382-L405)
- **Pot Existence** (1): Binary flag for reachable pot — [overcooked.h:408](overcooked.h#L408)
- **Pot State** (4): Binary flags (empty, full, cooking, ready) — [overcooked.h:410-418](overcooked.h#L410-L418)
- **Cooking Time** (1): Remaining cook time (normalized) — [overcooked.h:420-426](overcooked.h#L420-L426)
- **Wall Detection** (4): Binary flags for walls/obstacles (up, down, left, right) — [overcooked.h:428-438](overcooked.h#L428-L438)

### Spatial Features (4 dims)
- **Teammate Relative Position** (2): Normalized (dx, dy) to other agent — [overcooked.h:440-451](overcooked.h#L440-L451)
- **Absolute Position** (2): Normalized (x, y) coordinates — [overcooked.h:453-455](overcooked.h#L453-L455)

### Context (1 dim)
- **Reward** (1): Current step reward — [overcooked.h:458](overcooked.h#L458)

## Action Space

**6 discrete actions** — *see [c_step](overcooked.h#L804)*
- 0: No-op — [ACTION_NOOP](overcooked.h#L38)
- 1: Move up — [ACTION_UP](overcooked.h#L39)
- 2: Move down — [ACTION_DOWN](overcooked.h#L40)
- 3: Move left — [ACTION_LEFT](overcooked.h#L41)
- 4: Move right — [ACTION_RIGHT](overcooked.h#L42)
- 5: Interact (pick up/place items, use equipment) — [ACTION_INTERACT](overcooked.h#L43)

## Reward System

*See [evaluate_dish_served](overcooked.h#L720) and [handle_interaction](overcooked.h#L467)*

### Main Rewards
- **Correct dish served** (3 onions): +20.0 (shared), +5.0 (server bonus) — [overcooked.h:732-735](overcooked.h#L732-L735)
- **Wrong dish served** (incorrect recipe): +0.1 (shared) — [overcooked.h:741-745](overcooked.h#L741-L745)
- **Step penalty**: Configurable (default: 0.0) — [overcooked.h:807](overcooked.h#L807)

### Intermediate Rewards
- **Add onion to pot**: +0.1 — [overcooked.h:494](overcooked.h#L494)
- **Start cooking** (3 onions in pot): +0.1 — [overcooked.h:507](overcooked.h#L507)
- **Plate cooked soup**: +0.1 — [overcooked.h:520](overcooked.h#L520)

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

- **Cooking time**: 20 steps — [COOKING_TIME](overcooked.h#L32)
- **Max ingredients per pot**: 3 — [MAX_INGREDIENTS](overcooked.h#L33)
- **Grid size**: 5×5 (default) — [CRAMPED_ROOM](overcooked.h#L186)
- **Max episode steps**: 400 (default) — [overcooked.py:12](overcooked.py#L12)
